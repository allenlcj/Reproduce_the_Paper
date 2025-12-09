# src/sac_agent.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import random
from collections import deque, namedtuple
from src.common.config import *

# --- 1. SAC 网络架构 (参考 Table 4) ---
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, ACTOR_FC_UNITS),
            nn.ReLU(),
            nn.Linear(ACTOR_FC_UNITS, ACTOR_FC_UNITS),
            nn.ReLU(),
            # Table 4 for Actor also lists two "Fully connected 256", but let's check closely.
            # Image shows:
            # Actor:
            #   Input
            #   Fully connected 256
            #   Fully connected 256
            #   Output
            # This implies 2 hidden layers. My code had 2 hidden layers.
            # Wait, let me look at the image again.
            # Actor rows: Input -> FC 256 -> FC 256 -> Output. (2 hidden layers)
            # Critic rows: Input -> FC 256 -> FC 256 -> FC 256 -> Output. (3 hidden layers)
            # So Actor is fine. I will ONLY change Critic.
            nn.Linear(ACTOR_FC_UNITS, action_dim) 
        )
        self.to(DEVICE)

    def forward(self, s):
        logits = self.net(s)
        prob = torch.softmax(logits, dim=-1)
        log_prob = torch.log_softmax(logits, dim=-1)
        return prob, log_prob

    def sample_action(self, s):
        prob, log_prob_all = self.forward(s)
        m = Categorical(prob)
        action_idx = m.sample()
        log_prob = m.log_prob(action_idx)
        entropy = m.entropy()
        return action_idx.item(), log_prob, entropy

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, CRITIC_FC_UNITS),
            nn.ReLU(),
            nn.Linear(CRITIC_FC_UNITS, CRITIC_FC_UNITS),
            nn.ReLU(),
            nn.Linear(CRITIC_FC_UNITS, CRITIC_FC_UNITS), # Added layer to match Table 4 (3x 256 units)
            nn.ReLU(),
            nn.Linear(CRITIC_FC_UNITS, action_dim) 
        )
        self.to(DEVICE)

    def forward(self, s):
        return self.net(s)

# --- 2. 经验回放缓冲区 ---
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size):
        idxs = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in idxs]
        return Transition(*zip(*batch))

    def __len__(self):
        return len(self.buffer)

# --- 3. SAC Agent ---
class SACAgent:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        self.actor = Actor(state_dim, action_dim)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=LR_ACTOR)
        
        self.critic1 = Critic(state_dim, action_dim)
        self.critic2 = Critic(state_dim, action_dim)
        self.critic1_target = Critic(state_dim, action_dim)
        self.critic2_target = Critic(state_dim, action_dim)
        
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        
        self.critic_optimizer = optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()), lr=LR_CRITIC
        )
        self.memory = ReplayBuffer(REPLAY_SIZE)
        self.train_steps = 0
        
    def _soft_update(self, target_net, source_net, tau=0.005):
        for target_param, source_param in zip(target_net.parameters(), source_net.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + source_param.data * tau)

    def update_networks(self):
        if len(self.memory) < BATCH_SIZE:
            return
        
        self.train_steps += 1
        batch = self.memory.sample(BATCH_SIZE)
        
        state_batch = torch.stack(batch.state).to(DEVICE)
        action_batch = torch.tensor(batch.action, dtype=torch.long, device=DEVICE).unsqueeze(-1)
        reward_batch = torch.tensor(batch.reward, dtype=torch.float32, device=DEVICE).unsqueeze(-1)
        next_state_batch = torch.stack(batch.next_state).to(DEVICE)
        done_batch = torch.tensor(batch.done, dtype=torch.float32, device=DEVICE).unsqueeze(-1)

        # Critic Update
        next_prob, next_log_prob = self.actor(next_state_batch)
        q1_target = self.critic1_target(next_state_batch)
        q2_target = self.critic2_target(next_state_batch)
        min_q_target = torch.min(q1_target, q2_target) 
        next_state_value = (next_prob * (min_q_target - ENTROPY_WEIGHT_ALPHA * next_log_prob)).sum(dim=-1, keepdim=True)
        q_target = reward_batch + (1 - done_batch) * GAMMA * next_state_value
        
        q1_current = self.critic1(state_batch).gather(1, action_batch)
        q2_current = self.critic2(state_batch).gather(1, action_batch)
        critic_loss = nn.MSELoss()(q1_current, q_target.detach()) + nn.MSELoss()(q2_current, q_target.detach())
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Actor Update
        prob, log_prob_all = self.actor(state_batch)
        q1_current_actor = self.critic1(state_batch)
        q2_current_actor = self.critic2(state_batch)
        min_q_current_actor = torch.min(q1_current_actor, q2_current_actor)
        actor_loss = (prob * (ENTROPY_WEIGHT_ALPHA * log_prob_all - min_q_current_actor)).sum(dim=-1).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        if self.train_steps % TARGET_UPDATE_FREQ == 0:
            self._soft_update(self.critic1_target, self.critic1)
            self._soft_update(self.critic2_target, self.critic2)