import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import matplotlib.pyplot as plt
import torch
from config import *
from satellite_model import NetworkManager
from main import load_dnn_layers, create_task_slices
from sac_agent import SACAgent
from task_offloading import task_offloading_algorithm
from baselines import run_rola, run_rrp, run_scc, run_ddpg

def run_comparison_experiment():
    # 1. Setup
    model_name = 'resnet101'
    layers = load_dnn_layers(model_name)
    if not layers: return [], {}

    # Workload Normalization
    current_total_workload = sum(l.workload for l in layers)
    d_l_kb = np.random.uniform(TASK_DATA_SIZE_MIN, TASK_DATA_SIZE_MAX)
    d_l_bits = d_l_kb * 1024 * 8
    target_total_workload = d_l_bits * TASK_WORKLOAD_CA
    scale_factor = target_total_workload / current_total_workload
    for l in layers:
        l.workload *= scale_factor
        
    slices = create_task_slices(layers, L_SLICE_RESNET101)
    
    # 2. Setup SAC Agent (APT-SAT)
    state_dim = (2 * D_M + 1)**2 + 1 
    action_dim = (2 * D_M + 1) ** 2
    sac_agent = SACAgent(state_dim, action_dim)
    
    model_path = 'sac_actor.pth'
    if not os.path.exists(model_path): model_path = '../sac_actor.pth'
    if os.path.exists(model_path):
        try:
            sac_agent.actor.load_state_dict(torch.load(model_path, map_location=DEVICE))
            sac_agent.actor.eval()
        except: pass
    
    action_to_loc = {} 
    
    # 3. Experiment Loop
    lambdas = [5, 10, 15, 20, 25, 30]
    num_slots = 20
    
    methods = ['APT-SAT', 'RRP', 'ROLA', 'SCC', 'DDPG']
    results = {m: {'TCR': [], 'Delay': [], 'Energy': []} for m in methods}
    
    print(f"\n--- Starting Full Comparison Experiment (Methods: {methods}) ---")
    
    for lam in lambdas:
        print(f"Running Lambda = {lam} ...")
        
        # Temp accumulators
        acc = {m: {'succ': 0, 'tot': 0, 'd': 0.0, 'e': 0.0} for m in methods}
        
        for _ in range(num_slots):
            n_tasks = np.random.poisson(lam)
            if n_tasks == 0: continue
            
            # Independent Environments for fair comparison logic (same topology structure)
            # Technically we should reset the same env, but creating new ensures clean slate
            # Note: For strict comparison, random seed for topology/locations should be same.
            # Here we rely on statistical average over 20 slots.
            
            entries = [(np.random.randint(0, N_SIZE), np.random.randint(0, N_SIZE)) for _ in range(n_tasks)]
            
            # 1. APT-SAT
            env = NetworkManager(N_SIZE)
            for entry in entries:
                _, t, e, drop = task_offloading_algorithm(slices, entry, sac_agent, env, action_to_loc)
                acc['APT-SAT']['tot'] += 1
                if drop == 0:
                    acc['APT-SAT']['succ'] += 1
                    acc['APT-SAT']['d'] += t
                    acc['APT-SAT']['e'] += e
            
            # 2. RRP
            env = NetworkManager(N_SIZE)
            for entry in entries:
                t, e, drop = run_rrp(slices, entry, env)
                acc['RRP']['tot'] += 1
                if drop == 0:
                    acc['RRP']['succ'] += 1
                    acc['RRP']['d'] += t
                    acc['RRP']['e'] += e

            # 3. ROLA
            env = NetworkManager(N_SIZE)
            for entry in entries:
                t, e, drop = run_rola(slices, entry, env)
                acc['ROLA']['tot'] += 1
                if drop == 0:
                    acc['ROLA']['succ'] += 1
                    acc['ROLA']['d'] += t
                    acc['ROLA']['e'] += e
                    
            # 4. SCC
            env = NetworkManager(N_SIZE)
            for entry in entries:
                t, e, drop = run_scc(slices, entry, env)
                acc['SCC']['tot'] += 1
                if drop == 0:
                    acc['SCC']['succ'] += 1
                    acc['SCC']['d'] += t
                    acc['SCC']['e'] += e
                    
            # 5. DDPG
            env = NetworkManager(N_SIZE)
            for entry in entries:
                # pass agent=None for baseline heuristic/random fallback logic in DDPG function
                t, e, drop = run_ddpg(slices, entry, env) 
                acc['DDPG']['tot'] += 1
                if drop == 0:
                    acc['DDPG']['succ'] += 1
                    acc['DDPG']['d'] += t
                    acc['DDPG']['e'] += e
        
        # Aggregate
        for m in methods:
            s, tot = acc[m]['succ'], acc[m]['tot']
            tcr = s / tot if tot > 0 else 0
            avg_d = acc[m]['d'] / s if s > 0 else 0
            avg_e = acc[m]['e'] / s if s > 0 else 0
            
            results[m]['TCR'].append(tcr)
            results[m]['Delay'].append(avg_d)
            results[m]['Energy'].append(avg_e)
            
        print(f"  Lambda {lam} Done. APT-SAT TCR: {results['APT-SAT']['TCR'][-1]:.2f}")

    return lambdas, results

def plot_charts(x, res):
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    metrics = ['TCR', 'Delay', 'Energy']
    titles = ['Task Completion Rate', 'Average Delay (s)', 'Average Energy (J)']
    markers = {'APT-SAT': 'o', 'RRP': 's', 'ROLA': '^', 'SCC': 'D', 'DDPG': 'x'}
    
    for i, metric in enumerate(metrics):
        for m in res:
            axs[i].plot(x, res[m][metric], marker=markers.get(m, '.'), label=m)
        axs[i].set_title(titles[i])
        axs[i].set_xlabel("Task Arrival Rate")
        axs[i].grid(True)
        axs[i].legend()
    
    plt.tight_layout()
    plt.tight_layout()
    if not os.path.exists('results'):
        os.makedirs('results')
    plt.savefig('results/comparison_results.png')
    print("\nResults saved to results/comparison_results.png")

if __name__ == "__main__":
    x_vals, y_data = run_comparison_experiment()
    plot_charts(x_vals, y_data)
