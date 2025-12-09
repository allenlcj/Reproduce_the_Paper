# src/main.py
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import pandas as pd
import numpy as np
import torch
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.common.config import *
from src.models.satellite_model import NetworkManager, DNNLayer, setup_action_map
from src.algorithms.partition_algorithm import adaptive_dnn_partitioning_algorithm
from src.models.sac_agent import SACAgent
from src.algorithms.task_offloading import task_offloading_algorithm
from src.common.utils import load_dnn_layers, create_task_slices



if __name__ == "__main__":
    net_manager = NetworkManager(N_SIZE)
    # [修改] setup_action_map 不再需要
    loc_to_action, action_to_loc = {}, {} 
    
    # 状态维度: (2*D_M+1)^2 个邻居资源 + 1 个任务占比
    state_dim = (2 * D_M + 1)**2 + 1 
    
    # [修改] 动作维度：相对位移空间大小
    # Box width = 2 * D_M + 1
    action_dim = (2 * D_M + 1) ** 2
    
    sac_agent = SACAgent(state_dim, action_dim)
    
    model_name = 'resnet101'
    layers = load_dnn_layers(model_name)
    if not layers: exit()
    
    # -----------------------------------------------------
    # [新增] 根据论文参数 d_l 归一化工作负载
    # -----------------------------------------------------
    # 1. 计算当前的 Profiler 总工作量
    current_total_workload = sum(l.workload for l in layers)
    
    # 2. 生成论文设定的目标工作量
    # d_l: 随机生成 [200, 300] KB
    d_l_kb = np.random.uniform(TASK_DATA_SIZE_MIN, TASK_DATA_SIZE_MAX)
    d_l_bits = d_l_kb * 1024 * 8
    
    # Target Workload = d_l * c_l (Cycles)
    target_total_workload = d_l_bits * TASK_WORKLOAD_CA
    
    # 3. 计算缩放因子并应用
    scale_factor = target_total_workload / current_total_workload
    print(f"[Paper Params] Task Data Size d_l: {d_l_kb:.2f} KB")
    print(f"[Paper Params] Target Workload: {target_total_workload:.2e} Cycles (Scale Factor: {scale_factor:.4f})")
    
    for l in layers:
        l.workload *= scale_factor
    # -----------------------------------------------------

    slices = create_task_slices(layers, L_SLICE_RESNET101)
    
    print(f"Model: {model_name}, Slices: {len(slices)}")
    for i, s in enumerate(slices):
        print(f"  Slice {i}: Workload={s.workload:.2e} Cycles, TransData={s.intermediate_data:.2e} Bits")

    print("\n--- Starting SAC Training Loop ---")
    # [建议] 增加训练轮数，确保收敛
    for episode in range(500): 
        net_manager.reset()
        entry_node = (np.random.randint(0, N_SIZE), np.random.randint(0, N_SIZE)) # 随机入口
        
        _, t, e, dropped = task_offloading_algorithm(
            slices, entry_node, sac_agent, net_manager, action_to_loc
        )
        
        if episode % 10 == 0:
            print(f"Ep {episode}: Delay={t:.4f}s, Energy={e:.4f}J, Drop={dropped}")
            
    torch.save(sac_agent.actor.state_dict(), 'sac_actor.pth')
    print("Model saved to sac_actor.pth")