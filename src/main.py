# src/main.py
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import pandas as pd
import numpy as np
import torch
from config import *
from satellite_model import NetworkManager, DNNLayer, setup_action_map
from partition_algorithm import adaptive_dnn_partitioning_algorithm
from sac_agent import SACAgent
from task_offloading import task_offloading_algorithm

def load_dnn_layers(model_name):
    path = f'data/dnn_profiles/{model_name}_profile.csv'
    if not os.path.exists(path):
        print("Profile not found. Run dnn_profiler.py first.")
        return []
    df = pd.read_csv(path)
    layers = []
    for _, row in df.iterrows():
        if row['flops'] > 0:
            # DNNLayer 初始化会将 output_elements * 32 转换为 Bits
            l = DNNLayer(row['layer_id'], row['type'], row['flops'], row['output_elements'])
            layers.append(l)
    return layers

def create_task_slices(layers, L):
    """ 根据 Profiling 结果和 Algorithm 1 生成任务切片 """
    points, slice_workloads = adaptive_dnn_partitioning_algorithm(layers, L)
    
    if points is None:
        print("Partitioning failed, using fallback split.")
        indices = np.linspace(0, len(layers), L+1, dtype=int).tolist()
    else:
        indices = [0] + list(points) + [len(layers)]
    
    slices = []
    for k in range(L):
        start, end = indices[k], indices[k+1]
        
        # 1. 聚合 Workload (该切片内所有层的计算量)
        # 注意: adaptive_dnn_partitioning_algorithm 返回的 slice_workloads 已经是这个值
        total_workload = slice_workloads[k] if slice_workloads else sum(l.workload for l in layers[start:end])
        
        # 2. 获取切分点处的输出数据量 (Bits)
        if k < L - 1:
            # 切分点前一层的输出即为需要传输的数据
            # 例如: 切分点是 layer 5 (indices[k+1]=5)，则 layer 4 的输出需要传输
            cut_layer = layers[end - 1]
            data_bits = cut_layer.intermediate_data
        else:
            data_bits = 0
            
        # 3. 构造切片对象
        # 注意: 这里手动设置 workload 和 intermediate_data，避免被 init 再次缩放
        s = DNNLayer(k, "Slice", 0, 0)
        s.workload = total_workload
        s.intermediate_data = data_bits 
        slices.append(s)
        
    return slices

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