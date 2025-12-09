import os
import pandas as pd
import numpy as np
from src.models.satellite_model import DNNLayer
from src.algorithms.partition_algorithm import adaptive_dnn_partitioning_algorithm

def load_dnn_layers(model_name):
    path = f'data/dnn_profiles/{model_name}_profile.csv'
    if not os.path.exists(path):
        print(f"Profile not found at {path}. Run dnn_profiler.py first.")
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
