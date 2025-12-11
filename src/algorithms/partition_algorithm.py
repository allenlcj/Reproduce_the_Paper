# src/partition_algorithm.py
from itertools import combinations
import numpy as np
from src.common.config import *
from src.models.satellite_model import calculate_comp_delay, calculate_comp_energy, calculate_trans_rate

def calculate_workload_variance(workloads):
    """ 计算负载方差 (公式 3) """
    if not workloads: return 0.0
    return np.var(workloads)

def calculate_partition_metrics(dnn_layers, partition_points, L):
    N_l = len(dnn_layers)
    partition_indices = [0] + list(partition_points) + [N_l]
    
    total_T = 0.0
    total_E = 0.0
    slice_workloads = []
    
    # 假设传输速率
    noise_isl = K_BOLTZMANN * EQUIV_NOISE_TEMP * B_SAT
    R_avg = calculate_trans_rate(B_SAT, P_S_TRAN, noise_isl, XI_ISL)

    for k in range(L):
        start, end = partition_indices[k], partition_indices[k+1]
        slices = dnn_layers[start:end]
        
        # 1. 计算总量
        w_slice = sum(l.workload for l in slices)
        slice_workloads.append(w_slice)
        
        # 执行时延 (t^exec)
        total_T += calculate_comp_delay(w_slice, C_X_CAP)
        # 执行能耗 (e^exec)
        total_E += calculate_comp_energy(w_slice, C_X_CAP)
        
        # 2. 传输开销 (切片之间)
        if k < L - 1:
            # 取切分点前一层的输出数据量 (Bits)
            last_layer = dnn_layers[end - 1]
            data_bits = last_layer.intermediate_data
            
            # 传输时延 (t^trans)，切片之间
            t_trans = data_bits / R_avg
            # 传输能耗 (e^trans)
            e_trans = P_S_TRAN * t_trans
            
            total_T += t_trans
            total_E += e_trans
            
    VAR = calculate_workload_variance(slice_workloads)
    return VAR, total_T, total_E, slice_workloads

def adaptive_dnn_partitioning_algorithm(layers, L, theta=THETA, delta=DELTA, zeta=ZETA):
    """
    Algorithm 1: Adaptive DNN Partitioning
    Args:
        layers: List of DNNLayer objects
        L: Number of slices (so L-1 cut points)
        theta: Weight for Variance (Default: Config THETA)
        delta: Weight for Delay (Default: Config DELTA)
        zeta: Weight for Energy (Default: Config ZETA)
    Returns:
        best_partition_points: tuple of indices
        partition_workloads: list of workloads for each slice
    """
    N_l = len(layers)
    candidate_points = list(range(1, N_l)) # 候选切分点
    
    best_score = float('inf')
    best_points = None
    best_workloads = None
    
    # 简化：为了快速演示，如果层数太多，只搜索部分点
    search_space = candidate_points[::5] if len(candidate_points) > 50 else candidate_points
    
    # 遍历所有可能的 L-1 个切分点组合
    for cpp in combinations(search_space, L - 1):
        VAR, T, E, workloads = calculate_partition_metrics(layers, cpp, L)
        
        # 归一化加权 (Line 4)
        score = theta * VAR + delta * T + zeta * E
        
        if score < best_score:
            best_score = score
            best_points = cpp
            best_workloads = workloads
            
    return best_points, best_workloads

# --- Partitioning Baselines (for Fig 4 & 5 comparison) ---

def partition_eosat(layers):
    """
    EO-SAT: No partitioning (Edge Only or Offload Entirely).
    Effectively 1 slice containing all layers.
    """
    # Just sum everything into one mock slice object or logic
    # In our system, this means L=1, no cut points.
    # Return None points, and 1 combined workload
    w_total = sum(l.workload for l in layers)
    # Re-use calculation logic? 
    # Or just return empty points []
    return (), [w_total]

def partition_rptsat(layers, L):
    """
    RPT-SAT: Random Partitioning.
    Randomly select L-1 cut points.
    """
    N_l = len(layers)
    if N_l < L: return (), [l.workload for l in layers] # Fallback
    
    candidate_points = list(range(1, N_l))
    import random
    points = tuple(sorted(random.sample(candidate_points, L - 1)))
    
    # Calculate workloads for these random points
    # Reuse calculate_partition_metrics (ignoring VAR/T/E return, just want workloads)
    _, _, _, workloads = calculate_partition_metrics(layers, points, L)
    return points, workloads

def partition_sosat(layers):
    """
    SO-SAT: Single-Point Optimal Partitioning (L=2).
    Search for best single cut point minimizing Cost.
    Using default weights specific to SO-SAT if mentioned, otherwise same as APT.
    """
    # L=2
    return adaptive_dnn_partitioning_algorithm(layers, L=2)

def partition_dosat(layers):
    """
    DO-SAT: Dual-Point Optimal Partitioning (L=3).
    Search for best two cut points minimizing Cost.
    """
    # L=3
    return adaptive_dnn_partitioning_algorithm(layers, L=3)