# src/task_offloading.py
import numpy as np
import networkx as nx
from src.common.config import *
from src.models.satellite_model import (
    calculate_trans_rate, calculate_comp_delay, calculate_comp_energy,
    encode_state, decode_relative_action, calculate_uplink_rate # [修改] 导入解码函数 & 上传速率函数
)

def opt_bandwidth_routing(s_src, s_dst, q_bits, net_manager):
    # (保持不变)
    if s_src == s_dst:
        return [s_src]
    try:
        paths = list(nx.all_simple_paths(net_manager.graph, s_src, s_dst, cutoff=D_M))
    except:
        return None
    if not paths: return None
    valid_paths = []
    for path in paths:
        min_bw = float('inf')
        is_valid = True
        for i in range(len(path)-1):
            u, v = path[i], path[i+1]
            # [修改] 单位转换: Bits -> Hz
            # 1. 获取链路物理速率 (bps)
            link_rate = net_manager.graph.edges[u, v]['rate']
            
            # 2. 计算单时隙最大传输比特数 (Bits Capacity)
            max_bits_capacity = link_rate * TIME_SLOT
            
            # 3. 计算所需带宽 (Hz)
            # 逻辑: (需要传的 Bits / Link Capacity) * Total Bandwidth
            if max_bits_capacity > 0:
                b_k_hz = (q_bits / max_bits_capacity) * B_SAT
            else:
                b_k_hz = float('inf')

            # 4. 检查剩余带宽 (Hz)
            remain_hz = net_manager.satellites[u].M_b - net_manager.graph.edges[u, v]['used_b']
            
            if remain_hz < b_k_hz:
                is_valid = False
                break
            # 这里记录的是剩余的 Hz，或者为了寻找最大瓶颈带宽路径，我们记录剩余 Hz
            min_bw = min(min_bw, remain_hz)
        if is_valid:
            valid_paths.append((path, min_bw))
    if not valid_paths: return None
    valid_paths.sort(key=lambda x: x[1], reverse=True)
    return valid_paths[0][0]

def task_offloading_algorithm(dnn_task_slices, initial_loc, sac_agent, net_manager, action_to_loc):
    """ 核心卸载循环 (Algorithm 2) """
    s_src = initial_loc
    L = len(dnn_task_slices)
    scheme = []
    
    total_T = 0.0
    total_E = 0.0
    dropped = 0
    
    # 记录初始全网方差
    loads = [s.q for s in net_manager.satellites.values()]
    var_before = np.var(loads)
    
    for k in range(L):
        task = dnn_task_slices[k]
        
        # 1. 状态编码 & 动作采样
        state, neighbors = encode_state(s_src, net_manager, task.workload)
        action_idx, _, _ = sac_agent.actor.sample_action(state.unsqueeze(0))
        
        # [修改] 使用相对动作解码
        # 不再使用 action_to_loc[action_idx]
        s_dst_loc = decode_relative_action(s_src, action_idx, net_manager)
        
        # 如果解码失败（越界或太远），视为非法动作 -> 任务失败
        if s_dst_loc is None:
            dropped += 1
            scheme.append((s_src, None, False))
            reward = -PUNISHMENT_FACTOR
            # 存储经验
            sac_agent.memory.push(state, action_idx, reward, state, True)
            break 

        s_dst_node = net_manager.satellites[s_dst_loc]
        
        # 2. 路由与资源检查
        q_bits = task.intermediate_data if k < L-1 else 0
        path = None
        
        can_compute = s_dst_node.check_compute_capacity(task.workload)
        if q_bits > 0:
            path = opt_bandwidth_routing(s_src, s_dst_loc, q_bits, net_manager)
            can_transmit = (path is not None)
        else:
            can_transmit = True 
            
        # 3. 执行或丢弃
        if can_compute and can_transmit:
            # 更新资源
            s_dst_node.update_compute_resource(task.workload)
            if path and q_bits > 0:
                for i in range(len(path)-1):
                    u, v = path[i], path[i+1] # 获取路径上的两个节点 u, v
                    
                    # [修改] 单位转换: Bits -> Hz 用于资源更新
                    link_rate = net_manager.graph.edges[u, v]['rate']
                    max_bits_capacity = link_rate * TIME_SLOT
                    
                    if max_bits_capacity > 0:
                        b_k_hz = (q_bits / max_bits_capacity) * B_SAT
                    else:
                        b_k_hz = B_SAT # Fallback (should not happen if path found)
                        
                    # 更新链路上的带宽记录 (Hz)
                    # += 操作对应 U = b + b_k
                    net_manager.graph.edges[u, v]['used_b'] += b_k_hz
                    
                    # 同时更新卫星节点的发射带宽占用 (如果做了 Satellite.check_comm_capacity 检查的话)
                    # net_manager.satellites[u].update_comm_resource(b_k_hz)
            
            # 计算开销，公式(7)
            # 计算开销，公式(7)
            # [Fix] Include Queuing Delay: Wait Time (before current task) + Service Time
            t_wait = (s_dst_node.q - task.workload) / s_dst_node.C_x
            t_service = calculate_comp_delay(task.workload, s_dst_node.C_x)
            t_comp = t_wait + t_service
            # 计算开销，公式(11)
            e_comp = calculate_comp_energy(task.workload, s_dst_node.C_x)
            
            t_trans = 0
            e_trans = 0
            
            # --- [新增] 星地上传时延 (Ground-to-Satellite Uplink) Eq. 1 ---
            # 只有当这是第一个任务切片 (k=0) 时，才计算上传时延
            if k == 0:
                # 假设输入数据量与这层工作量成正比，或者简单假设一个固定输入大小
                input_bits = 224 * 224 * 3 * 8 # 假设 ImageNet 图片
                uplink_rate = calculate_uplink_rate(B_0, P_G, NOISE_POWER, XI_GI)
                t_uplink = input_bits / uplink_rate
                
                # 将上传时延加到总时延中
                # total_T += t_uplink
                # 上传能耗通常算在地面设备上，论文 Eq. 13 的 Total Energy 指的是 "Satellite Energy"
                # 所以这里只加时延，不加卫星能耗
            
            if path and q_bits > 0:
                # 路径长度 - 1 即为跳数 (Hops)，在网格拓扑中最短路跳数等于曼哈顿距离 MH(i, j)
                hops = len(path) - 1
                if hops > 0:
                    # 获取链路速率 (分母)
                    rate = net_manager.graph.edges[path[0], path[1]]['rate']
                    # 对应公式 (8): Time = (Data / Rate) * Distance
                    t_trans = (q_bits / rate) * hops
                    # 对应公式 (12): Energy = Power * Time
                    e_trans = P_S_TRAN * (q_bits / rate) * hops
            
            O_t = (t_comp + t_trans) + (e_comp + e_trans)
            # 对应公式 (9) 
            total_T += (t_comp + t_trans)
            # 对应公式 (13) 
            total_E += (e_comp + e_trans)
            
            scheme.append((s_src, s_dst_loc, True))
            s_src = s_dst_loc
            
            # 方差奖励
            loads_after = [s.q for s in net_manager.satellites.values()]
            var_after = np.var(loads_after)
            sigma_t = var_after - var_before 
            var_before = var_after
            
            reward = REWARD_CORRECTION_R - SCALING_COEFF_OMEGA * O_t - SCALING_COEFF_RHO * sigma_t
            
            next_state, _ = encode_state(s_src, net_manager, 0)
            sac_agent.memory.push(state, action_idx, reward, next_state, False)
            
        else:
            # 公式 (14)
            dropped += 1
            scheme.append((s_src, None, False))
            reward = REWARD_CORRECTION_R - PUNISHMENT_FACTOR
            sac_agent.memory.push(state, action_idx, reward, state, True)
            break
            
        sac_agent.update_networks()
        
    return scheme, total_T, total_E, dropped