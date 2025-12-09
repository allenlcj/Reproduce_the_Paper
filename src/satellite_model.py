# src/satellite_model.py
import networkx as nx
import numpy as np
import torch
from config import *

# --- 辅助函数：相对动作解码 (支持圆柱体拓扑) ---
def decode_relative_action(current_loc, action_idx, net_manager):
    """
    将动作索引转换为目标卫星坐标
    关键修改：支持 Y 轴（轨道内）循环穿越
    """
    grid_size = 2 * D_M + 1
    
    # 1. 解码相对位移 (dx, dy)
    dx_idx = action_idx % grid_size
    dy_idx = action_idx // grid_size
    
    dx = dx_idx - D_M
    dy = dy_idx - D_M
    
    # 2. 计算绝对坐标
    x, y = current_loc
    target_x = x + dx
    target_y = y + dy
    
    # [关键修改] Y轴 (轨道内): 循环连接 (Cylindrical Topology)
    target_y = target_y % net_manager.N 
    
    # [关键修改] X轴 (轨道间): 边界检查 (Bounded)
    # 如果试图跨越左右边界 (如从轨道0向左跳)，视为非法，强制原地执行
    if target_x < 0 or target_x >= net_manager.N:
        return current_loc 
        
    # 3. 距离检查 (Manhattan)
    dist = abs(dx) + abs(dy)
    if dist > D_M:
        return current_loc 
        
    return (target_x, target_y)

class DNNLayer:
    def __init__(self, layer_id, type, flops, output_elements):
        self.id = layer_id
        self.type = type
        self.flops = flops
        self.intermediate_data = output_elements * FLOAT_BITS 
        self.workload = flops 
class Satellite:
    def __init__(self, location, sat_id=None):
        self.sat_id = sat_id
        self.location = location
        
        # --- 1. 计算资源 (Computing Resources) ---
        # Cx: 计算速率 (3 GHz)
        self.C_x = C_X_CAP 
        # Mw: 单个时间槽内的总计算容量 (Cycles) [公式 4]
        # 维度: Cycles/s * s = Cycles
        self.M_w = self.C_x * TIME_SLOT 
        
        # q: 当前计算负载 (对应论文中的 q)
        self.q = 0.0  
        
        # --- 2. 通信资源 (Communication Resources) ---
        # Mb: 最大带宽容量 (Hz) [公式 5]
        # 注意：这里直接取带宽值，不要乘时间！
        self.M_b = B_SAT  # 20 MHz
        
        # b: 当前带宽占用 (对应论文中的 b)
        self.b = 0.0

        # --- 统计指标 ---
        self.total_workload = 0.0
        self.dropped_tasks = 0

    def reset(self):
        """ 每个时间槽开始时重置资源状态 """
        # 注意：通常 q (计算队列) 在时间槽之间可能会继承（如果排队），
        # 但如果是"即时处理或丢弃"模型，则可以清零。
        # 论文 Eq(4) 看起来是针对"当前决策瞬间"的快照。
        self.q = 0.0  
        self.b = 0.0  # 带宽占用每时隙重置
        
        # 统计数据不清零，除非是 Episode 结束
        # self.total_workload = 0.0 

    def get_remaining_resources(self):
        """ 获取剩余计算资源比例 (用于 State) """
        return max(0.0, (self.M_w - self.q) / self.M_w)
    
    def get_remaining_bandwidth(self):
        """ 获取剩余带宽资源比例 (用于 State) """
        return max(0.0, (self.M_b - self.b) / self.M_b)

    # ==========================
    # 公式 (4) 计算资源检查
    # ==========================
    def check_compute_capacity(self, m_k):
        """
        判断是否满足: q + m_k <= Mw
        m_k: 新任务的计算量 (Cycles)
        注意：为了支持多时隙处理 (Mult-slot processing)，允许排队。
        假设缓冲区大小为 50 个时间槽的容量。
        """
        QUEUE_BUFFER_SIZE = 50 
        return (self.q + m_k) <= (self.M_w * QUEUE_BUFFER_SIZE)

    def update_compute_resource(self, m_k):
        """ 更新计算负载 """
        if self.check_compute_capacity(m_k):
            self.q += m_k
            self.total_workload += m_k
            return True
        else:
            self.dropped_tasks += 1
            return False

    # ==========================
    # 公式 (5) 通信资源检查
    # ==========================
    def check_comm_capacity(self, b_k):
        """
        判断是否满足: b + b_k <= Mb
        b_k: 新任务需要的带宽 (Hz)
        """
        return (self.b + b_k) <= self.M_b

    def update_comm_resource(self, b_k):
        """ 更新带宽占用 """
        # 注意：这通常在建立链路时调用
        if self.check_comm_capacity(b_k):
            self.b += b_k
            return True
        return False

def calculate_comp_delay(workload, compute_capacity):
    # workload 对应公式中的 q (或 d_l * c_l)，公式(6)
    # compute_capacity 对应 C_x
    return workload / compute_capacity

def calculate_comp_energy(workload, compute_capacity):
    # KAPPA 对应公式中的 \kappa，公式(10)
    # compute_capacity ** 2 对应 f^2
    return KAPPA * (compute_capacity ** 2) * workload

def calculate_trans_rate(bandwidth, power, noise, xi_isl):
    """
    计算星间链路传输速率 (Equation 2)
    r(i, j) = B * log2(1 + (Pt * Gi * Gj * Fi * Fj * L_p) / (k * Theta * B))
    @param xi_isl: 星间链路增益因子 (Gt * Gr * F * F * FSPL)，已在config中预计算
    """
    # 2. 计算接收信号功率 (Signal Power)
    # Pr = Pt * XI_ISL
    pr = power * xi_isl

    # 4. 计算信噪比 (SNR)
    snr = pr / noise

    # 5. 代入香农公式计算速率
    rate_bps = bandwidth * np.log2(1 + snr)
    return rate_bps

def calculate_uplink_rate(bandwidth, power, noise, xi_gi):
    r"""
    计算星地上传速率 (Equation 1)
    v_{g,i}(t) = B0 * log2(1 + (Pg * \xi_{g,i}) / GN)
    @param xi_gi: 信道增益因子 (G_gw * G_sat * FSPL_GS)，已在config中预计算
    """
    # 信噪比
    snr = (power * xi_gi) / noise
    rate = bandwidth * np.log2(1 + snr)
    return rate

class NetworkManager:
    def __init__(self, N):
        self.N = N
        self.graph = nx.Graph() # [修改] 手动构建图
        self.satellites = {}
        
        # 1. 初始化节点
        for i in range(N):
            for j in range(N):
                node = (i, j)
                self.satellites[node] = Satellite(node)
                self.graph.add_node(node)
                
        # 2. 建立连接 (圆柱体拓扑)
        self._build_topology()
        self._setup_isl_links()

    def _build_topology(self):
        """ 构建符合论文的圆柱体拓扑 (轨道内循环，轨道间不循环) """
        N = self.N
        for i in range(N):
            for j in range(N):
                curr = (i, j)
                
                # A. 轨道内 (Intra-plane): 循环连接 (上下相连)
                # (i, j) <-> (i, j+1)
                next_j = (j + 1) % N
                self.graph.add_edge(curr, (i, next_j))
                
                # B. 轨道间 (Inter-plane): 线性连接 (左右不循环)
                # (i, j) <-> (i+1, j)
                if i < N - 1:
                    self.graph.add_edge(curr, (i + 1, j))

    def _setup_isl_links(self):
        # 计算星间链路噪声功率
        
        # 使用更新后的签名: (bandwidth, power, noise, xi_isl)
        avg_rate = calculate_trans_rate(B_SAT, P_S_TRAN, noise, XI_ISL) 
        for u, v in self.graph.edges():
            self.graph.edges[u, v]['rate'] = avg_rate
            self.graph.edges[u, v]['used_b'] = 0.0

    def reset(self):
        for s in self.satellites.values():
            s.reset()
        for u, v in self.graph.edges():
            self.graph.edges[u, v]['used_b'] = 0.0

    def get_neighbors(self, u_loc):
        # 使用 NetworkX 的 adj 获取真实邻居 (包含循环连接的邻居)
        # 注意: 需要包含自身
        candidates = list(self.graph.neighbors(u_loc))
        candidates.append(u_loc)
        
        # 过滤 (虽然拓扑已经限制了连接，但为了保险起见，或者如果 D_M > 1)
        # 简单的 BFS 查找 D_M 范围内的节点会更准确，这里简化直接返回直连邻居
        # 如果 D_M=3, 实际上 SAC 可以跳多跳，但在 get_neighbors 用于 State 编码时
        # 我们通常只编码直连邻居的状态，或者编码 D_M 范围内的。
        # 这里为了 State 维度一致性，我们还是按距离遍历
        
        # 简单实现：只返回 1-hop 邻居 + 自身
        return candidates

def encode_state(current_loc, net_manager, w_k):
    neighbors = net_manager.get_neighbors(current_loc)
    state_vec = []
    for loc in neighbors:
        state_vec.append(net_manager.satellites[loc].get_remaining_resources())
    
    # 填充到固定大小 
    # 注意：圆柱拓扑下，邻居数量是固定的 (4个 + 自身 = 5个，除了边界只有3-4个)
    # 我们之前的 max_len 是 (2*D_M+1)^2 = 49，这里为了兼容 SAC 输入维度，保持填充
    max_len = (2 * D_M + 1)**2
    while len(state_vec) < max_len:
        state_vec.append(0.0)
    state_vec = state_vec[:max_len]
        
    task_ratio = w_k / C_X_CAP 
    state_vec.append(task_ratio)
    
    return torch.tensor(state_vec, dtype=torch.float32, device=DEVICE), neighbors

def setup_action_map(N):
    return {}, {}