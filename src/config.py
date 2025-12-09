# src/config.py
import torch
import numpy as np

# ==========================================
# 1. 基础设备配置 (Device Configuration)
# ==========================================
# 自动检测加速硬件：优先使用 Mac 的 MPS (Metal Performance Shaders)，其次是 NVIDIA 的 CUDA，最后回退到 CPU。
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")
print(f"Using Device: {DEVICE}")

# ==========================================
# 2. 卫星网络环境参数 (Satellite Network Environment)
# ==========================================
N_SIZE = 10  #卫星数量         

# 频率与波长 (Carrier Frequency: 12 GHz)
CARRIER_FREQ = 12e9
LIGHT_SPEED = 3e8
WAVELENGTH = LIGHT_SPEED / CARRIER_FREQ # Lambda

K_BOLTZMANN = 1.380649e-23 # 玻尔兹曼常量 (J/K)
EQUIV_NOISE_TEMP = 300 # 等效噪声温度 (K)

# --- A. 星地链路 (Ground-Satellite Link) ---
B_0 = 10e6            # Gateway Bandwidth 10MHz
P_G = 1000.0          # Gateway Power (Pg) = 30 dBW
NOISE_POWER = K_BOLTZMANN * EQUIV_NOISE_TEMP * B_0 # Gaussian noise

ANTENNA_GAIN_GW = 10**(4.5)  # 45 dBi 天线增益
ANTENNA_GAIN_SAT = 1000.0    # 30 dBi 天线增益

DIST_GS = 600e3       # 星地距离600 km

# 星地信道增益因子 XI_GI
_fspl_gs = (WAVELENGTH / (4 * np.pi * DIST_GS)) ** 2
XI_GI = ANTENNA_GAIN_GW * ANTENNA_GAIN_SAT * _fspl_gs

# --- B. 星间链路 (ISL) 参数 ---
B_SAT = 20e6
P_S_TRAN = 1000.0       # Ps = 1000 W
ANTENNA_GAIN = 1000.0   # G_sat = 1000
POINTING_FACTOR = 1.0   
ISL_DISTANCE = 2000e3   # 2000 km (Requested by user)

noise = K_BOLTZMANN * EQUIV_NOISE_TEMP * B_SAT

# 预计算星间信道增益因子 XI_ISL
# FSPL = (Lambda / (4 * pi * d))^2
# XI_ISL = Gt * Gr * F * F * FSPL
_fspl_isl = (WAVELENGTH / (4 * np.pi * ISL_DISTANCE)) ** 2
XI_ISL = ANTENNA_GAIN * ANTENNA_GAIN * POINTING_FACTOR * POINTING_FACTOR * _fspl_isl


# --- B. 卫星计算能力 ---
C_X_CAP = 3e9           # 3 GHz

# 最大通信跳数
D_M = 3
# 对应论文 Table 3: Maximum communication distance D_M = 3
# 限制了 SAC 智能体的动作空间范围，只允许卸载给 3 跳内的邻居
D_M = 3               

# ==========================================
# 4. 任务与计算模型 (Task & Computation Model)
# 对应论文 Section 3.3 & 3.4
# ==========================================
# 任务计算密度 c_l (单位: Cycles/bit)
# 对应论文 Table 3: Task workload c_l = [2, 2.5] Kcycles/bit
# 这里取 2500 Cycles/bit
# 对应论文 Table 3: Task workload c_l = [2, 2.5] Kcycles/bit
# 这里取 2500 Cycles/bit
# 对应论文 Table 3: Task workload c_l = [2, 2.5] Kcycles/bit
# 这里取 2500 Cycles/bit
TASK_WORKLOAD_CA = 2500  

# 对应论文 Table 3: Size of computation data of tasks d_l = [200, 300] KB
# 这是一个范围值，用于随机生成任务的大小
# 我们将使用它来归一化 Profiler 生成的理论 FLOPs，使其符合论文的实验设定
TASK_DATA_SIZE_MIN = 200 # KB
TASK_DATA_SIZE_MAX = 300 # KB

# FLOPs to Cycles conversion scale
# Assuming 1 FLOP approx 1 Cycle for simplified estimation, or use this to tune.
# [校准]: Set to 0.01 to simulate hardware acceleration (e.g. GPU), bringing 4e9 FLOPs down to ~4e7 Cycles,
# fitting within the 1.5e8 Cycles capacity of a 0.05s time slot.
WORKLOAD_SCALE = 0.01

# [校准参数] 能耗系数 \kappa (用于公式 10: e = \kappa * f^2 * workload)
# 对应论文 Table 3: Effective capacitance coefficient \kappa = 10^-28
# 【复现调整】：为了降低计算能耗以匹配论文实验结果，微调为 3e-29。
KAPPA = 1e-28            

# 系统时间槽长度 \tau (单位: 秒)
# 对应论文 Table 3: System time slot length = 0.05s
TIME_SLOT = 0.05         

# 任务到达率 \lambda (泊松分布参数)
# 对应论文 Table 3: Task arrival rate \lambda = 4~40
# 这里设为 40，用于模拟高负载场景
TASK_ARRIVAL_RATE_LAMBDA = 10

# 数据精度 (Bits)，用于将 output elements 转换为比特数 (float32 = 32 bits)
FLOAT_BITS = 32          

# DNN 模型切片数量 L
# 对应论文 Table 3
L_SLICE_VGG19 = 3        
L_SLICE_RESNET101 = 4    

# ==========================================
# 5. 分区算法参数 (Partitioning Algorithm)
# 对应论文 Algorithm 1 (Line 4)
# ==========================================
# 目标函数: cost = \theta * VAR + \delta * T + \zeta * E
# 用于在预处理阶段寻找最优切分点
THETA = 1e-8  # 方差 (VAR) 的权重 (由于方差数值通常极大，给予极小权重以平衡)
DELTA = 1.0   # 总延迟 (T) 的权重
ZETA = 1.0    # 总能耗 (E) 的权重

# ==========================================
# 6. SAC 强化学习参数 (SAC Hyperparameters)
# 对应论文 Section 4.4 & Eq. (21)-(25)
# ==========================================
# 熵正则化系数 \alpha (Eq. 22)
# 控制探索程度。论文 Table 3 设为 0.01，这里微调为 0.05 以增加初期探索。
ENTROPY_WEIGHT_ALPHA = 0.01

# 奖励修正因子 \Re (Eq. 21: r_t = \Re - ...)
# 一个正数常数，确保奖励倾向于正值，防止 Critic 发散。
REWARD_CORRECTION_R = 3

# 优化目标缩放系数 \varpi (Eq. 21)
# 用于缩放 (Delay + Energy) 的惩罚项
SCALING_COEFF_OMEGA = 1.5  

# 负载均衡缩放系数 \rho (Eq. 21)
# 用于缩放方差变化项 \sigma^t
SCALING_COEFF_RHO = 1    

# 任务失败惩罚 \vartheta (Eq. 21)
# 当任务被 Drop 时给予的巨大负奖励
PUNISHMENT_FACTOR = 0.5   

# ==========================================
# 7. 神经网络架构 (Neural Network Architecture)
# 对应论文 Table 4
# ==========================================
# Actor 网络隐藏层单元数
ACTOR_FC_UNITS = 256
# Critic 网络隐藏层单元数
CRITIC_FC_UNITS = 256

# 学习率 (Learning Rate)
LR_ACTOR = 3e-4
LR_CRITIC = 3e-4

# 折扣因子 \gamma (Eq. 25)
GAMMA = 0.99

# 训练参数
BATCH_SIZE = 128        # 经验回放采样批次大小
REPLAY_SIZE = 100000    # 经验回放池容量
TARGET_UPDATE_FREQ = 20 # 目标网络软更新频率 (steps)

# 路径规划参数 (Algorithm 2)
# OptBW 算法中考虑的前 k 条最短路径
K_PATHS = 3

