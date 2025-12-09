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

def run_apt_sat_experiment():
    # 1. 加载模型和数据
    model_name = 'resnet101'
    layers = load_dnn_layers(model_name)
    if not layers:
        print("Error: Could not load DNN layers.")
        return [], {}
        
    # --- [关键] 应用与 main.py 相同的工作负载归一化逻辑，确保一致性 ---
    current_total_workload = sum(l.workload for l in layers)
    d_l_kb = np.random.uniform(TASK_DATA_SIZE_MIN, TASK_DATA_SIZE_MAX)
    d_l_bits = d_l_kb * 1024 * 8
    target_total_workload = d_l_bits * TASK_WORKLOAD_CA
    scale_factor = target_total_workload / current_total_workload
    for l in layers:
        l.workload *= scale_factor
    # ---------------------------------------------------------------

    slices = create_task_slices(layers, L_SLICE_RESNET101)
    
    # 2. 初始化 SAC Agent
    state_dim = (2 * D_M + 1)**2 + 1 
    action_dim = (2 * D_M + 1) ** 2
    sac_agent = SACAgent(state_dim, action_dim)
    
    # 尝试加载训练好的模型
    model_path = 'sac_actor.pth'
    if not os.path.exists(model_path):
        model_path = '../sac_actor.pth' 
    
    if os.path.exists(model_path):
        try:
            sac_agent.actor.load_state_dict(torch.load(model_path, map_location=DEVICE))
            sac_agent.actor.eval()
            print(f"Loaded trained SAC model from {model_path}")
        except Exception as e:
            print(f"Warning: Failed to load model ({e}), using random agent.")
    else:
        print("Warning: No pre-trained model found, using random agent.")

    action_to_loc = {} # Not used in relative action mode
    
    # 3. 实验参数 (Lambda 范围)
    lambdas = [5, 10, 15, 20, 25, 30]
    num_slots = 20 # 每个 lambda 跑多少个时间槽取平均
    
    # 仅记录 APT-SAT 的结果
    results = {
        'APT-SAT': {'TCR': [], 'Delay': [], 'Energy': []}
    }
    
    print("\n--- Starting APT-SAT Experiment (Lambda Sweep) ---")
    
    for lam in lambdas:
        print(f"Running Lambda = {lam} ...")
        
        # 统计变量
        success_count = 0
        total_count = 0
        total_delay = 0.0
        total_energy = 0.0
        
        for _ in range(num_slots):
            n_tasks = np.random.poisson(lam)
            if n_tasks == 0: continue
            
            # 创建环境
            env_apt = NetworkManager(N_SIZE)
            
            # 随机生成任务接入点
            entries = [(np.random.randint(0, N_SIZE), np.random.randint(0, N_SIZE)) for _ in range(n_tasks)]
            
            # 执行任务
            for entry in entries:
                _, t, e, drop = task_offloading_algorithm(slices, entry, sac_agent, env_apt, action_to_loc)
                total_count += 1
                if drop == 0:
                    success_count += 1
                    total_delay += t
                    total_energy += e
        
        # 计算平均值
        tcr = success_count / total_count if total_count > 0 else 0
        avg_delay = total_delay / success_count if success_count > 0 else 0
        avg_energy = total_energy / success_count if success_count > 0 else 0
        
        results['APT-SAT']['TCR'].append(tcr)
        results['APT-SAT']['Delay'].append(avg_delay)
        results['APT-SAT']['Energy'].append(avg_energy)
        
        print(f"  > TCR: {tcr:.4f}, Delay: {avg_delay:.4f}s, Energy: {avg_energy:.4f}J")

    return lambdas, results

def plot_charts(x, res):
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    
    # 1. Completion Rate
    m = 'APT-SAT'
    axs[0].plot(x, res[m]['TCR'], marker='o', label=m, color='blue')
    axs[0].set_title("Task Completion Rate")
    axs[0].set_xlabel("Task Arrival Rate")
    axs[0].set_ylabel("Rate")
    axs[0].legend()
    axs[0].grid(True)
    
    # 2. Delay
    axs[1].plot(x, res[m]['Delay'], marker='s', label=m, color='orange')
    axs[1].set_title("Average Delay")
    axs[1].set_xlabel("Task Arrival Rate")
    axs[1].set_ylabel("Time (s)")
    axs[1].grid(True)
    
    # 3. Energy
    axs[2].plot(x, res[m]['Energy'], marker='^', label=m, color='green')
    axs[2].set_title("Avg Energy Consumption")
    axs[2].set_xlabel("Task Arrival Rate")
    axs[2].set_ylabel("Energy (J)")
    axs[2].grid(True)
    
    plt.tight_layout()
    plt.savefig('apt_sat_results.png')
    print("\nResults saved to apt_sat_results.png")

if __name__ == "__main__":
    x_vals, y_data = run_apt_sat_experiment()
    plot_charts(x_vals, y_data)
