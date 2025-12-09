# APT-SAT Paper Reproduction

This repository contains the reproduction of the paper:
**"APT-SAT: An Adaptive DNN Partitioning and Task Offloading Framework within Collaborative Satellite Continuum"** (Peng et al., 2025).

The project implements the complete framework described in the paper, including the satellite network environment, adaptive partitioning algorithm (Algorithm 1), and SAC-based task offloading mechanism (Algorithm 2). It provides scripts to reproduce the key experimental results (Fig 4, Fig 5, and comparison with baselines).

## 📂 Project Structure

- `src/`: Source code for the framework.
    - `config.py`: Global configuration parameters (Table 3 & 4).
    - `satellite_model.py`: Satellite network environment and resource models.
    - `dnn_profiler.py`: DNN model profiling (ResNet, VGG).
    - `partition_algorithm.py`: Adaptive partitioning algorithm.
    - `sac_agent.py`: Soft Actor-Critic (SAC) reinforcement learning agent.
    - `task_offloading.py`: Task offloading and routing logic.
    - `baselines.py`: Implementation of baseline methods (RRP, ROLA, SCC, DDPG).
    - **Experiment Scripts**:
        - `run_apt_sat_experiment.py`: Runs the standalone APT-SAT method.
        - `run_comparison_experiment.py`: Runs comparison against benchmarks (Section 5.1).
        - `run_partition_experiment.py`: Runs partitioning strategy comparison (Section 5.2).
- `data/dnn_profiles/`: Profiling data for DNN models.
- `results/`: Directory where all experimental result plots are saved.

## 🚀 Getting Started

### 1. Environment Setup

Create the conda environment using the provided `environment.yml` file:

```bash
conda env create -f environment.yml
conda activate apt-sat
```

### 2. Reproduction of Experiments

This project provides three main scripts to reproduce different parts of the paper's results. All generated plots will be saved in the `results/` directory.

#### Experiment A: Standalone APT-SAT Performance
Run this script to evaluate the performance of the proposed APT-SAT method under different task arrival rates ($\lambda$).

```bash
python src/run_apt_sat_experiment.py
```
**Output**: `results/apt_sat_results.png`

#### Experiment B: Comparative Analysis (Baselines)
Run this script to compare APT-SAT against four baseline methods:
- **RRP**: Residual-Resource-Priority (Greedy)
- **ROLA**: Random Offloading
- **SCC**: Self-adaptive / Genetic Algorithm
- **DDPG**: Deep Deterministic Policy Gradient

```bash
python src/run_comparison_experiment.py
```
**Output**: `results/comparison_results.png`

#### Experiment C: Partitioning Strategy Analysis (Fig 4 & 5)
Run this script to reproduce the partitioning performance results for ResNet101 and VGG19, comparing:
- **APT-SAT** (Proposed Adaptive)
- **DO-SAT** (Delay Optimal)
- **EO-SAT** (Energy Optimal)
- **RPT-SAT** (Random Partitioning)
- **SO-SAT** (Static Partitioning)

```bash
python src/run_partition_experiment.py
```
**Output**: 
- `results/partition_results_ResNet101.png`
- `results/partition_results_VGG19.png`

## 📊 Results

The reproduction focuses on verifying the core claims of the paper:
1. **Efficiency**: APT-SAT achieves higher Task Completion Rate (TCR) compared to baselines in high-load scenarios.
2. **Adaptability**: The partitioning algorithm effectively balances delay and energy consumption compared to single-objective optimization (DO/EO).

Check the `results/` folder after running the scripts to visualize these findings.
