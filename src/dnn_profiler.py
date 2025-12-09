import torch
import torch.nn as nn
import torchvision.models as models
import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LinearRegression

# --- 1. 设备配置 ---
# 优先尝试 MPS (Mac M芯片加速) 或 CUDA，没有则使用 CPU
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print("Using Device: MPS (Mac GPU acceleration)")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print("Using Device: CUDA")
else:
    DEVICE = torch.device("cpu")
    print("Using Device: CPU")

class DNNProfiler:
    def __init__(self, model_name):
        self.model_name = model_name
        self.layer_data = []
        self.hooks = []
        self.reg_models = {} # Store trained regression models: {type_name: {'latency': model, 'energy': model}}
        
        print(f"[{model_name}] Loading model structure (without pre-trained weights)...")
        
        # --- 关键修正：weights=None ---
        # 仅加载网络结构，不下载参数文件。
        # 这样做无需联网，速度极快，且计算出的 FLOPs 和数据量是完全真实的。
        if model_name == 'vgg19':
            self.model = models.vgg19(weights=None)
        elif model_name == 'resnet101':
            self.model = models.resnet101(weights=None)
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        self.model.to(DEVICE)
        self.model.eval() # 设置为推理模式

    def _count_flops(self, layer, input_shape, output_shape):
        """
        计算层的浮点运算次数 (FLOPs)
        标准定义：1 MAC (乘加) = 2 FLOPs
        """
        flops = 0
        
        # 1. 卷积层: H_out * W_out * (C_in * K * K) * C_out
        if isinstance(layer, nn.Conv2d):
            out_h, out_w = output_shape[2], output_shape[3]
            
            # 单个输出像素需要的 MACs (乘加运算次数)
            # kernel_ops = kernel_h * kernel_w * in_channels
            kernel_ops = layer.kernel_size[0] * layer.kernel_size[1] * layer.in_channels
            macs = out_h * out_w * kernel_ops * layer.out_channels
            
            # 转换为 FLOPs
            flops = macs * 2
        
        # 2. 全连接层: In * Out
        elif isinstance(layer, nn.Linear):
            # MACs = Input_Features * Output_Features
            macs = layer.in_features * layer.out_features
            
            # 转换为 FLOPs
            flops = macs * 2
            
        return int(flops)

    def _extract_regression_features(self, layer, input_shape, output_shape):
        """
        Extract features for MLR based on paper Section 4.2:
        - Conv: [C_in, (K/S)^2 * C_out]
        - FC/Pool: [In_Size, Out_Size]
        - Activation: [In_Size]
        """
        features = []
        if isinstance(layer, nn.Conv2d):
            # x1: Number of features in input feature maps (Channels)
            c_in = layer.in_channels
            
            # x2: (Kernel/Stride)^2 * Filters
            k = layer.kernel_size[0] if isinstance(layer.kernel_size, tuple) else layer.kernel_size
            s = layer.stride[0] if isinstance(layer.stride, tuple) else layer.stride
            c_out = layer.out_channels
            term2 = ((k / s) ** 2) * c_out
            
            features = [c_in, term2]
            
        elif isinstance(layer, nn.Linear):
            # FC: Input and Output sizes
            features = [layer.in_features, layer.out_features]
            
        elif isinstance(layer, (nn.MaxPool2d, nn.AvgPool2d, nn.AdaptiveAvgPool2d)):
            # Pooling: Input and Output map sizes (Total elements)
            in_elems = np.prod(input_shape)
            out_elems = np.prod(output_shape)
            features = [in_elems, out_elems]
            
        elif isinstance(layer, (nn.ReLU, nn.BatchNorm2d)):
            # Activation/BN: Input data size
            in_elems = np.prod(input_shape)
            features = [in_elems]
            
        return features

    def _hook_fn(self, layer, input, output):
        """
        钩子函数：拦截每一层的输入输出，记录数据
        """
        # input 是一个 tuple，input[0] 才是 tensor 数据
        in_tensor = input[0]
        
        input_shape = list(in_tensor.shape)
        output_shape = list(output.shape)
        
        # 计算输出数据量 (Elements)
        # 对应论文 Table 3 中的数据传输需求
        # Size (bits) = output_elements * 32 (如果是 float32)
        output_elements = np.prod(output_shape)
        
        # 计算计算量 (FLOPs)
        # 对应论文 Table 3 中的 Task Workload
        flops = self._count_flops(layer, input_shape, output_shape)
        
        # 记录关键数据
        self.layer_data.append({
            'layer_id': len(self.layer_data),     # 层序号
            'type': layer.__class__.__name__,     # 层类型
            'input_shape': str(input_shape),      # 输入形状
            'output_shape': str(output_shape),    # 输出形状
            'output_elements': output_elements,   # 输出元素数 (用于计算传输时延)
            'flops': flops,                       # 计算量 (用于计算执行时延)
            'reg_features': self._extract_regression_features(layer, input_shape, output_shape) # MLR Features
        })

    def profile(self):
        """执行 Profiling 主流程"""
        # 1. 注册 Hook
        # 监听所有常见的计算层
        for name, layer in self.model.named_modules():
            if isinstance(layer, (nn.Conv2d, nn.Linear, nn.MaxPool2d, nn.AvgPool2d, nn.AdaptiveAvgPool2d, nn.ReLU, nn.BatchNorm2d)):
                self.hooks.append(layer.register_forward_hook(self._hook_fn))
        
        # 2. 构造假数据 (ImageNet 标准尺寸: Batch=1, RGB=3, 224x224)
        dummy_input = torch.randn(1, 3, 224, 224).to(DEVICE)
        
        # 3. 跑一次前向传播 (Hook 自动触发)
        print(f"[{self.model_name}] Running dummy inference to capture profile...")
        with torch.no_grad():
            self.model(dummy_input)
            
        # 4. 清理 Hook
        for h in self.hooks:
            h.remove()
            
        print(f"[{self.model_name}] Profiling complete. Captured {len(self.layer_data)} layers.")
        
        # Train MLR models after profiling
        self.train_regression_models()

    def train_regression_models(self):
        """Train MLR models using collected data as Ground Truth"""
        print(f"[{self.model_name}] Training MLR models for prediction...")
        
        # Group data by layer type
        data_by_type = {}
        for item in self.layer_data:
            l_type = item['type']
            if l_type not in data_by_type:
                data_by_type[l_type] = {'X': [], 'y_lat': [], 'y_eng': []}
            
            # Features
            if not item['reg_features']: continue
            
            # Ground Truth (Theoretical)
            # Latency ~ FLOPs (Simplified assumption: Latency = FLOPs / Capacity)
            # Energy ~ FLOPs (Simplified assumption: Energy = k * FLOPs)
            # We use FLOPs as a proxy for both since they are linear to it. 
            # In a real scenario, y would be measured time/energy.
            # Here we just learn to predict FLOPs as a proxy for "Cost".
            
            data_by_type[l_type]['X'].append(item['reg_features'])
            data_by_type[l_type]['y_lat'].append(item['flops']) # Target: Compute workload
            data_by_type[l_type]['y_eng'].append(item['flops']) 

        # Train models
        for l_type, data in data_by_type.items():
            if len(data['X']) < 2: continue # Need at least 2 samples
            
            X = np.array(data['X'])
            y_lat = np.array(data['y_lat'])
            
            model_lat = LinearRegression()
            model_lat.fit(X, y_lat)
            
            self.reg_models[l_type] = model_lat
            
            print(f"  > [{l_type}] Trained (Samples={len(X)}). Coeffs: {model_lat.coef_}, Intercept: {model_lat.intercept_:.2f}")

    def save_to_csv(self, save_dir='data/dnn_profiles'):
        """保存结果到 CSV"""
        # 自动创建目录
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
            
        df = pd.DataFrame(self.layer_data)
        
        # 保存文件
        file_path = os.path.join(save_dir, f'{self.model_name}_profile.csv')
        df.to_csv(file_path, index=False)
        print(f"[{self.model_name}] Data saved to: {file_path}")

# --- 主执行入口 ---
if __name__ == "__main__":
    print("=== Starting DNN Profiling ===\n")
    
    # 1. 跑 VGG19
    try:
        profiler_vgg = DNNProfiler('vgg19')
        profiler_vgg.profile()
        profiler_vgg.save_to_csv()
    except Exception as e:
        print(f"Error profiling VGG19: {e}")
    
    print("-" * 40)
    
    # 2. 跑 ResNet101
    try:
        profiler_resnet = DNNProfiler('resnet101')
        profiler_resnet.profile()
        profiler_resnet.save_to_csv()
    except Exception as e:
        print(f"Error profiling ResNet101: {e}")
    
    print("\n=== All Done! ===")