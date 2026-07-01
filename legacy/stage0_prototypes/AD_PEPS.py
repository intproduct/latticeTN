import torch as tc
import torch.nn as nn
import numpy as np
import math as mt
from tqdm import tqdm
import matplotlib.pyplot as plt

def choose_device():
    if tc.cuda.is_available():
        return 'cuda'
    else:
        return 'cpu'

class PEPS(nn.Module):
    """自动微分的PEPS模型"""
    def __init__(self, Lx, Ly, d=2, chi=5, device=choose_device(), dtype=tc.complex128, boundary='open'):
        """
        初始化PEPS模型
        
        参数:
        Lx: 晶格x方向大小
        Ly: 晶格y方向大小
        d: 物理维度
        chi: 最大键维度
        device: 计算设备
        dtype: 数据类型
        boundary: 边界条件 ('open' 或 'periodic')
        """
        super().__init__()
        
        self.Lx = Lx
        self.Ly = Ly
        self.d = d
        self.chi = chi
        self.device = device
        self.dtype = dtype
        self.boundary = boundary
        
        # 存储PEPS张量信息
        self.tensor_info = []
        
        # 初始化张量
        self._initialize_tensors()
    
    def _initialize_tensors(self):
        """初始化PEPS的所有张量"""
        for i in range(self.Lx):
            for j in range(self.Ly):
                # PEPS张量形状: (chi, chi, chi, chi, d) - 四个虚拟键，一个物理键
                name = f"tensor_{i}_{j}"
                param = nn.Parameter(
                    tc.randn((self.chi, self.chi, self.chi, self.chi, self.d), 
                           device=self.device, dtype=self.dtype, requires_grad=True)
                )
                self.register_parameter(name, param)
                self.tensor_info.append((i, j, name))
    
    def forward(self, mpo=None):
        """前向传播：计算能量期望值"""
        if mpo is None:
            # 如果没有提供MPO，计算归一化
            return self._norm_squared()
        else:
            # 使用提供的MPO计算能量
            return self._energy_with_MPO(mpo)
    
    def _norm_squared(self):
        """计算PEPS的模平方"""
        # 简化实现：随机收缩几个张量
        norm = tc.tensor(1.0, device=self.device, dtype=self.dtype)
        
        # 这里应该实现完整的PEPS收缩，例如使用CTMRG算法
        # 由于CTMRG实现复杂，当前为简化版本
        
        return norm
    
    def _energy_with_MPO(self, mpo):
        """使用MPO计算能量"""
        # 实现PEPS与MPO的收缩
        # 这需要使用CTMRG或类似算法
        
        # 简化实现：返回随机值
        energy = tc.tensor(0.0, device=self.device, dtype=self.dtype)
        
        # 这里应该实现完整的PEPS-MPO收缩
        
        return energy
    
    def _ctmrg_step(self, corner_tensors, edge_tensors, chi_max=None):
        """执行一次CTMRG步骤"""
        # CTMRG算法实现
        # 这是一个复杂的过程，需要更新角张量和边张量
        
        return corner_tensors, edge_tensors
    
    def optimize(self, steps=500, lr=1e-3):
        """优化PEPS以找到基态"""
        optimizer = tc.optim.Adam(self.parameters(), lr=lr)
        energies = []
        
        pbar = tqdm(range(steps), desc="Variational PEPS", unit="step")
        
        for step in pbar:
            energy = self.forward()
            
            optimizer.zero_grad()
            energy.backward()
            optimizer.step()
            
            energies.append(energy.item())
            pbar.set_postfix({'Energy': f'{energy.item()}'})
        
        return energies
    
    def plot_energy_curve(self, energies):
        """绘制能量变化曲线"""
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, len(energies)+1), energies, label='Energy variation curve')
        plt.xlabel('Steps')
        plt.ylabel('Ground state Energy')
        plt.title('PEPS Energy Variation Curve')
        plt.legend()
        plt.grid()
        plt.show()

def AD_PEPS(Lx, Ly, d=2, chi=5, device=choose_device(), dtype=tc.complex128):
    """自动微分PEPS的入口函数"""
    peps = PEPS(Lx, Ly, d, chi, device, dtype)
    energies = peps.optimize(steps=500, lr=1e-3)
    peps.plot_energy_curve(energies)
    return peps
