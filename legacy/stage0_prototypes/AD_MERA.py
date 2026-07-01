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

class MERA(nn.Module):
    """自动微分的MERA模型"""
    def __init__(self, N, d=2, chi=10, device=choose_device(), dtype=tc.complex128):
        """
        初始化MERA模型
        
        参数:
        N: 系统大小（必须是2的幂次）
        d: 物理维度
        chi: 最大键维度
        device: 计算设备
        dtype: 数据类型
        """
        super().__init__()
        assert mt.log2(N).is_integer(), "系统大小必须是2的幂次"
        
        self.N = N
        self.d = d
        self.chi = chi
        self.device = device
        self.dtype = dtype
        self.levels = int(mt.log2(N))
        
        # 存储MERA张量名称
        self.disentanglers = []
        self.isometries = []
        
        # 初始化张量
        self._initialize_tensors()
    
    def _initialize_tensors(self):
        """初始化MERA的所有张量"""
        for level in range(self.levels):
            # 计算当前层的站点数
            n_sites = self.N // (2 ** level)
            
            # 初始化disentanglers (U) - 形状: (chi, chi, d, d)
            for site in range(0, n_sites, 2):
                # 生成唯一的参数名称
                param = nn.Parameter(
                    tc.randn((self.chi, self.chi, self.d, self.d), 
                           device=self.device, dtype=self.dtype, requires_grad=True)
                )
                self.register_parameter(f"disentangler_{level}_{site}", param)
                self.disentanglers.append((level, site, param))
            
            # 初始化isometries (V) - 形状: (chi, d, chi)
            for site in range(n_sites):
                # 生成唯一的参数名称
                param = nn.Parameter(
                    tc.randn((self.chi, self.d, self.chi), 
                           device=self.device, dtype=self.dtype, requires_grad=True)
                )
                self.register_parameter(f"isometry_{level}_{site}", param)
                self.isometries.append((level, site, param))
        
        # 初始化顶层的有效哈密顿量张量（如果需要）
        top_ham = nn.Parameter(
            tc.randn((self.chi, self.chi, self.chi, self.chi), 
                   device=self.device, dtype=self.dtype, requires_grad=True)
        )
        self.register_parameter("top_hamiltonian", top_ham)
    
    def forward(self, mpo=None):
        """前向传播：计算能量期望值"""
        if mpo is None:
            # 如果没有提供MPO，使用顶层哈密顿量
            return self._energy_with_top_hamiltonian()
        else:
            # 使用提供的MPO计算能量
            return self._energy_with_MPO(mpo)
    
    def _energy_with_top_hamiltonian(self):
        """使用顶层哈密顿量计算能量"""
        # 简化实现：直接使用顶层哈密顿量和顶层张量收缩
        top_ham = getattr(self, "top_hamiltonian")
        
        # 这里应该实现完整的MERA收缩，当前为简化版本
        energy = tc.einsum('abcd,abcd->', top_ham, top_ham.conj())
        return energy
    
    def _energy_with_MPO(self, mpo):
        """使用MPO计算能量"""
        # 实现MERA与MPO的收缩
        # 这是一个复杂的过程，需要逐层进行重正化群变换
        
        # 简化实现：计算重叠
        energy = tc.tensor(0.0, device=self.device, dtype=self.dtype)
        
        # 这里应该实现完整的MERA-MPO收缩
        # 包括：1. 将MPO重正化到顶层 2. 与MERA张量收缩
        
        return energy
    
    def optimize(self, steps=1000, lr=1e-3):
        """优化MERA以找到基态"""
        optimizer = tc.optim.Adam(self.parameters(), lr=lr)
        energies = []
        
        pbar = tqdm(range(steps), desc="Variational MERA", unit="step")
        
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
        plt.title('MERA Energy Variation Curve')
        plt.legend()
        plt.grid()
        plt.show()

def AD_MERA(N, d=2, chi=10, device=choose_device(), dtype=tc.complex128):
    """自动微分MERA的入口函数"""
    mera = MERA(N, d, chi, device, dtype)
    energies = mera.optimize(steps=1000, lr=1e-3)
    mera.plot_energy_curve(energies)
    return mera
