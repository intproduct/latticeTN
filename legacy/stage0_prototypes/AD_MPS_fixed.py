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

def clone_func(tensors):
    """辅助函数：克隆 MPS 张量列表"""
    return [tensor.clone() for tensor in tensors]

def check_paras(mps, mpo):
    assert mps.length == mpo.length, "MPS和MPO长度不匹配"
    assert mps.dim == mpo.dim, "MPS和MPO维度不匹配"
    assert mps.dtype == mpo.dtype, "MPS和MPO数据类型不匹配"
    assert mps.device == mpo.device, "MPS和MPO设备不匹配"
    assert mpo.tensors is not None, "MPO 尚未生成张量（请先调用 generate_TFI_MPO）"

class MPS(nn.Module):
    def __init__(self, tensors=None, length=None, dim=None, chi=None, device=choose_device(), dtype=tc.complex128, boundary='open'):
        super().__init__()
        if tensors is None:
            assert length is not None, "长度必须提供"
            assert dim is not None, "维度必须提供"
            assert chi is not None, "键维度必须提供"
            
            self.length = length
            self.dim = dim
            self.chi = chi
            self.device = device
            self.dtype = dtype
            self.boundary = boundary
        
            # 将张量注册为模块参数
            if self.boundary == 'open':
                tensors = [tc.randn((1, dim, chi), device=device, dtype=dtype, requires_grad=True)] + [tc.randn((chi, dim, chi), device=device, dtype=dtype, requires_grad=True) for _ in range(length-2)] + [tc.randn((chi, dim, 1), device=device, dtype=dtype, requires_grad=True)]
            elif self.boundary == 'periodic':
                tensors = [tc.randn((chi, dim, chi), device=device, dtype=dtype, requires_grad=True) for _ in range(length)]
        else:
            self.length = len(tensors)
            self.dim = tensors[0].shape[1]
            self.chi = tensors[0].shape[2]
            self.device = choose_device()
            self.dtype = tensors[0].dtype
            self.boundary = boundary
        
        # 将张量注册为参数
        self.tensors = nn.ParameterList([nn.Parameter(t, requires_grad=True) for t in tensors])

    def clone_tensors(self):
        return clone_func(self.tensors)

    def clone_mps(self):
        new_tensors = self.clone_tensors()
        psi = MPS(new_tensors, boundary=self.boundary)
        return psi

    def normalize(self, return_mps=True):
        tensors = [tensor.clone() for tensor in self.tensors]
        
        if self.boundary == 'open':
            # 开边界条件归一化 - 实现更鲁棒的数值稳定策略
            v = tc.ones((1, 1), dtype=self.dtype, device=self.device)
            norm_factors = []
            
            for i in range(len(tensors)):
                # 收缩当前张量和其共轭
                v = tc.einsum('ab,acd,bce->de', v, tensors[i].conj(), tensors[i])
                
                # 数值稳定策略：每步收缩后归一化v，记录归一化因子
                # 移除break条件，确保处理所有张量
                current_norm = tc.norm(v) + 1e-16
                
                # 即使current_norm很小，也继续处理，确保计算完整性
                v = v / current_norm
                norm_factors.append(current_norm)
            
            # 计算总归一化因子（使用张量进行计算）
            total_norm = tc.tensor(1.0, dtype=self.dtype, device=self.device)
            for factor in norm_factors:
                total_norm *= factor
            
            # 开边界条件下，我们可以归一化任何一个张量，这里选择第一个张量
            norm_factor = tc.sqrt(total_norm)
            tensors[0] = tensors[0] / norm_factor
        
        elif self.boundary == 'periodic':
            # 周期边界条件归一化
            v0 = tc.eye(tensors[0].shape[0], dtype=tensors[0].dtype, device=tensors[0].device)
            v1 = tc.eye(tensors[0].shape[0], dtype=tensors[0].dtype, device=tensors[0].device)
            v = tc.kron(v0, v1).reshape([tensors[0].shape[0], tensors[0].shape[0],
                                        tensors[0].shape[0], tensors[0].shape[0]])
            norm_list = []
            
            for n in range(len(tensors)):
                v = tc.einsum('uvap,adb,pdq->uvbq', v, tensors[n].conj(), tensors[n])
                
                # 数值稳定策略：每步收缩后归一化v，记录归一化因子
                # 移除break条件，确保处理所有张量
                current_norm = tc.norm(v) + 1e-16
                
                # 即使current_norm很小，也继续处理，确保计算完整性
                v = v / current_norm
                norm_list.append(current_norm)
            
            if v.numel() > 1:
                norm1 = tc.einsum('acac->', v)
                norm_list.append(norm1 + 1e-10)
            else:
                norm_list.append(v[0, 0, 0, 0] + 1e-10)
            
            # 计算总归一化因子（使用张量进行计算）
            total_norm = tc.tensor(1.0, dtype=self.dtype, device=self.device)
            for x in norm_list:
                total_norm = total_norm * x
            
            # 周期边界条件下，归一化第一个张量
            norm_factor = tc.sqrt(tc.abs(total_norm))
            tensors[0] = tensors[0] / norm_factor
        
        if return_mps:
            # 返回新的MPS实例，保持计算图
            return MPS(tensors=tensors, boundary=self.boundary)
        else:
            # 返回张量列表（兼容旧代码）
            return tensors

    def contract_mps(self):
        """收缩整个 MPS 为单个张量"""
        tensor = self.tensors[0]
        for A in self.tensors[1:]:
            tensor = tc.einsum('...i,ijk->...jk', tensor, A)
        
        if self.boundary == 'open':
            return tensor.squeeze()
        elif self.boundary == 'periodic':
            return tc.einsum('i...i->...', tensor)

    def add_single_opr(self, op, site):
        """在指定位置添加单体操作"""
        return tc.einsum('abc, bd -> adc', self.tensors[site], op)

    def inner_product(self, others, form='log', boundary='open'):
        tensors0 = self.tensors
        tensors1 = others.tensors
        assert tensors0[0].shape[0] == tensors0[-1].shape[-1]
        assert tensors1[0].shape[0] == tensors1[-1].shape[-1]
        assert len(tensors0) == len(tensors1)
        
        if boundary == 'open':
            v = tc.ones((1, 1), dtype=self.dtype, device=self.device)
            norm_tensor = tc.ones(len(tensors0), dtype=self.dtype, device=self.device)
            
            for i in range(len(tensors0)):
                v = tc.einsum('ab,acd,bce->de', v, tensors0[i].conj(), tensors1[i])
                n = tc.norm(v) + 1e-16
                if n.item() < 1e-12:
                    return tc.tensor(0.0, dtype=self.dtype, device=self.device)
                v = v / n
                norm_tensor[i] = n
            
            norms = tc.prod(norm_tensor)
            return norms * v.squeeze()

        elif boundary == 'periodic':
            v0 = tc.eye(tensors0[0].shape[0], dtype=tensors0[0].dtype, device=tensors0[0].device)
            v1 = tc.eye(tensors1[0].shape[0], dtype=tensors0[0].dtype, device=tensors0[0].device)
            v = tc.kron(v0, v1).reshape([tensors0[0].shape[0], tensors1[0].shape[0],
                                        tensors0[0].shape[0], tensors1[0].shape[0]])
            norm_list = list()
            for n in range(len(tensors0)):
                v = tc.einsum('uvap,adb,pdq->uvbq', v, tensors0[n].conj(), tensors1[n])
                norm_list.append(v.norm() + 1e-10)
                v = v / norm_list[-1]
            
            if v.numel() > 1:
                norm1 = tc.einsum('acac->', v)
                norm_list.append(norm1 + 1e-10)
            else:
                norm_list.append(v[0, 0, 0, 0] + 1e-10)
            
            if form == 'log':
                norm = 0.0
                for x in norm_list:
                    norm = norm + tc.log(x.abs())
            elif form == 'list':
                return norm_list
            else:
                norm = 1.0
                for x in norm_list:
                    norm = norm * x
            return norm
    
    def energy_with_MPO(self, mpo):
        """
        计算MPS与MPO的能量期望值⟨ψ|H|ψ⟩
        
        参数:
        mpo: MPO实例，包含待计算的哈密顿量
        
        返回:
        能量期望值和MPS张量列表
        """
        check_paras(self, mpo)
        
        # 核心实现：正确处理MPO张量的收缩
        # 确保每个操作数的索引数量与它的维度数量匹配
        # - A_dag: [a×s×b] → 3维，需要3个索引
        # - M: [c×d×s×s'] → 4维，需要4个索引
        # - A: [a×s'×b] → 3维，需要3个索引
        
        # 获取MPS和MPO张量
        mps_tensors = self.tensors
        mpo_tensors = mpo.tensors
        
        # 初始化左环境为标量
        left_env = tc.tensor(1.0, dtype=self.dtype, device=self.device)
        
        # 遍历所有站点，进行左到右收缩
        for i in range(self.length):
            A = mps_tensors[i]          # MPS张量: [a×s×b]
            A_dag = A.conj()             # 共轭MPS张量: [a×s×b]
            M = mpo_tensors[i]           # MPO张量: [c×d×s×s']
            
            # 确保MPO张量是4维的
            if len(M.shape) != 4:
                raise ValueError(f"MPO张量必须是4维的，当前形状为{M.shape}")
            
            # 正确的索引数量：
            # - A_dag (3维) → 3个索引
            # - M (4维) → 4个索引
            # - A (3维) → 3个索引
            if i == 0:  # 左边界
                # 左边界MPO张量形状: [1×D×d×d]
                # 收缩：A_dag × M × A
                # 结果形状: [D]
                left_env = tc.einsum('asb, cdss, aeb -> d', A_dag, M, A)
            elif i == self.length - 1:  # 右边界
                # 右边界MPO张量形状: [D×1×d×d]
                # 收缩：left_env × A_dag × M × A
                # 结果形状: 标量
                energy = tc.einsum('d, asb, dess, aeb ->', left_env, A_dag, M, A)
            else:  # 中间站点
                # 中间MPO张量形状: [D×D×d×d]
                # 收缩：left_env × A_dag × M × A
                # 结果形状: [D]
                left_env = tc.einsum('d, asb, dess, aeb -> e', left_env, A_dag, M, A)
        
        return energy, self.tensors
        
    def to_normalize(self):
        """用于训练后的结果归一化"""
        psi = self.clone_mps()
        with tc.no_grad():
            normalized_tensors = self.normalize()
            for i in range(self.length):
                psi.tensors[i].data = normalized_tensors[i]
        return psi
        
    def measurement_calcu(self, ops, sites):
        """用于计算训练后测量值"""
        assert self.dim == ops[0].shape[0]
        assert len(ops) == len(sites), "操作和位置列表长度不匹配"

        measurement = 0
        with tc.no_grad():
            psi = self.to_normalize()
            psi_ = psi.clone_mps()
            for op, site in zip(ops, sites):
                psi_.tensors[site] = psi_.add_single_opr(op, site)
            measurement = psi.inner_product(psi_, boundary=self.boundary)
        return measurement
                 

class MPO():
    def __init__(self, length, dim, device=choose_device(), dtype=tc.complex128, name=None):
        super().__init__()
        self.length = length
        self.dim = dim
        self.device = device
        self.dtype = dtype
        self.name = name
        self.tensors = None
        
    def generate_spin_operators(self):
        """生成自旋维度为dim的所有自旋算符矩阵"""
        S = (self.dim - 1) / 2.0
        m_values = np.arange(S, -S-1, -1)

        Sz = tc.zeros((self.dim, self.dim), dtype=self.dtype, device=self.device)
        S_plus = tc.zeros((self.dim, self.dim), dtype=self.dtype, device=self.device)
        S_minus = tc.zeros((self.dim, self.dim), dtype=self.dtype, device=self.device)
        
        for i, m in enumerate(m_values):
            Sz[i, i] = m
        
        for i, m in enumerate(m_values):
            if i > 0:
                coeff = np.sqrt(S * (S + 1) - m * (m + 1))
                S_plus[i-1, i] = coeff

            if i < self.dim - 1:
                coeff = np.sqrt(S * (S + 1) - m * (m - 1))
                S_minus[i+1, i] = coeff
        
        Sx = (S_plus + S_minus) / 2.0
        Sy = (S_plus - S_minus) / (2.0j)
        I = tc.eye(self.dim, dtype=self.dtype, device=self.device)
        
        return {
            'I': I,
            'Sx': Sx,
            'Sy': Sy, 
            'Sz': Sz,
            'S_plus': S_plus,
            'S_minus': S_minus,
            'S': S,
            'dim': self.dim
        }
        
    def get_spin_operator(self, op_name):
        """获取指定的自旋算符"""
        operators = self.generate_spin_operators()
        
        if op_name in operators:
            return operators[op_name]
        else:
            raise ValueError(f"未知的算符名称: {op_name}")
    
    def get_pauli_matrices(self):
        """获取泡利矩阵 (自旋-1/2)"""
        if self.dim != 2:
            raise ValueError("泡利矩阵只适用于自旋-1/2系统 (dim=2)")
        
        operators = self.generate_spin_operators()
        pauli_x = 2 * operators['Sx']
        pauli_y = 2 * operators['Sy'] 
        pauli_z = 2 * operators['Sz']
        
        return {
            'I': operators['I'],
            'sigma_x': pauli_x,
            'sigma_y': pauli_y,
            'sigma_z': pauli_z,
            'sigma_plus': pauli_x + 1j * pauli_y,
            'sigma_minus': pauli_x - 1j * pauli_y
        }
    
    def generate_TFI_MPO(self, J, h):
        """生成横场Ising模型的MPO张量列表"""
        self.tensors = [None] * self.length
        d = self.dim
        D = 3  # 标准横场Ising模型MPO使用D=3的虚拟维度
        
        # 获取泡利矩阵
        pauli = self.get_pauli_matrices()
        I = pauli['I']
        sigma_z = pauli['sigma_z']
        sigma_x = pauli['sigma_x']
        
        # 为每个站点构造MPO张量
        for i in range(self.length):
            # 初始化MPO张量：形状为(D, D, d, d)
            mpo_tensor = tc.zeros((D, D, d, d), dtype=self.dtype, device=self.device)
            
            # 标准横场Ising模型MPO构造
            # 参考：https://arxiv.org/abs/cond-mat/0407066
            
            # 单位算符通道
            mpo_tensor[0, 0, :, :] = I
            
            # σ^z 相互作用通道
            mpo_tensor[0, 1, :, :] = sigma_z
            
            # -hσ^x 外场通道
            mpo_tensor[0, 2, :, :] = -h * sigma_x
            
            # -Jσ^z 相互作用通道
            mpo_tensor[1, 2, :, :] = -J * sigma_z
            
            # 单位算符通道（用于传递到下一个站点）
            mpo_tensor[2, 2, :, :] = I
            
            # 边界条件处理
            if i == 0:
                # 左边界：只保留第一个左虚拟维度
                mpo_tensor = mpo_tensor[0:1, :, :, :]
            elif i == self.length - 1:
                # 右边界：只保留最后一个右虚拟维度
                mpo_tensor = mpo_tensor[:, 2:3, :, :]
            
            self.tensors[i] = mpo_tensor
        
        return self
    

# 完整的AD_DMRG方法类
class DMRGnet(nn.Module):
    """完整的DMRG收缩网络"""
    def __init__(self, mps, mpo):
        super().__init__()
        check_paras(mps, mpo)
        self.length = mps.length
        self.dim = mps.dim
        self.chi = mps.chi
        self.device = mps.device
        self.dtype = mps.dtype
        
        # 初始化MPS和MPO
        self.mps = mps
        self.mpo = mpo
        
    def forward(self):
        """前向传播：计算能量期望值"""
        energy, _ = self.mps.energy_with_MPO(self.mpo)
        # 确保返回的是标量张量
        return tc.sum(energy.real)  # 使用sum确保是标量
    
    
    def optimize(self, steps=10000, lr=1e-4):
        """优化MPS以找到基态"""
        optimizer = tc.optim.Adam(self.mps.parameters(), lr=lr)
        energies = []
        
        pbar = tqdm(range(steps), desc="Variational DMRG", unit="step")
        
        for step in pbar:
            # 前向传播计算能量
            energy = self.forward()
            
            # 反向传播
            optimizer.zero_grad()
            energy.backward()
            optimizer.step()
            
            # 归一化MPS（每10步归一化一次，避免梯度消失）
            if step % 10 == 0:
                with tc.no_grad():
                    # 使用return_mps=False获取张量列表
                    normalized_tensors = self.mps.normalize(return_mps=False)
                    for i in range(self.mps.length):
                        self.mps.tensors[i].data = normalized_tensors[i]
            
            energies.append(energy.item())
            pbar.set_postfix({'Energy': f'{energy.item()}'})
        
        return energies
    
    def plot_energy_curve(self, energies):
        """绘制能量变化曲线"""
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, len(energies)+1), energies, label='Energy variation curve')
        plt.xlabel('Steps')
        plt.ylabel('Ground state Energy')
        plt.title('Energy Variation Curve')
        plt.legend()
        plt.grid()
        plt.show()

def AD_DMRG(mps, mpo, N, d, chi, device = choose_device(), dtype=tc.complex128):
    if mps is None and mpo is None:
        mps = MPS(length=N, dim=d, chi=chi, device=device, dtype=dtype)
        mpo = MPO(length=N, dim=d, device=device, dtype=dtype)
    else:
        mps = mps
        mpo = mpo

    model = DMRGnet(mps, mpo)
    energy = model.optimize(steps=5000, lr=1e-3)
    model.plot_energy_curve(energy)
    return model.mps
