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
    def __init__(self, tensors, length, dim, chi, device=choose_device(), dtype=tc.complex128, boundary='open'):
        super().__init__()
        if tensors is None:
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
            self.tensors = tensors
        
        # 将张量注册为参数
        self.tensors = nn.ParameterList([nn.Parameter(t, requires_grad=True) for t in tensors])

    def clone_tensors(self):
        return clone_func(self.tensors)

    def clone_mps(self):
        psi = MPS(self.length, self.dim, self.chi, self.device, self.dtype)
        # 复制参数值
        for i, param in enumerate(self.tensors):
            psi.tensors[i].data = param.data.clone()
        return psi

    def normalize(self):
        # 不要 clone，不要 detach
        normalize_tensors = []
        v = tc.ones((self.tensors[0].shape[0], self.tensors[0].shape[0]), dtype=self.dtype, device=self.device)
        for i in range(len(self.tensors)):
            v = tc.einsum('ab,acd,bce->de', v, self.tensors[i].conj(), self.tensors[i])
            n = tc.norm(v)
            normalize_tensors.append(self.tensors[i] / tc.sqrt(n))
            v = v / n
        return normalize_tensors

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
        assert self.length == mpo.length, "MPS和MPO长度不匹配"
    
        if self.boundary == 'open':
            v = tc.ones((1, 1, 1), dtype=self.dtype, device=self.device)
            norm_tensor = tc.ones(self.length, dtype=self.dtype, device=self.device)

            norm_tensors = self.normalize()
            
            for i in range(self.length):
                v = tc.einsum('abc, adi, cek, bjde -> ijk', v, 
                             norm_tensors[i].conj(), norm_tensors[i], mpo.tensors[i])
                n = tc.norm(v) + 1e-16
                if n.item() < 1e-12:
                    return tc.tensor(0.0, dtype=self.dtype, device=self.device)
                v = v / n
                norm_tensor[i] = n
            
            norms = tc.prod(norm_tensor)
            return norms * v.squeeze(), norm_tensors
        
        elif self.boundary == 'periodic':
            v0 = tc.eye(self.tensors[0].shape[0], dtype=self.dtype, device=self.device)
            v1 = tc.eye(self.tensors[0].shape[0], dtype=self.dtype, device=self.device)
            v = tc.kron(v0, v1).reshape([self.tensors[0].shape[0], self.tensors[0].shape[0],
                                        self.tensors[0].shape[0], self.tensors[0].shape[0]])
            norm_list = list()
            
            for n in range(self.length):
                v = tc.einsum('uvap,adb,pdq->uvbq', v, self.tensors[n].conj(), self.tensors[n])
                norm_list.append(v.norm() + 1e-10)
                v = v / norm_list[-1]
            
            if v.numel() > 1:
                norm1 = tc.einsum('acac->', v)
                norm_list.append(norm1 + 1e-10)
            else:
                norm_list.append(v[0, 0, 0, 0] + 1e-10)
            
            norm = 1.0
            for x in norm_list:
                norm = norm * x
            return norm
        
    def to_normalize(self):
        "用于训练后的结果归一化"
        psi = self.clone_mps()
        with tc.no_grad():
            normalized_tensors = self.normalize()
            for i in range(self.length):
                psi.tensors[i].data = normalized_tensors[i]
        return psi
        
    def measurement_calcu(self, ops, sites):
        "用于计算训练后测量值"
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
        D = 3  # 虚拟维度
        
        for i in range(self.length):
            mpo_tensor = tc.zeros((D, D, d, d), dtype=self.dtype, device=self.device)
            
            # 单位算符部分
            mpo_tensor[0, 0, :, :] = self.get_spin_operator('I')
            mpo_tensor[2:, 2:, :, :] = self.get_spin_operator('I')

            # Sz算符
            mpo_tensor[0, 1, :, :] =  self.get_spin_operator('Sz')
            mpo_tensor[1, 2, :, :] = -J * self.get_spin_operator('Sz')

            # Sx算符
            mpo_tensor[0, 2, :, :] = -h * self.get_spin_operator('Sx')
            
            # 边界条件处理
            if i == 0:
                mpo_tensor = mpo_tensor[0:1, :, :, :]
            if i == self.length - 1:
                mpo_tensor = mpo_tensor[:, 2:3, :, :]
            
            self.tensors[i] = mpo_tensor
            
        return self
    

#任意的AD——DMRG方法类
class DMRGnet(nn.Module):
    """任意的DMRG收缩网络"""
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
        return self.mps.energy_with_MPO(self.mpo)
    
    
    def optimize(self, steps=10000, lr=1e-4):
        """优化MPS以找到基态"""
        optimizer = tc.optim.Adam(self.mps.parameters(), lr=lr)
        energies = []
        
        pbar = tqdm(range(steps), desc="Variational DMRG", unit="step")
        
        for step in pbar:
            energy, _ = self.forward()
            
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
    