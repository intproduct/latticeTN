import torch as tc
import numpy as np
import math as mt
from tqdm import tqdm

def choose_device():
    if tc.cuda.is_available():
        return 'cuda'
    else:
        return 'cpu'

def clone_func(tensors):
    """辅助函数：克隆 MPS 张量列表"""
    return [tensor.clone() for tensor in tensors]

class MPS:

    def __init__(self, length, dim, chi, device='cpu', dtype=tc.complex128, boundary='open'):
        super().__init__()
        self.length = length
        self.dim = dim
        self.chi = chi
        self.device = device
        self.dtype = dtype
        self.boundary = boundary

        if self.boundary == 'open':
            self.tensors = [tc.randn((1, dim, chi), device=device, dtype=dtype, requires_grad=True)] + [tc.randn((chi, dim, chi), device=device, dtype=dtype, requires_grad=True) for _ in range(length-2)] + [tc.randn((chi, dim, 1), device=device, dtype=dtype, requires_grad=True)]
        elif self.boundary == 'periodic':
            self.tensors = [tc.randn((chi, dim, chi), device=device, dtype=dtype, requires_grad=True) for _ in range(length)]

    def clone_tensors(self):
        return clone_func(self.tensors)

    def clone_mps(self):
        psi = MPS(self.length, self.dim, self.chi, self.device, self.dtype)
        psi.tensors = clone_func(self.tensors)
        return psi

    def normalize(self):
        mps0 = self.clone_tensors()
        #v = tc.eye(mps0[0].shape[0]**2, dtype=tc.float64, device=paras['device']).reshape([mps0[0].shape[0]] * 4)
        v = tc.ones((mps0[0].shape[0], mps0[0].shape[0]), dtype=self.dtype, device=self.device)
        mps_n = []
        norm = tc.zeros(len(self.tensors), dtype=tc.float64, device=self.device)
        for i in range(len(self.tensors)):
            #v = tc.einsum('uvab,acd,bce->uvde', v, mps0[i].conj(), mps0[i])
            v = tc.einsum('ab,acd,bce->de', v, mps0[i].conj(), mps0[i])
            #n = tc.einsum('abcd->', v)
            n = tc.norm(v) + 1e-12
            mps_n.append(self.tensors[i] / tc.sqrt(n))
            v = v / n
            norm[i] = n
        norms = tc.prod(norm)
        
        # 保持计算图：原地更新每个张量而不是重新赋值列表
        for i in range(len(self.tensors)):
            self.tensors[i].data = mps_n[i].data  # 只更新数据，保持计算图
        
        return norms

    def contract_mps(self):
        """收缩整个 MPS 为单个张量"""
        tensor = self.tensors[0]
        for A in self.tensors[1:]:
            #print("收缩前tensor：",tensor.shape)
            #print("A：",A.shape)
            tensor = tc.einsum('...i,ijk->...jk', tensor, A)
            #print("收缩后tensor：",tensor.shape)
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
                
                # 添加安全保护
                n = tc.norm(v) + 1e-16  # 防止除以零
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
            if form == 'log':  # 返回模方的log，舍弃符号
                norm = 0.0
                for x in norm_list:
                    norm = norm + tc.log(x.abs())
            elif form == 'list':  # 返回列表
                return norm_list
            else:  # 直接返回模方
                norm = 1.0
                for x in norm_list:
                    norm = norm * x
            return norm

    def energy_calcu(self, ops, sites):
        """计算给定操作和位置的能量期望值"""
        assert len(ops) == len(sites), "操作和位置列表长度不匹配"
        energy = 0
        psi = self.clone_mps()
        for op, site in zip(ops, sites):
            psi.tensors[site] = psi.add_single_opr(op, site)
        energy = self.inner_product(psi, boundary=self.boundary)
        return energy
    
    def energy_with_MPO(self, mpo):
        assert self.length == mpo.length, "MPS和MPO长度不匹配"
    
        if self.boundary == 'open':
             v = tc.ones((1, 1, 1), dtype=self.dtype, device=self.device)
             norm_tensor = tc.ones(self.length, dtype=self.dtype, device=self.device)
            
             for i in range(self.length):
                 #print(f"shape of self.tensors[{i}]:", self.tensors[i].shape)
                 #print(f"shape of mpo.tensors[{i}]:", mpo.tensors[i].shape)
                 v = tc.einsum('abc, adi, cek, bjde -> ijk', v, self.tensors[i].conj(), self.tensors[i], mpo.tensors[i])
                 # 添加安全保护
                 n = tc.norm(v) + 1e-16  # 防止除以零
                 if n.item() < 1e-12:
                    return tc.tensor(0.0, dtype=self.dtype, device=self.device)
                 v = v / n
                 norm_tensor[i] = n
            
             norms = tc.prod(norm_tensor)
             return norms * v.squeeze()
        
        if self.boundary == 'periodic':
            assert self.length == mpo.length

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
            if form == 'log':  # 返回模方的log，舍弃符号
                norm = 0.0
                for x in norm_list:
                    norm = norm + tc.log(x.abs())
            elif form == 'list':  # 返回列表
                return norm_list
            else:  # 直接返回模方
                norm = 1.0
                for x in norm_list:
                    norm = norm * x
            return norm
            


class MPO:
    def __init__(self, length, dim, device='cpu', dtype=tc.complex128, name=None):
        self.length = length
        self.dim = dim
        self.device = device
        self.dtype = dtype
        self.name = name
        self.tensors = None
        
    def generate_spin_operators(self):
        """
        生成自旋维度为dim的所有自旋算符矩阵
        
        Returns:
            dict: 包含所有自旋算符的字典
        """
        # 计算自旋量子数 S
        S = (self.dim - 1) / 2.0
        
        # 生成 m 量子数：从 -S 到 +S
        m_values = np.arange(S, -S-1, -1)

        # 初始化算符矩阵
        Sz = tc.zeros((self.dim, self.dim), dtype=self.dtype, device=self.device)
        S_plus = tc.zeros((self.dim, self.dim), dtype=self.dtype, device=self.device)
        S_minus = tc.zeros((self.dim, self.dim), dtype=self.dtype, device=self.device)
        
        # Sz 算符：对角矩阵，对角元为 m 值
        for i, m in enumerate(m_values):
            Sz[i, i] = m
        
        # S+ 和 S- 算符
        for i, m in enumerate(m_values):
            if i > 0:  # S+ 连接 |m⟩ 和 |m+1⟩
                # S+|m⟩ = sqrt(S(S+1) - m(m+1))|m+1⟩
                coeff = np.sqrt(S * (S + 1) - m * (m + 1))
                S_plus[i-1, i] = coeff  # (m+1, m) 元素

            if i < self.dim - 1:  # S- 连接 |m⟩ 和 |m-1⟩
                # S-|m⟩ = sqrt(S(S+1) - m(m-1))|m-1⟩
                coeff = np.sqrt(S * (S + 1) - m * (m - 1))
                S_minus[i+1, i] = coeff  # (m-1, m) 元素
        
        # Sx 和 Sy 算符
        Sx = (S_plus + S_minus) / 2.0
        Sy = (S_plus - S_minus) / (2.0j)
        
        # 单位矩阵
        I = tc.eye(self.dim, dtype=self.dtype, device=self.device)
        
        return {
            'I': I,
            'Sx': Sx,
            'Sy': Sy, 
            'Sz': Sz,
            'S_plus': S_plus,
            'S_minus': S_minus,
            'S': S,  # 自旋量子数
            'dim': self.dim
        }
        
    def get_spin_operator(self, op_name):
        """
        获取指定的自旋算符
        
        Args:
            op_name: 算符名称 ('Sx', 'Sy', 'Sz', 'S_plus', 'S_minus', 'I')
        """
        operators = self.generate_spin_operators()
        
        if op_name in operators:
            self.tensor = operators[op_name]
            self.name = op_name
            return self.tensor
        else:
            raise ValueError(f"未知的算符名称: {op_name}")
    
    def get_pauli_matrices(self):
        """获取泡利矩阵 (自旋-1/2)"""
        if self.dim != 2:
            raise ValueError("泡利矩阵只适用于自旋-1/2系统 (dim=2)")
        
        operators = self.generate_spin_operators()
        
        # 泡利矩阵是自旋算符的2倍
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
        #初始化一些空列表
        self.tensors = [None] * self.length

        # 物理维度
        d = self.dim
        
        # 虚拟维度：开边界时为2
        D = 3
        
        for i in range(self.length):
            # 创建MPO张量 (左虚拟, 右虚拟, 物理入, 物理出)
            mpo_tensor = tc.zeros((D, D, d, d), dtype=self.dtype, device=self.device)
            
            # 单位算符部分
            mpo_tensor[0, 0, :, :] = self.get_spin_operator('I')
            mpo_tensor[2:, 2:, :, :] = self.get_spin_operator('I')

            #Sz算符
            mpo_tensor[0, 1, :, :] = mt.sqrt(J) * self.get_spin_operator('Sz')
            mpo_tensor[1, 2, :, :] = - mt.sqrt(J) * self.get_spin_operator('Sz')

            #Sx算符
            mpo_tensor[0, 2, :, :] = h * self.get_spin_operator('Sx')
            
            # 边界条件处理
            if i == 0:  # 左边界
                mpo_tensor = mpo_tensor[0:1, :, :, :]  # 只保留第一行
            if i == self.length - 1:  # 右边界
                mpo_tensor = mpo_tensor[:, 2:3, :, :]  # 只保留第二列
            
            self.tensors[i] = mpo_tensor
            
        return self



def calc_TFI_3(mps, opr, J, h):
    "计算三粒子横场Ising model的能量期望值"
    assert mps.length == 3, "MPS长度必须为3"
    assert opr.dim == 2, "操作维度必须为2"

    # 获取Sz算符
    sz_op = opr.get_spin_operator('Sz')
    sx_op = opr.get_spin_operator('Sx')
    id_op = opr.get_spin_operator('I')

    # 直接写出mpo列表：
    mpo1 = [sz_op, sz_op, id_op]
    mpo2 = [id_op, sz_op, sz_op]
    mpo3 = [sx_op, id_op, id_op]
    mpo4 = [id_op, sx_op, id_op]
    mpo5 = [id_op, id_op, sx_op]

    mpo_list1 = [mpo1, mpo2]
    mpo_list2 = [mpo3, mpo4, mpo5]

    #计算对应的能量：
    energy = 0
    for mpo in mpo_list1:
        energy += -J * mps.energy_calcu(mpo, [0, 1, 2])
    for mpo in mpo_list2:
        energy += h * mps.energy_calcu(mpo, [0, 1, 2])

    return energy

def main_calcu():
    #输出当前使用的设备，cuda可用则输出当前gpu，否则是当前的cpu
    print(f"Using device: {choose_device()}")
    mps_i = MPS(length=3, dim=2, chi=6, device=choose_device(), dtype=tc.float64)
    spinopr = MPO(length=3, dim=2, device=choose_device(), dtype=tc.float64)

    #输出mps_i和spinor的device
    print(f"mps_i device: {mps_i.device}")
    print(f"spinopr device: {spinopr.device}")

    opti = tc.optim.Adam(mps_i.tensors, lr = 1e-1)

    step_times = 20000

    pbar = tqdm(range(step_times), desc="Variational DMRG", unit="step")

    for step in pbar:
        norm = mps_i.normalize()
        # 计算能量期望值
        energy = calc_TFI_3(mps_i, spinopr, J=1.0, h=0.5)

        # 反向传播和优化
        opti.zero_grad()
        energy.backward()
        opti.step()

        pbar.set_postfix({'Energy': f'{energy.item()}'})

def test_if_norm():
    tmps = MPS(length=3, dim=2, chi=30, device=choose_device(), dtype=tc.float64, boundary='open')
    #print(tmps.tensors)

    norm = tmps.normalize()
    print("Normalization factor:", norm.item())

    norm2 = tmps.normalize()
    print("Normalization factor (second call):", norm2.item())

    psi = tmps.clone_mps()

    factor = tmps.inner_product(psi, boundary='open')
    print("Inner product factor:", factor)

def check_calcu():
    mps_i = MPS(length=3, dim=2, chi=10, device=choose_device(),dtype=tc.float64)
    spinopr = MPO(length=3, dim=2, device=choose_device(),dtype=tc.float64)
    
    with tc.no_grad():
        # 设置全向上态 |↑↑↑⟩
        # 第一个张量 [1, 2, 10]
        mps_i.tensors[0].data = tc.zeros((1, 2, 10), device=choose_device(),dtype=tc.float64)
        mps_i.tensors[0].data[0, 0, 0] = 1.0  # |↑⟩态
        
        # 中间张量 [10, 2, 10]
        mps_i.tensors[1].data = tc.zeros((10, 2, 10), device=choose_device(),dtype=tc.float64)
        mps_i.tensors[1].data[0, 0, 0] = 1.0  # |↑⟩态
        
        # 最后一个张量 [10, 2, 1]
        mps_i.tensors[2].data = tc.zeros((10, 2, 1), device=choose_device(),dtype=tc.float64)
        mps_i.tensors[2].data[0, 0, 0] = 1.0  # |↑⟩态
    
    # 计算内积确保归一化
    norm_factor = mps_i.inner_product(mps_i.clone_mps(), boundary='open')
    print(f"初始态内积: {norm_factor.item()}")

    tensor = mps_i.contract_mps()
    print("收缩后的张量形状:", tensor.shape)
    print(tensor.reshape(8,1,1).squeeze())
    
    # 计算能量
    energy = calc_TFI_3(mps_i, spinopr, J=1.0, h=0.5)
    print("|↑↑↑⟩ Energy:", energy.item())

def calc_TFI_with_MPO():
    mps_i = MPS(length=100, dim=2, chi=300, device=choose_device(),dtype=tc.float64)
    tfimpo = MPO(length=100, dim=2, device=choose_device(),dtype=tc.float64).generate_TFI_MPO(J=1.0, h=0.5)

    opti = tc.optim.Adam(mps_i.tensors, lr=1e-3)

    steps = 10000

    pbar = tqdm(range(steps), desc="Variational DMRG", unit="step")

    for step in pbar:
        norm = mps_i.normalize()
        energy = mps_i.energy_with_MPO(tfimpo)

        opti.zero_grad()
        energy.backward()
        opti.step()

        pbar.set_postfix({'Energy': f'{energy.item()}'})

if __name__ == "__main__":
    #main_calcu()
    #check_calcu()
    #test_if_norm()
    #tfi_model = MPO(length=3, dim=2, device=choose_device(), dtype=tc.float64)
    #tfi_model = tfi_model.generate_TFI_MPO(J=1.0, h=0.5)
    #print(tfi_model.tensors)
    calc_TFI_with_MPO()