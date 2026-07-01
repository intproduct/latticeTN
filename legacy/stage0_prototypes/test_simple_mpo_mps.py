import torch as tc
import numpy as np
from AD_MPS_fixed import MPS, MPO

# 设置计算设备
device = tc.device('cuda' if tc.cuda.is_available() else 'cpu')

def exact_energy_2site_ising(J, h, psi):
    """
    精确计算两粒子横场Ising模型的能量期望值
    哈密顿量：H = -J σ_1^zσ_2^z - h (σ_1^x + σ_2^x)
    """
    # 将MPS转换为状态向量
    psi_mat = psi.contract_mps()
    
    # 重塑为向量形状：(4, 1)
    psi_vec = psi_mat.reshape(-1, 1)
    
    # 泡利矩阵
    sigma_x = tc.tensor([[0, 1], [1, 0]], dtype=tc.complex128, device=device)
    sigma_z = tc.tensor([[1, 0], [0, -1]], dtype=tc.complex128, device=device)
    I = tc.eye(2, dtype=tc.complex128, device=device)
    
    # 构造哈密顿量
    term1 = -J * tc.kron(sigma_z, sigma_z)
    term2 = -h * tc.kron(sigma_x, I)
    term3 = -h * tc.kron(I, sigma_x)
    H = term1 + term2 + term3
    
    # 计算能量期望值
    psi_conj = psi_vec.conj().T
    energy = tc.matmul(psi_conj, tc.matmul(H, psi_vec))
    
    # 归一化
    norm = tc.matmul(psi_conj, psi_vec)
    energy = energy / norm
    
    return energy

def mpo_energy_2site_ising(J, h, psi):
    """
    使用MPO计算两粒子横场Ising模型的能量期望值
    """
    # 创建MPO
    mpo = MPO(length=2, dim=2, device=device)
    mpo.generate_TFI_MPO(J=J, h=h)
    
    # 计算能量
    energy, _ = psi.energy_with_MPO(mpo)
    
    # 归一化
    norm = psi.inner_product(psi)
    energy = energy / norm
    
    return energy

def test_simple_mpo_mps():
    """
    简单测试MPO与MPS的收缩
    """
    # 设置参数
    J = 1.0
    h = 0.5
    
    print(f"两粒子横场Ising模型参数：J={J}, h={h}")
    
    # 创建一个简单的MPS（全1张量）
    tensors = []
    # 第一个张量：(1, 2, 2)
    tensors.append(tc.ones((1, 2, 2), device=device, dtype=tc.complex128, requires_grad=True))
    # 第二个张量：(2, 2, 1)
    tensors.append(tc.ones((2, 2, 1), device=device, dtype=tc.complex128, requires_grad=True))
    
    psi = MPS(tensors=tensors, boundary='open')
    
    # 计算精确能量
    exact_e = exact_energy_2site_ising(J, h, psi)
    print(f"精确能量：{exact_e.item()}")
    
    # 使用MPO计算能量
    mpo_e = mpo_energy_2site_ising(J, h, psi)
    print(f"MPO能量：{mpo_e.item()}")
    
    # 比较结果
    diff = abs((exact_e - mpo_e).item())
    print(f"差异：{diff}")
    
    if diff < 1e-6:
        print("✓ MPO能量计算正确！")
        return True
    else:
        print("✗ MPO能量计算错误！")
        return False

if __name__ == "__main__":
    test_simple_mpo_mps()
