import torch as tc
import numpy as np
from AD_MPS_fixed import MPS, MPO, DMRGnet

# 设置计算设备
device = tc.device('cuda' if tc.cuda.is_available() else 'cpu')

def exact_diagonalization_2site_ising(J, h):
    """
    两粒子横场Ising模型的精确对角化
    哈密顿量：H = -J σ_1^zσ_2^z - h (σ_1^x + σ_2^x)
    采用开放边界条件
    """
    # 泡利矩阵
    sigma_x = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    sigma_z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
    I = np.eye(2, dtype=np.complex128)
    
    # 构造哈密顿量
    # σ_1^zσ_2^z
    term1 = np.kron(sigma_z, sigma_z)
    # σ_1^x
    term2 = np.kron(sigma_x, I)
    # σ_2^x
    term3 = np.kron(I, sigma_x)
    
    H = -J * term1 - h * (term2 + term3)
    
    # 计算本征值和本征向量
    eigenvalues, eigenvectors = np.linalg.eigh(H)
    
    # 返回基态能量
    return eigenvalues[0]

def run_dmrg_2site_ising(J, h):
    """
    使用DMRG计算两粒子横场Ising模型的基态能量
    """
    # 设置参数
    N = 2
    d = 2
    chi = 5
    steps = 3000
    lr = 5e-4
    
    # 创建MPS
    mps = MPS(length=N, dim=d, chi=chi, device=device, boundary='open')
    
    # 创建MPO
    mpo = MPO(length=N, dim=d, device=device)
    mpo.generate_TFI_MPO(J=J, h=h)
    
    # 创建DMRG网络
    model = DMRGnet(mps, mpo)
    
    # 优化
    energies = model.optimize(steps=steps, lr=lr)
    
    # 返回最终能量
    return energies[-1], energies

if __name__ == "__main__":
    # 设置参数
    J = 1.0
    h = 0.5
    
    print(f"两粒子横场Ising模型参数：J={J}, h={h}")
    
    # 精确对角化
    exact_energy = exact_diagonalization_2site_ising(J, h)
    print(f"精确对角化基态能量：{exact_energy:.10f}")
    
    # DMRG计算
    print("\n开始DMRG计算...")
    dmrg_energy, energies = run_dmrg_2site_ising(J, h)
    print(f"DMRG基态能量：{dmrg_energy:.10f}")
    
    # 比较结果
    print(f"\n相对误差：{abs((dmrg_energy - exact_energy) / exact_energy):.10f}")
    print(f"绝对误差：{abs(dmrg_energy - exact_energy):.10f}")
    
    # 打印能量演化的最后几个值
    print(f"\n最后10步能量值：{energies[-10:]}")
