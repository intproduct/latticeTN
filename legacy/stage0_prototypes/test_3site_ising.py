import torch as tc
import numpy as np
from AD_MPS_fixed import MPS, MPO, DMRGnet

# 设置计算设备
device = tc.device('cuda' if tc.cuda.is_available() else 'cpu')

def exact_diagonalization_3site_ising(J, h, boundary='open'):
    """
    三粒子横场Ising模型的精确对角化
    哈密顿量：H = -J Σσ_i^zσ_{i+1}^z - h Σσ_i^x
    支持开放边界条件和周期边界条件
    """
    # 泡利矩阵
    sigma_x = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    sigma_z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
    I = np.eye(2, dtype=np.complex128)
    
    # 构造单粒子算符在三粒子希尔伯特空间中的表示
    
    # σ_1^zσ_2^z
    term1 = np.kron(np.kron(sigma_z, sigma_z), I)
    # σ_2^zσ_3^z
    term2 = np.kron(np.kron(I, sigma_z), sigma_z)
    
    if boundary == 'periodic':
        # σ_3^zσ_1^z （周期边界条件）
        term3 = np.kron(np.kron(sigma_z, I), sigma_z)
        # 构造哈密顿量（周期边界条件）
        H = -J * (term1 + term2 + term3) - h * (np.kron(np.kron(sigma_x, I), I) + np.kron(np.kron(I, sigma_x), I) + np.kron(np.kron(I, I), sigma_x))
    else:
        # 构造哈密顿量（开放边界条件）
        H = -J * (term1 + term2) - h * (np.kron(np.kron(sigma_x, I), I) + np.kron(np.kron(I, sigma_x), I) + np.kron(np.kron(I, I), sigma_x))
    
    # 计算本征值和本征向量
    eigenvalues, eigenvectors = np.linalg.eigh(H)
    
    # 返回基态能量
    return eigenvalues[0]

def run_dmrg_3site_ising(J, h):
    """
    使用DMRG计算三粒子横场Ising模型的基态能量
    """
    # 设置参数
    N = 3
    d = 2
    chi = 10
    steps = 5000  # 增加优化步骤
    lr = 1e-5     # 减小学习率
    
    # 创建MPS - 确保初始化合理
    mps = MPS(length=N, dim=d, chi=chi, device=device, boundary='open')
    
    # 创建MPO
    mpo = MPO(length=N, dim=d, device=device)
    mpo.generate_TFI_MPO(J=J, h=h)
    
    # 创建DMRG网络
    model = DMRGnet(mps, mpo)
    
    # 优化
    energies = model.optimize(steps=steps, lr=lr)
    
    # 返回最终能量
    return energies[-1]

if __name__ == "__main__":
    # 设置参数
    J = 1.0
    h = 0.5
    boundary = 'open'  # 统一使用开放边界条件
    
    print(f"三粒子横场Ising模型参数：J={J}, h={h}, 边界条件={boundary}")
    
    # 精确对角化
    exact_energy = exact_diagonalization_3site_ising(J, h, boundary=boundary)
    print(f"精确对角化基态能量：{exact_energy:.10f}")
    
    # DMRG计算
    print("\n开始DMRG计算...")
    dmrg_energy = run_dmrg_3site_ising(J, h)
    print(f"DMRG基态能量：{dmrg_energy:.10f}")
    
    # 比较结果
    print(f"\n相对误差：{abs((dmrg_energy - exact_energy) / exact_energy):.10f}")
    print(f"绝对误差：{abs(dmrg_energy - exact_energy):.10f}")
