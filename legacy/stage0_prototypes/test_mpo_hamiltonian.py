import torch as tc
import numpy as np
from AD_MPS_fixed import MPO

# 设置计算设备
device = tc.device('cuda' if tc.cuda.is_available() else 'cpu')

def exact_hamiltonian_2site_ising(J, h):
    """
    生成两粒子横场Ising模型的精确哈密顿量矩阵
    """
    # 泡利矩阵
    sigma_x = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    sigma_z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
    I = np.eye(2, dtype=np.complex128)
    
    # 构造哈密顿量
    term1 = np.kron(sigma_z, sigma_z)  # σ1^zσ2^z
    term2 = np.kron(sigma_x, I)        # σ1^x
    term3 = np.kron(I, sigma_x)        # σ2^x
    
    H = -J * term1 - h * (term2 + term3)
    
    return H

def mpo_to_matrix(mpo):
    """
    将MPO转换为完整的哈密顿量矩阵
    仅适用于小规模系统（N<=3）
    """
    N = mpo.length
    d = mpo.dim
    
    assert N <= 3, "此函数仅适用于N<=3的系统"
    
    # 获取MPO张量
    tensors = mpo.tensors
    
    # 打印MPO张量形状，用于调试
    print(f"MPO张量形状：")
    for i, tensor in enumerate(tensors):
        print(f"张量 {i} 形状：{tensor.shape}")
    
    if N == 2:
        # 两粒子系统：收缩MPO张量
        # 第一个MPO张量形状：(A, B, d, d)
        # 第二个MPO张量形状：(B, C, d, d)
        A, B, d1, d2 = tensors[0].shape
        B2, C, d3, d4 = tensors[1].shape
        
        assert d1 == d2 == d3 == d4 == d, "物理维度不匹配"
        assert B == B2, "MPO键维度不匹配"
        
        # 收缩两个MPO张量
        # 索引：a(左虚拟), b(右虚拟), i(入物理), j(出物理)
        contracted = tc.einsum('abij,bckl->ackjil', tensors[0], tensors[1])
        
        # 重塑为完整的哈密顿量矩阵
        contracted_reshaped = contracted.reshape(A*C, d*d, d*d)
        
        # 对于开放边界条件，A和C都应该是1
        assert A == 1 and C == 1, f"开放边界条件下，左右虚拟维度应该为1，实际为A={A}, C={C}"
        
        H = contracted_reshaped[0]
        
    elif N == 3:
        # 三粒子系统：收缩三个MPO张量
        A, B, d1, d2 = tensors[0].shape
        B2, C, d3, d4 = tensors[1].shape
        C2, D, d5, d6 = tensors[2].shape
        
        assert d1 == d2 == d3 == d4 == d5 == d6 == d, "物理维度不匹配"
        assert B == B2 and C == C2, "MPO键维度不匹配"
        
        # 先收缩前两个MPO张量
        contracted1 = tc.einsum('abij,bckl->ackjil', tensors[0], tensors[1])
        # 重塑为形状：(A, C, d*d, d*d)
        contracted1_reshaped = contracted1.reshape(A, C, d*d, d*d)
        
        # 再与第三个MPO张量收缩
        contracted2 = tc.einsum('acij,cdekl->adekljik', contracted1_reshaped, tensors[2])
        # 重塑为完整的哈密顿量矩阵
        H = contracted2.reshape(d**3, d**3)
    
    return H

def test_mpo_vs_exact():
    """
    比较MPO生成的哈密顿量与精确哈密顿量
    """
    # 设置参数
    N = 2
    d = 2
    J = 1.0
    h = 0.5
    
    print(f"测试MPO vs 精确哈密顿量：")
    print(f"系统大小：{N}")
    print(f"参数：J={J}, h={h}")
    
    # 生成精确哈密顿量
    exact_H = exact_hamiltonian_2site_ising(J, h)
    
    # 生成MPO
    mpo = MPO(length=N, dim=d, device=device)
    mpo.generate_TFI_MPO(J=J, h=h)
    
    # 将MPO转换为矩阵
    mpo_H = mpo_to_matrix(mpo).cpu().detach().numpy()
    
    # 比较两者
    print(f"\n精确哈密顿量：")
    print(np.round(exact_H, 6))
    
    print(f"\nMPO转换的哈密顿量：")
    print(np.round(mpo_H, 6))
    
    # 计算差异
    diff = np.max(np.abs(exact_H - mpo_H))
    print(f"\n最大差异：{diff:.10f}")
    
    if diff < 1e-6:
        print("\n✓ MPO正确实现了哈密顿量！")
        return True
    else:
        print("\n✗ MPO实现的哈密顿量与精确解不符！")
        return False

if __name__ == "__main__":
    test_mpo_vs_exact()
