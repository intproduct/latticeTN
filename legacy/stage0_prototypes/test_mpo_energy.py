import torch as tc
import numpy as np
from AD_MPS_fixed import MPS, MPO

# 测试MPO能量计算，特别是周期性边界条件
def test_mpo_energy():
    print("测试MPO能量计算...")
    
    # 测试参数
    length = 5
    dim = 2
    chi = 4
    device = 'cpu'
    dtype = tc.complex128
    
    # 测试开边界条件
    print("\n=== 测试开边界条件 ===")
    test_mpo_energy_with_boundary('open', length, dim, chi, device, dtype)
    
    # 测试周期边界条件
    print("\n=== 测试周期边界条件 ===")
    test_mpo_energy_with_boundary('periodic', length, dim, chi, device, dtype)

def test_mpo_energy_with_boundary(boundary, length, dim, chi, device, dtype):
    print(f"\n边界条件: {boundary}")
    
    # 创建随机MPS
    psi = MPS(length=length, dim=dim, chi=chi, device=device, dtype=dtype, boundary=boundary)
    
    # 创建TFI模型MPO
    mpo_tfi = MPO(length=length, dim=dim, device=device, dtype=dtype, name='TFI')
    mpo_tfi.generate_TFI_MPO(J=1.0, h=1.0)
    
    print(f"\n1. TFI模型MPO能量计算:")
    try:
        energy_tfi, _ = psi.energy_with_MPO(mpo_tfi)
        print(f"   ✓ 成功计算TFI模型能量: {energy_tfi}")
    except Exception as e:
        print(f"   ✗ TFI模型能量计算失败: {e}")
    
    # 创建全单位算符MPO
    print(f"\n2. 全单位算符MPO能量计算:")
    mpo_identity = create_identity_mpo(length, dim, device, dtype)
    
    try:
        energy_identity, _ = psi.energy_with_MPO(mpo_identity)
        print(f"   ✓ 成功计算全单位算符能量: {energy_identity}")
        
        # 计算内积作为参考
        inner_product = psi.inner_product(psi, boundary=boundary)
        print(f"   ✓ 内积计算结果: {inner_product}")
        
        # 验证能量与内积是否一致
        error = abs(energy_identity - inner_product)
        print(f"   ✓ 能量与内积误差: {error}")
        
        if error < 1e-6:
            print(f"   ✓ 验证通过: 全单位算符能量与内积一致")
        else:
            print(f"   ✗ 验证失败: 全单位算符能量与内积不一致")
    except Exception as e:
        print(f"   ✗ 全单位算符能量计算失败: {e}")

def create_identity_mpo(length, dim, device, dtype):
    """创建全单位算符MPO"""
    mpo = MPO(length=length, dim=dim, device=device, dtype=dtype)
    
    # 生成单位算符
    I = tc.eye(dim, dtype=dtype, device=device)
    
    # 创建MPO张量列表
    tensors = []
    D = 2  # 单位算符MPO只需要D=2的虚拟维度
    
    for i in range(length):
        # 初始化MPO张量：形状为(D, D, d, d)
        mpo_tensor = tc.zeros((D, D, dim, dim), dtype=dtype, device=device)
        
        # 单位算符通道
        mpo_tensor[0, 0, :, :] = I
        mpo_tensor[1, 1, :, :] = I
        
        # 单位算符转移通道
        mpo_tensor[0, 1, :, :] = I
        
        # 边界条件处理
        if i == 0:
            # 左边界：只保留第一个左虚拟维度
            mpo_tensor = mpo_tensor[0:1, :, :, :]
        elif i == length - 1:
            # 右边界：只保留最后一个右虚拟维度
            mpo_tensor = mpo_tensor[:, 1:2, :, :]
        
        tensors.append(mpo_tensor)
    
    mpo.tensors = tensors
    return mpo

if __name__ == "__main__":
    test_mpo_energy()
