import torch as tc
import numpy as np
from AD_MPS_fixed import MPS

# 全面测试归一化方法，包括与inner_product方法的兼容性
def test_normalize_full():
    print("全面测试归一化方法...")
    
    # 测试参数
    lengths = [5, 10, 15]
    dim = 2
    chi = 4
    device = 'cpu'
    dtype = tc.complex128
    
    # 测试开边界条件
    print("\n=== 测试开边界条件 ===")
    for length in lengths:
        print(f"\n长度 {length}:")
        
        # 创建随机MPS
        psi = MPS(length=length, dim=dim, chi=chi, device=device, dtype=dtype, boundary='open')
        
        # 测试1：使用inner_product方法计算归一化前内积
        overlap_before = psi.inner_product(psi, boundary='open')
        print(f"  inner_product方法 - 归一化前: {overlap_before}")
        
        # 归一化MPS
        psi_normalized = psi.normalize()
        
        # 测试2：使用inner_product方法计算归一化后内积
        overlap_after = psi_normalized.inner_product(psi_normalized, boundary='open')
        print(f"  inner_product方法 - 归一化后: {overlap_after}")
        
        # 测试3：验证归一化前后的MPS是否接近
        overlap_psi_psi_normalized = psi.inner_product(psi_normalized, boundary='open')
        print(f"  归一化前后overlap: {overlap_psi_psi_normalized}")
        
        # 测试4：验证归一化后的MPS是否正交
        # 创建另一个随机MPS
        phi = MPS(length=length, dim=dim, chi=chi, device=device, dtype=dtype, boundary='open')
        phi_normalized = phi.normalize()
        overlap_psi_phi = psi_normalized.inner_product(phi_normalized, boundary='open')
        print(f"  与另一个归一化MPS的overlap: {overlap_psi_phi}")
    
    # 测试周期边界条件
    print("\n=== 测试周期边界条件 ===")
    for length in [5, 8]:
        print(f"\n长度 {length}:")
        
        # 创建随机MPS
        psi = MPS(length=length, dim=dim, chi=chi, device=device, dtype=dtype, boundary='periodic')
        
        # 测试1：使用inner_product方法计算归一化前内积
        overlap_before = psi.inner_product(psi, boundary='periodic', form='normal')
        print(f"  inner_product方法 - 归一化前: {overlap_before}")
        
        # 归一化MPS
        psi_normalized = psi.normalize()
        
        # 测试2：使用inner_product方法计算归一化后内积
        overlap_after = psi_normalized.inner_product(psi_normalized, boundary='periodic', form='normal')
        print(f"  inner_product方法 - 归一化后: {overlap_after}")
        
        # 测试3：使用log形式的inner_product方法
        overlap_after_log = psi_normalized.inner_product(psi_normalized, boundary='periodic', form='log')
        print(f"  inner_product方法（log形式）- 归一化后: {overlap_after_log}")
    
    # 测试梯度兼容性
    print("\n=== 测试梯度兼容性 ===")
    length = 5
    print(f"\n长度 {length}:")
    
    # 创建随机MPS
    psi = MPS(length=length, dim=dim, chi=chi, device=device, dtype=dtype, boundary='open')
    
    # 记录原始张量
    original_tensors = [t.clone() for t in psi.tensors]
    
    # 归一化MPS
    psi_normalized = psi.normalize()
    
    # 计算能量（模拟DMRG过程）
    energy = psi_normalized.inner_product(psi_normalized, boundary='open')
    
    # 尝试反向传播
    try:
        energy.backward()
        print("  ✓ 反向传播成功")
        
        # 检查是否有梯度
        has_gradients = any(t.grad is not None for t in psi.tensors)
        if has_gradients:
            print("  ✓ 张量有梯度")
        else:
            print("  ✗ 张量没有梯度")
    except Exception as e:
        print(f"  ✗ 反向传播失败: {e}")

if __name__ == "__main__":
    test_normalize_full()
