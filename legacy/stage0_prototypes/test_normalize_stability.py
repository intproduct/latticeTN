import torch as tc
import numpy as np
from AD_MPS_fixed import MPS

# 测试不同长度MPS的归一化稳定性
def test_normalize_stability():
    print("测试归一化方法的数值稳定性...")
    
    # 测试参数
    lengths = [5, 10, 15, 20]  # 不同长度的MPS
    dim = 2  # 自旋-1/2系统
    chi = 4  # 键维度
    device = 'cpu'  # 简化测试，使用CPU
    dtype = tc.complex128
    
    # 直接计算内积的函数，用于验证归一化结果
    def direct_open_boundary_innerproduct(psi):
        """直接计算开边界MPS的内积，用于验证归一化"""
        tensors = psi.tensors
        v = tc.ones((1, 1), dtype=psi.dtype, device=psi.device)
        
        for i in range(len(tensors)):
            v = tc.einsum('ab,acd,bce->de', v, tensors[i].conj(), tensors[i])
        
        return v.squeeze()
    
    # 测试开边界条件
    print("\n=== 测试开边界条件 ===")
    for length in lengths:
        print(f"\n长度 {length}:")
        
        # 创建随机MPS
        psi = MPS(length=length, dim=dim, chi=chi, device=device, dtype=dtype, boundary='open')
        
        # 归一化前的内积
        overlap_before = direct_open_boundary_innerproduct(psi)
        print(f"  归一化前内积: {overlap_before}")
        
        # 归一化MPS
        psi_normalized = psi.normalize()
        
        # 归一化后的内积
        overlap_after = direct_open_boundary_innerproduct(psi_normalized)
        print(f"  归一化后内积: {overlap_after}")
        
        # 检查是否接近1
        error = abs(overlap_after - 1.0)
        print(f"  归一化误差: {error}")
        
        if error < 1e-6:
            print("  ✓ 归一化成功")
        else:
            print("  ✗ 归一化失败")
    
    # 测试周期边界条件
    print("\n=== 测试周期边界条件 ===")
    # 注意：周期边界条件的直接内积计算更复杂，这里简化测试
    for length in [5, 10]:  # 限制周期边界测试长度，避免计算复杂度过高
        print(f"\n长度 {length}:")
        
        # 创建随机MPS
        psi = MPS(length=length, dim=dim, chi=chi, device=device, dtype=dtype, boundary='periodic')
        
        # 使用inner_product方法计算内积
        overlap_before = psi.inner_product(psi, boundary='periodic', form='normal')
        print(f"  归一化前内积: {overlap_before}")
        
        # 归一化MPS
        psi_normalized = psi.normalize()
        
        # 归一化后的内积
        overlap_after = psi_normalized.inner_product(psi_normalized, boundary='periodic', form='normal')
        print(f"  归一化后内积: {overlap_after}")
        
        # 检查是否接近1
        error = abs(overlap_after - 1.0)
        print(f"  归一化误差: {error}")
        
        if error < 1e-6:
            print("  ✓ 归一化成功")
        else:
            print("  ✗ 归一化失败")
    
    # 测试数值不稳定情况
    print("\n=== 测试数值不稳定情况 ===")
    length = 10
    print(f"长度 {length}:")
    
    # 创建一个数值不稳定的MPS（故意缩放张量）
    psi = MPS(length=length, dim=dim, chi=chi, device=device, dtype=dtype, boundary='open')
    
    # 故意缩放张量，制造数值不稳定情况
    scaled_tensors = []
    for i, tensor in enumerate(psi.tensors):
        # 交替缩放张量，创造极大和极小的值
        scale = 1e3 if i % 2 == 0 else 1e-3
        scaled_tensors.append(tensor * scale)
    
    # 更新MPS张量
    psi_scaled = MPS(scaled_tensors, boundary='open')
    
    # 直接计算内积会出现数值问题
    try:
        overlap_before = direct_open_boundary_innerproduct(psi_scaled)
        print(f"  缩放后直接内积: {overlap_before}")
    except Exception as e:
        print(f"  缩放后直接内积计算失败: {e}")
    
    # 使用改进的归一化方法
    try:
        psi_normalized = psi_scaled.normalize()
        overlap_after = direct_open_boundary_innerproduct(psi_normalized)
        print(f"  归一化后内积: {overlap_after}")
        
        error = abs(overlap_after - 1.0)
        print(f"  归一化误差: {error}")
        
        if error < 1e-6:
            print("  ✓ 归一化成功")
        else:
            print("  ✗ 归一化失败")
    except Exception as e:
        print(f"  归一化失败: {e}")

if __name__ == "__main__":
    test_normalize_stability()
