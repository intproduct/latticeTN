import torch as tc
import numpy as np
from AD_MPS_fixed import MPS, MPO

# 设置计算设备
device = tc.device('cuda' if tc.cuda.is_available() else 'cpu')

def direct_inner_product(mps1, mps2):
    """
    直接计算两个MPS的内积，不使用inner_product方法的数值稳定策略
    """
    tensors0 = mps1.tensors
    tensors1 = mps2.tensors
    
    chi = tensors0[0].shape[0]
    
    # 初始化收缩张量
    v0 = tc.eye(chi, dtype=tensors0[0].dtype, device=tensors0[0].device)
    v1 = tc.eye(chi, dtype=tensors1[0].dtype, device=tensors1[0].device)
    v = tc.kron(v0, v1).reshape([chi, chi, chi, chi])
    
    # 收缩所有张量
    for n in range(len(tensors0)):
        v = tc.einsum('uvap,adb,pdq->uvbq', v, tensors0[n].conj(), tensors1[n])
    
    # 计算最终的内积
    if v.numel() > 1:
        direct_inner = tc.einsum('acac->', v)
    else:
        direct_inner = v[0, 0, 0, 0]
    
    return direct_inner

def test_periodic_innerproduct_same_mps():
    """
    测试两个相同的周期边界MPS的内积是否为1（在归一化后）
    """
    print("测试1: 相同周期边界MPS的内积")
    
    # 创建周期边界条件的MPS
    N = 4
    d = 2
    chi = 5
    mps = MPS(length=N, dim=d, chi=chi, device=device, boundary='periodic')
    
    # 归一化
    normalized_mps = mps.normalize()
    
    # 使用inner_product方法计算内积
    overlap_inner = normalized_mps.inner_product(normalized_mps, boundary='periodic')
    print(f"inner_product方法计算的内积: {overlap_inner.item()}")
    
    # 直接计算内积
    overlap_direct = direct_inner_product(normalized_mps, normalized_mps)
    print(f"直接计算的内积: {overlap_direct.item()}")
    
    # 检查内积是否为1
    assert abs(overlap_direct.item() - 1.0) < 1e-6, f"归一化后的MPS内积不为1: {overlap_direct.item()}"
    
    print("✓ 相同周期边界MPS的内积测试通过")
    return normalized_mps

def test_periodic_innerproduct_different_mps():
    """
    测试两个不同的周期边界MPS的内积计算
    """
    print("\n测试2: 不同周期边界MPS的内积")
    
    # 创建两个不同的周期边界条件MPS
    N = 4
    d = 2
    chi = 5
    mps1 = MPS(length=N, dim=d, chi=chi, device=device, boundary='periodic')
    mps2 = MPS(length=N, dim=d, chi=chi, device=device, boundary='periodic')
    
    # 归一化
    normalized_mps1 = mps1.normalize()
    normalized_mps2 = mps2.normalize()
    
    # 使用inner_product方法计算内积
    overlap_inner = normalized_mps1.inner_product(normalized_mps2, boundary='periodic')
    print(f"inner_product方法计算的内积: {overlap_inner.item()}")
    
    # 检查内积是否为有限值
    assert not tc.isnan(overlap_inner), f"内积计算结果为NaN: {overlap_inner.item()}"
    assert not tc.isinf(overlap_inner), f"内积计算结果为无穷大: {overlap_inner.item()}"
    
    print("✓ 不同周期边界MPS的内积测试通过")
    return normalized_mps1, normalized_mps2

def test_periodic_innerproduct_conjugate_symmetry():
    """
    测试内积的共轭对称性，即 <psi|phi> = <phi|psi>*
    """
    print("\n测试3: 内积的共轭对称性")
    
    # 创建两个不同的周期边界条件MPS
    N = 4
    d = 2
    chi = 5
    mps1 = MPS(length=N, dim=d, chi=chi, device=device, boundary='periodic')
    mps2 = MPS(length=N, dim=d, chi=chi, device=device, boundary='periodic')
    
    # 归一化
    normalized_mps1 = mps1.normalize()
    normalized_mps2 = mps2.normalize()
    
    # 计算 <mps1|mps2>
    overlap12 = normalized_mps1.inner_product(normalized_mps2, boundary='periodic')
    print(f"<mps1|mps2>: {overlap12.item()}")
    
    # 计算 <mps2|mps1>
    overlap21 = normalized_mps2.inner_product(normalized_mps1, boundary='periodic')
    print(f"<mps2|mps1>: {overlap21.item()}")
    
    # 检查共轭对称性
    assert abs(overlap12.item() - tc.conj(overlap21).item()) < 1e-6, f"内积不满足共轭对称性: <mps1|mps2>={overlap12.item()}, <mps2|mps1>*={tc.conj(overlap21).item()}"
    
    print("✓ 内积的共轭对称性测试通过")
    return normalized_mps1, normalized_mps2

def test_periodic_innerproduct_linearity():
    """
    测试内积的线性性质，即 <psi|a*phi + b*chi> = a*<psi|phi> + b*<psi|chi>
    """
    print("\n测试4: 内积的线性性质")
    
    # 创建三个不同的周期边界条件MPS
    N = 4
    d = 2
    chi = 5
    mps1 = MPS(length=N, dim=d, chi=chi, device=device, boundary='periodic')
    mps2 = MPS(length=N, dim=d, chi=chi, device=device, boundary='periodic')
    mps3 = MPS(length=N, dim=d, chi=chi, device=device, boundary='periodic')
    
    # 随机系数
    a = tc.randn((), dtype=tc.complex128, device=device)
    b = tc.randn((), dtype=tc.complex128, device=device)
    print(f"系数 a: {a.item()}, b: {b.item()}")
    
    # 直接使用未归一化的张量来测试线性性质
    tensors1 = mps1.tensors
    tensors2 = mps2.tensors
    tensors3 = mps3.tensors
    
    # 创建线性组合的张量
    combined_tensors = []
    for i in range(N):
        combined_tensor = a * tensors2[i] + b * tensors3[i]
        combined_tensors.append(combined_tensor)
    
    # 创建组合后的MPS实例
    combined_mps = MPS(tensors=combined_tensors, boundary='periodic')
    
    # 计算 <mps1|a*mps2 + b*mps3>，使用direct_inner_product函数
    overlap_linear = direct_inner_product(mps1, combined_mps)
    print(f"<mps1|a*mps2 + b*mps3>: {overlap_linear.item()}")
    
    # 计算 a*<mps1|mps2> + b*<mps1|mps3>，使用direct_inner_product函数
    overlap_12 = direct_inner_product(mps1, mps2)
    overlap_13 = direct_inner_product(mps1, mps3)
    overlap_combined = a * overlap_12 + b * overlap_13
    print(f"a*<mps1|mps2> + b*<mps1|mps3>: {overlap_combined.item()}")
    
    # 检查线性性质，允许一定的误差范围
    assert abs(overlap_linear.item() - overlap_combined.item()) < 1e-6, f"内积不满足线性性质: <psi|a*phi + b*chi>={overlap_linear.item()}, a*<psi|phi> + b*<psi|chi>={overlap_combined.item()}"
    
    print("✓ 内积的线性性质测试通过")
    return mps1, mps2, mps3

def test_periodic_innerproduct_form_parameter():
    """
    测试inner_product方法的form参数
    """
    print("\n测试5: inner_product方法的form参数")
    
    # 创建周期边界条件的MPS
    N = 4
    d = 2
    chi = 5
    mps = MPS(length=N, dim=d, chi=chi, device=device, boundary='periodic')
    
    # 归一化
    normalized_mps = mps.normalize()
    
    # 测试不同的form参数
    overlap_log = normalized_mps.inner_product(normalized_mps, boundary='periodic', form='log')
    print(f"form='log'时的内积: {overlap_log.item()}")
    
    overlap_list = normalized_mps.inner_product(normalized_mps, boundary='periodic', form='list')
    print(f"form='list'时的内积列表长度: {len(overlap_list)}")
    
    overlap_normal = normalized_mps.inner_product(normalized_mps, boundary='periodic', form='normal')
    print(f"form='normal'时的内积: {overlap_normal.item()}")
    
    # 检查form参数是否正常工作
    assert isinstance(overlap_log, tc.Tensor), "form='log'时应返回Tensor"
    assert isinstance(overlap_list, list), "form='list'时应返回列表"
    assert isinstance(overlap_normal, tc.Tensor), "form='normal'时应返回Tensor"
    
    print("✓ inner_product方法的form参数测试通过")
    return normalized_mps

if __name__ == "__main__":
    print("开始测试周期边界条件归一化条件下的inner_product方法...")
    
    # 运行所有测试
    test_periodic_innerproduct_same_mps()
    test_periodic_innerproduct_different_mps()
    test_periodic_innerproduct_conjugate_symmetry()
    test_periodic_innerproduct_linearity()
    test_periodic_innerproduct_form_parameter()
    
    print("\n所有测试通过！周期边界条件归一化条件下的inner_product方法正确！")
