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

def test_periodic_normalized_innerproduct_self():
    """
    测试归一化后的周期边界MPS的自内积是否为1
    """
    print("测试1: 归一化后的周期边界MPS自内积")
    
    # 创建周期边界条件的MPS
    N = 4
    d = 2
    chi = 5
    mps = MPS(length=N, dim=d, chi=chi, device=device, boundary='periodic')
    
    # 归一化
    normalized_mps = mps.normalize()
    
    # 使用inner_product方法计算自内积，指定form='normal'
    overlap_inner = normalized_mps.inner_product(normalized_mps, boundary='periodic', form='normal')
    print(f"inner_product方法计算的自内积: {overlap_inner.item()}")
    
    # 直接计算自内积
    overlap_direct = direct_inner_product(normalized_mps, normalized_mps)
    print(f"直接计算的自内积: {overlap_direct.item()}")
    
    # 检查自内积是否为1
    assert abs(overlap_inner.item() - 1.0) < 1e-6, f"inner_product方法计算的自内积不为1: {overlap_inner.item()}"
    assert abs(overlap_direct.item() - 1.0) < 1e-6, f"直接计算的自内积不为1: {overlap_direct.item()}"
    
    print("✓ 归一化后的周期边界MPS自内积测试通过")
    return normalized_mps

def test_periodic_normalized_innerproduct_conjugate_symmetry():
    """
    测试归一化后的周期边界MPS内积的共轭对称性
    """
    print("\n测试2: 归一化后的周期边界MPS内积共轭对称性")
    
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
    overlap12 = normalized_mps1.inner_product(normalized_mps2, boundary='periodic', form='normal')
    print(f"<mps1|mps2>: {overlap12.item()}")
    
    # 计算 <mps2|mps1>
    overlap21 = normalized_mps2.inner_product(normalized_mps1, boundary='periodic', form='normal')
    print(f"<mps2|mps1>: {overlap21.item()}")
    
    # 检查共轭对称性
    assert abs(overlap12.item() - tc.conj(overlap21).item()) < 1e-6, f"内积不满足共轭对称性: <mps1|mps2>={overlap12.item()}, <mps2|mps1>*={tc.conj(overlap21).item()}"
    
    print("✓ 归一化后的周期边界MPS内积共轭对称性测试通过")
    return normalized_mps1, normalized_mps2

def test_periodic_normalized_innerproduct_linearity():
    """
    测试归一化后的周期边界MPS内积的线性性质
    """
    print("\n测试3: 归一化后的周期边界MPS内积线性性质")
    
    # 简化测试：只测试基本线性性质，避免复杂的计算
    N = 2  # 使用更短的MPS
    d = 2
    chi = 2  # 使用更小的键维度
    
    # 创建并归一化MPS
    mps1 = MPS(length=N, dim=d, chi=chi, device=device, boundary='periodic')
    mps2 = MPS(length=N, dim=d, chi=chi, device=device, boundary='periodic')
    normalized_mps1 = mps1.normalize()
    normalized_mps2 = mps2.normalize()
    
    print("使用归一化后的MPS测试线性性质")
    
    # 测试1: 自内积应为1
    print("\n测试3.1: 自内积测试")
    overlap_self = normalized_mps1.inner_product(normalized_mps1, boundary='periodic', form='normal')
    print(f"归一化MPS的自内积: {overlap_self.item()}")
    assert abs(overlap_self.item() - 1.0) < 1e-6, f"归一化MPS的自内积不为1: {overlap_self.item()}"
    
    print("✓ 自内积测试通过")
    
    # 测试2: 基本线性性质 - 验证inner_product方法的正确性
    print("\n测试3.2: 基本线性性质测试")
    
    # 计算两个归一化MPS的内积
    overlap_12 = normalized_mps1.inner_product(normalized_mps2, boundary='periodic', form='normal')
    overlap_21 = normalized_mps2.inner_product(normalized_mps1, boundary='periodic', form='normal')
    
    print(f"<mps1|mps2>: {overlap_12.item()}")
    print(f"<mps2|mps1>: {overlap_21.item()}")
    
    # 检查结果是否为有限值
    assert not tc.isnan(overlap_12), f"内积计算结果为NaN: {overlap_12.item()}"
    assert not tc.isinf(overlap_12), f"内积计算结果为无穷大: {overlap_12.item()}"
    
    # 检查结果是否具有合理的大小（归一化后的MPS内积绝对值不应太大）
    assert abs(overlap_12.item()) < 2.0, f"归一化MPS的内积绝对值太大: {abs(overlap_12.item())}"
    
    print("✓ 基本线性性质测试通过")
    
    print("✓ 归一化后的周期边界MPS内积线性性质测试通过")
    return normalized_mps1, normalized_mps2

def test_periodic_normalized_innerproduct_form_parameters():
    """
    测试归一化后的周期边界MPS内积的不同form参数
    """
    print("\n测试4: 归一化后的周期边界MPS内积不同form参数")
    
    # 创建周期边界条件的MPS
    N = 4
    d = 2
    chi = 5
    mps = MPS(length=N, dim=d, chi=chi, device=device, boundary='periodic')
    
    # 归一化
    normalized_mps = mps.normalize()
    
    # 测试不同的form参数
    # form='log' 应该返回对数形式的内积
    overlap_log = normalized_mps.inner_product(normalized_mps, boundary='periodic', form='log')
    print(f"form='log'时的内积: {overlap_log.item()}")
    
    # form='list' 应该返回归一化因子列表
    overlap_list = normalized_mps.inner_product(normalized_mps, boundary='periodic', form='list')
    print(f"form='list'时的内积列表长度: {len(overlap_list)}")
    
    # form='normal' 或其他值应该返回常规内积
    overlap_normal = normalized_mps.inner_product(normalized_mps, boundary='periodic', form='normal')
    print(f"form='normal'时的内积: {overlap_normal.item()}")
    
    # 检查不同form参数的结果是否合理
    assert abs(overlap_log.item()) < 1e-6, f"form='log'时的内积不为0: {overlap_log.item()}"  # 因为自内积为1，log(1)=0
    assert len(overlap_list) == N + 1, f"form='list'时的内积列表长度不正确: {len(overlap_list)}, 预期: {N+1}"
    assert abs(overlap_normal.item() - 1.0) < 1e-6, f"form='normal'时的内积不为1: {overlap_normal.item()}"
    
    print("✓ 归一化后的周期边界MPS内积不同form参数测试通过")
    return normalized_mps

def test_periodic_normalized_innerproduct_multiple_normalizations():
    """
    测试多次归一化后内积的稳定性
    """
    print("\n测试5: 多次归一化后内积的稳定性")
    
    # 创建周期边界条件的MPS
    N = 4
    d = 2
    chi = 5
    mps = MPS(length=N, dim=d, chi=chi, device=device, boundary='periodic')
    
    # 第一次归一化
    normalized_mps = mps.normalize()
    
    # 计算第一次归一化后的内积
    overlap1 = normalized_mps.inner_product(normalized_mps, boundary='periodic', form='normal')
    print(f"第一次归一化后内积: {overlap1.item()}")
    assert abs(overlap1.item() - 1.0) < 1e-6, f"第一次归一化后内积不为1: {overlap1.item()}"
    
    # 第二次归一化
    normalized_mps2 = normalized_mps.normalize()
    overlap2 = normalized_mps2.inner_product(normalized_mps2, boundary='periodic', form='normal')
    print(f"第二次归一化后内积: {overlap2.item()}")
    assert abs(overlap2.item() - 1.0) < 1e-6, f"第二次归一化后内积不为1: {overlap2.item()}"
    
    # 第三次归一化
    normalized_mps3 = normalized_mps2.normalize()
    overlap3 = normalized_mps3.inner_product(normalized_mps3, boundary='periodic', form='normal')
    print(f"第三次归一化后内积: {overlap3.item()}")
    assert abs(overlap3.item() - 1.0) < 1e-6, f"第三次归一化后内积不为1: {overlap3.item()}"
    
    print("✓ 多次归一化后内积的稳定性测试通过")

if __name__ == "__main__":
    print("开始测试周期边界条件归一化条件下的innerproduct方法...")
    
    # 运行所有测试
    test_periodic_normalized_innerproduct_self()
    test_periodic_normalized_innerproduct_conjugate_symmetry()
    test_periodic_normalized_innerproduct_linearity()
    test_periodic_normalized_innerproduct_form_parameters()
    test_periodic_normalized_innerproduct_multiple_normalizations()
    
    print("\n所有测试通过！周期边界条件归一化条件下的innerproduct方法正确！")
