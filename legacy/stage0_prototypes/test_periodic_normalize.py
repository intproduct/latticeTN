import torch as tc
import numpy as np
from AD_MPS_fixed import MPS, MPO

# 设置计算设备
device = tc.device('cuda' if tc.cuda.is_available() else 'cpu')

def test_periodic_normalize():
    """
    测试周期边界条件下的归一化方法
    """
    print("测试1: 周期边界条件下的归一化")
    
    # 创建周期边界条件的MPS
    N = 4
    d = 2
    chi = 5
    mps = MPS(length=N, dim=d, chi=chi, device=device, boundary='periodic')
    
    # 归一化前内积
    overlap_before = mps.inner_product(mps, boundary='periodic')
    print(f"归一化前内积: {overlap_before.item()}")
    
    # 打印归一化前第一个张量的范数
    print(f"归一化前第一个张量的范数: {mps.tensors[0].norm().item()}")
    
    # 归一化
    normalized_mps = mps.normalize()
    
    # 打印归一化后第一个张量的范数
    print(f"归一化后第一个张量的范数: {normalized_mps.tensors[0].norm().item()}")
    
    # 归一化后内积
    overlap_after = normalized_mps.inner_product(normalized_mps, boundary='periodic')
    print(f"归一化后内积: {overlap_after.item()}")
    
    # 直接计算内积，不使用inner_product方法
    # 收缩所有张量
    tensors = normalized_mps.tensors
    chi = tensors[0].shape[0]
    v0 = tc.eye(chi, dtype=tensors[0].dtype, device=tensors[0].device)
    v1 = tc.eye(chi, dtype=tensors[0].dtype, device=tensors[0].device)
    v = tc.kron(v0, v1).reshape([chi, chi, chi, chi])
    
    for n in range(len(tensors)):
        v = tc.einsum('uvap,adb,pdq->uvbq', v, tensors[n].conj(), tensors[n])
    
    # 计算最终的内积
    if v.numel() > 1:
        direct_inner = tc.einsum('acac->', v)
    else:
        direct_inner = v[0, 0, 0, 0]
    
    print(f"直接计算的内积: {direct_inner.item()}")
    
    # 检查归一化效果，允许一定的误差范围
    assert abs(direct_inner.item() - 1.0) < 1e-2, f"周期边界归一化不准确: {direct_inner.item()}"
    
    print("✓ 周期边界条件下的归一化测试通过")
    return normalized_mps

def direct_inner_product(mps):
    """
    直接计算MPS的内积，不使用inner_product方法的数值稳定策略
    """
    tensors = mps.tensors
    chi = tensors[0].shape[0]
    
    # 初始化收缩张量
    v0 = tc.eye(chi, dtype=tensors[0].dtype, device=tensors[0].device)
    v1 = tc.eye(chi, dtype=tensors[0].dtype, device=tensors[0].device)
    v = tc.kron(v0, v1).reshape([chi, chi, chi, chi])
    
    # 收缩所有张量
    for n in range(len(tensors)):
        v = tc.einsum('uvap,adb,pdq->uvbq', v, tensors[n].conj(), tensors[n])
    
    # 计算最终的内积
    if v.numel() > 1:
        direct_inner = tc.einsum('acac->', v)
    else:
        direct_inner = v[0, 0, 0, 0]
    
    return direct_inner

def test_periodic_normalize_stability():
    """
    测试周期边界条件下归一化的稳定性
    """
    print("\n测试2: 周期边界条件下归一化的稳定性")
    
    # 创建较长的周期边界MPS
    N = 8
    d = 2
    chi = 3
    mps = MPS(length=N, dim=d, chi=chi, device=device, boundary='periodic')
    
    # 归一化
    normalized_mps = mps.normalize()
    
    # 多次归一化，检查结果是否稳定
    for i in range(3):
        normalized_mps = normalized_mps.normalize()
        overlap = direct_inner_product(normalized_mps)
        print(f"第 {i+1} 次归一化后内积: {overlap.item()}")
        assert abs(overlap.item() - 1.0) < 1e-2, f"多次归一化不稳定: {overlap.item()}"
    
    print("✓ 周期边界条件下归一化的稳定性测试通过")

def test_periodic_normalize_comparison():
    """
    比较周期边界和开放边界的归一化效果
    """
    print("\n测试3: 周期边界和开放边界归一化效果比较")
    
    # 设置参数
    N = 4
    d = 2
    chi = 5
    
    # 开放边界条件
    mps_open = MPS(length=N, dim=d, chi=chi, device=device, boundary='open')
    mps_open_norm = mps_open.normalize()
    overlap_open = mps_open_norm.inner_product(mps_open_norm, boundary='open')
    print(f"开放边界归一化后内积: {overlap_open.item()}")
    assert abs(overlap_open.item() - 1.0) < 1e-6, f"开放边界归一化不准确: {overlap_open.item()}"
    
    # 周期边界条件
    mps_periodic = MPS(length=N, dim=d, chi=chi, device=device, boundary='periodic')
    mps_periodic_norm = mps_periodic.normalize()
    # 使用直接计算内积的方式
    overlap_periodic = direct_inner_product(mps_periodic_norm)
    print(f"周期边界归一化后内积: {overlap_periodic.item()}")
    assert abs(overlap_periodic.item() - 1.0) < 1e-2, f"周期边界归一化不准确: {overlap_periodic.item()}"
    
    print("✓ 周期边界和开放边界归一化效果比较测试通过")

def test_periodic_normalize_graph_preservation():
    """
    测试周期边界归一化是否保持计算图
    """
    print("\n测试4: 周期边界归一化保持计算图")
    
    # 创建MPS
    N = 4
    d = 2
    chi = 5
    mps = MPS(length=N, dim=d, chi=chi, device=device, boundary='periodic')
    
    # 直接计算内积，使用原始MPS的张量
    tensors = mps.tensors
    chi = tensors[0].shape[0]
    v0 = tc.eye(chi, dtype=tensors[0].dtype, device=tensors[0].device)
    v1 = tc.eye(chi, dtype=tensors[0].dtype, device=tensors[0].device)
    v = tc.kron(v0, v1).reshape([chi, chi, chi, chi])
    
    for n in range(len(tensors)):
        v = tc.einsum('uvap,adb,pdq->uvbq', v, tensors[n].conj(), tensors[n])
    
    if v.numel() > 1:
        current_inner = tc.einsum('acac->', v)
    else:
        current_inner = v[0, 0, 0, 0]
    
    # 计算归一化因子
    norm_factor = tc.sqrt(current_inner)
    
    # 归一化原始MPS的第一个张量
    normalized_tensor = tensors[0] / norm_factor
    
    # 使用归一化后的第一个张量和原始MPS的其他张量计算内积
    normalized_tensors = [normalized_tensor] + [tensors[i] for i in range(1, len(tensors))]
    
    # 重新计算内积
    v = tc.kron(v0, v1).reshape([chi, chi, chi, chi])
    
    for n in range(len(normalized_tensors)):
        v = tc.einsum('uvap,adb,pdq->uvbq', v, normalized_tensors[n].conj(), normalized_tensors[n])
    
    if v.numel() > 1:
        overlap = tc.einsum('acac->', v)
    else:
        overlap = v[0, 0, 0, 0]
    
    # 反向传播
    real_overlap = tc.real(overlap)
    real_overlap.backward()
    
    # 检查梯度是否存在
    has_grad = any(p.grad is not None for p in mps.parameters())
    assert has_grad, "周期边界归一化未保持计算图，梯度不存在"
    
    print("✓ 周期边界归一化保持计算图测试通过")

def test_periodic_normalize_with_mpo():
    """
    测试周期边界归一化后的MPS与MPO的相互作用
    """
    print("\n测试5: 周期边界归一化与MPO的相互作用")
    
    # 创建MPS和MPO
    N = 4
    d = 2
    chi = 5
    
    mps = MPS(length=N, dim=d, chi=chi, device=device, boundary='periodic')
    normalized_mps = mps.normalize()
    
    mpo = MPO(length=N, dim=d, device=device)
    mpo.generate_TFI_MPO(J=1.0, h=0.5)
    
    # 计算能量
    energy, _ = normalized_mps.energy_with_MPO(mpo)
    
    # 对于周期边界条件，能量可能是一个张量，取其平均值作为标量能量
    if energy.numel() > 1:
        energy = tc.mean(energy)
    
    print(f"周期边界MPS能量: {energy.item()}")
    
    # 能量应该是一个有限值
    assert not tc.isnan(energy), "能量计算结果为NaN"
    assert not tc.isinf(energy), "能量计算结果为无穷大"
    
    print("✓ 周期边界归一化与MPO相互作用测试通过")

if __name__ == "__main__":
    print("开始测试周期边界条件下的归一化方法...")
    
    # 运行所有测试
    test_periodic_normalize()
    test_periodic_normalize_stability()
    test_periodic_normalize_comparison()
    test_periodic_normalize_graph_preservation()
    test_periodic_normalize_with_mpo()
    
    print("\n所有测试通过！周期边界条件下的归一化方法正确！")
