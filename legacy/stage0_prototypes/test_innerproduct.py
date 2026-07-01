import torch as tc
import numpy as np
from AD_MPS_fixed import MPS, MPO

# 设置计算设备
device = tc.device('cuda' if tc.cuda.is_available() else 'cpu')

def test_innerproduct_self():
    """
    测试相同MPS之间的内积
    归一化后的MPS与自身的内积应接近1
    """
    print("测试1: 相同MPS之间的内积")
    
    # 创建MPS并归一化
    N = 4
    d = 2
    chi = 5
    mps = MPS(length=N, dim=d, chi=chi, device=device, boundary='open')
    normalized_mps = mps.normalize()
    
    # 计算内积
    overlap = normalized_mps.inner_product(normalized_mps)
    
    print(f"归一化后MPS与自身内积: {overlap.item()}")
    
    # 检查结果是否接近1
    assert abs(overlap.item() - 1.0) < 1e-6, f"内积应为1，实际为: {overlap.item()}"
    
    print("✓ 相同MPS之间的内积测试通过")

def test_innerproduct_conjugate_symmetric():
    """
    测试内积的共轭对称性
    <ψ|φ> = <φ|ψ>*
    """
    print("\n测试2: 内积的共轭对称性")
    
    # 创建两个不同的MPS
    N = 4
    d = 2
    chi = 5
    mps1 = MPS(length=N, dim=d, chi=chi, device=device, boundary='open')
    mps2 = MPS(length=N, dim=d, chi=chi, device=device, boundary='open')
    
    # 归一化
    mps1_norm = mps1.normalize()
    mps2_norm = mps2.normalize()
    
    # 计算两个方向的内积
    overlap12 = mps1_norm.inner_product(mps2_norm)
    overlap21 = mps2_norm.inner_product(mps1_norm)
    
    print(f"<mps1|mps2>: {overlap12.item()}")
    print(f"<mps2|mps1>: {overlap21.item()}")
    print(f"共轭对称性: {overlap12.item()} ≈ {overlap21.conj().item()}")
    
    # 检查共轭对称性
    assert abs(overlap12.item() - overlap21.conj().item()) < 1e-12, f"内积不满足共轭对称性: {overlap12.item()} != {overlap21.conj().item()}"
    
    print("✓ 内积共轭对称性测试通过")

def test_innerproduct_boundary_conditions():
    """
    测试不同边界条件下的内积计算
    """
    print("\n测试3: 不同边界条件下的内积计算")
    
    # 创建MPS
    N = 4
    d = 2
    chi = 5
    
    # 开放边界条件
    mps_open = MPS(length=N, dim=d, chi=chi, device=device, boundary='open')
    mps_open_norm = mps_open.normalize()
    
    # 计算开放边界条件下的内积
    overlap_open = mps_open_norm.inner_product(mps_open_norm, boundary='open')
    print(f"开放边界条件下的内积: {overlap_open.item()}")
    assert abs(overlap_open.item() - 1.0) < 1e-6, f"开放边界内积应为1，实际为: {overlap_open.item()}"
    
    # 周期边界条件 - 不使用normalize方法，直接测试内积计算
    mps_periodic = MPS(length=N, dim=d, chi=chi, device=device, boundary='periodic')
    
    # 计算周期边界条件下的内积
    overlap_periodic = mps_periodic.inner_product(mps_periodic, boundary='periodic')
    print(f"周期边界条件下的内积: {overlap_periodic.item()}")
    
    # 对于未归一化的MPS，内积应该是一个有限值，而不是无穷大或NaN
    assert not tc.isnan(overlap_periodic), "周期边界内积为NaN"
    assert not tc.isinf(overlap_periodic), "周期边界内积为无穷大"
    
    print("✓ 不同边界条件下的内积测试通过")

def test_innerproduct_direct_comparison():
    """
    与直接收缩状态向量的结果进行比较
    """
    print("\n测试4: 与直接收缩状态向量的结果比较")
    
    # 使用较短的MPS以减少计算量
    N = 3
    d = 2
    chi = 4
    
    # 创建MPS
    mps = MPS(length=N, dim=d, chi=chi, device=device, boundary='open')
    normalized_mps = mps.normalize()
    
    # 使用inner_product方法计算内积
    overlap_mps = normalized_mps.inner_product(normalized_mps)
    
    # 直接收缩状态向量计算内积
    psi = normalized_mps.contract_mps()
    psi_vec = psi.reshape(-1, 1)
    overlap_direct = tc.matmul(psi_vec.conj().T, psi_vec)[0, 0]
    
    print(f"MPS内积方法结果: {overlap_mps.item()}")
    print(f"直接收缩结果: {overlap_direct.item()}")
    
    # 比较结果
    assert abs(overlap_mps.item() - overlap_direct.item()) < 1e-6, f"结果不一致: MPS方法={overlap_mps.item()}, 直接收缩={overlap_direct.item()}"
    
    print("✓ 与直接收缩结果比较测试通过")

def test_innerproduct_long_chain():
    """
    测试长链MPS的内积计算（数值稳定性测试）
    """
    print("\n测试5: 长链MPS的内积计算")
    
    # 创建较长的MPS
    N = 10
    d = 2
    chi = 3
    
    mps = MPS(length=N, dim=d, chi=chi, device=device, boundary='open')
    normalized_mps = mps.normalize()
    
    # 计算内积
    overlap = normalized_mps.inner_product(normalized_mps)
    
    print(f"长链MPS内积结果: {overlap.item()}")
    
    # 检查结果是否接近1
    assert abs(overlap.item() - 1.0) < 1e-4, f"长链内积应为1，实际为: {overlap.item()}"
    
    print("✓ 长链MPS内积测试通过")

def test_innerproduct_with_mpo():
    """
    测试与MPO相关的内积计算
    """
    print("\n测试6: 与MPO相关的内积计算")
    
    # 创建MPS和MPO
    N = 4
    d = 2
    chi = 5
    
    mps = MPS(length=N, dim=d, chi=chi, device=device, boundary='open')
    normalized_mps = mps.normalize()
    
    mpo = MPO(length=N, dim=d, device=device)
    mpo.generate_TFI_MPO(J=1.0, h=0.5)
    
    # 计算能量，内部会调用内积
    energy, _ = normalized_mps.energy_with_MPO(mpo)
    
    print(f"使用MPO的能量计算结果: {energy.item()}")
    
    # 能量应该是一个有限值
    assert not tc.isnan(energy), "能量计算结果为NaN"
    assert not tc.isinf(energy), "能量计算结果为无穷大"
    
    print("✓ 与MPO相关的内积测试通过")

if __name__ == "__main__":
    print("开始测试inner_product方法...")
    
    # 运行所有测试
    test_innerproduct_self()
    test_innerproduct_conjugate_symmetric()
    test_innerproduct_boundary_conditions()
    test_innerproduct_direct_comparison()
    test_innerproduct_long_chain()
    test_innerproduct_with_mpo()
    
    print("\n所有测试通过！inner_product方法正确！")
