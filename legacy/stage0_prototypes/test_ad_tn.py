import torch as tc
import numpy as np
from AD_MPS_fixed import MPS, MPO, AD_DMRG
from AD_MERA import AD_MERA
from AD_PEPS import AD_PEPS

def test_mps():
    """测试修复后的MPS实现"""
    print("测试MPS实现...")
    
    # 创建MPS
    N = 8
    d = 2
    chi = 10
    mps = MPS(length=N, dim=d, chi=chi, boundary='open')
    
    # 创建MPO (横场Ising模型)
    mpo = MPO(length=N, dim=d)
    mpo.generate_TFI_MPO(J=1.0, h=0.5)
    
    # 测试内积
    overlap = mps.inner_product(mps)
    print(f"MPS内积: {overlap.item()}")
    
    # 测试能量计算
    energy, _ = mps.energy_with_MPO(mpo)
    print(f"初始能量: {energy.item()}")
    
    return mps, mpo

def test_dmrg():
    """测试AD_DMRG实现"""
    print("\n测试AD_DMRG实现...")
    
    N = 8
    d = 2
    chi = 10
    
    # 创建MPO
    mpo = MPO(length=N, dim=d)
    mpo.generate_TFI_MPO(J=1.0, h=0.5)
    
    # 运行AD_DMRG
    mps = AD_DMRG(None, mpo, N, d, chi)
    
    # 测试归一化
    mps_normalized = mps.to_normalize()
    overlap = mps_normalized.inner_product(mps_normalized)
    print(f"归一化后内积: {overlap.item()}")
    
    return mps

def test_mera():
    """测试AD_MERA实现"""
    print("\n测试AD_MERA实现...")
    
    N = 8  # 必须是2的幂次
    d = 2
    chi = 5
    
    # 运行AD_MERA
    mera = AD_MERA(N, d, chi)
    
    print(f"MERA层数: {mera.levels}")
    print(f"MERA张量数量: {len(mera.tensors)}")
    
    return mera

def test_peps():
    """测试AD_PEPS实现"""
    print("\n测试AD_PEPS实现...")
    
    Lx = 2
    Ly = 2
    d = 2
    chi = 3
    
    # 运行AD_PEPS
    peps = AD_PEPS(Lx, Ly, d, chi)
    
    print(f"PEPS晶格大小: {Lx}x{Ly}")
    print(f"PEPS张量数量: {len(peps.tensors)}")
    
    return peps

if __name__ == "__main__":
    print("开始测试自动微分张量网络实现...")
    
    # 测试MPS
    mps, mpo = test_mps()
    
    # 测试DMRG
    optimized_mps = test_dmrg()
    
    # 测试MERA
    mera = test_mera()
    
    # 测试PEPS
    peps = test_peps()
    
    print("\n所有测试完成！")
