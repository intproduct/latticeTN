import torch as tc
import numpy as np
from AD_MPS_fixed import MPS, MPO
from AD_MERA import MERA
from AD_PEPS import PEPS

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

def test_mera():
    """测试MERA实现"""
    print("\n测试MERA实现...")
    
    N = 8  # 必须是2的幂次
    d = 2
    chi = 5
    
    # 创建MERA
    mera = MERA(N, d, chi)
    
    print(f"MERA层数: {mera.levels}")
    print(f"MERA disentanglers数量: {len(mera.disentanglers)}")
    print(f"MERA isometries数量: {len(mera.isometries)}")
    print(f"MERA总参数数量: {len(list(mera.parameters()))}")
    
    return mera

def test_peps():
    """测试PEPS实现"""
    print("\n测试PEPS实现...")
    
    Lx = 2
    Ly = 2
    d = 2
    chi = 3
    
    # 创建PEPS
    peps = PEPS(Lx, Ly, d, chi)
    
    print(f"PEPS晶格大小: {Lx}x{Ly}")
    print(f"PEPS张量数量: {len(peps.tensor_info)}")
    print(f"PEPS总参数数量: {len(list(peps.parameters()))}")
    
    return peps

def test_mpo():
    """测试MPO实现"""
    print("\n测试MPO实现...")
    
    N = 8
    d = 2
    
    # 创建MPO
    mpo = MPO(length=N, dim=d)
    
    # 测试自旋算符生成
    ops = mpo.generate_spin_operators()
    print(f"生成的自旋算符: {list(ops.keys())}")
    
    # 测试横场Ising模型MPO生成
    mpo.generate_TFI_MPO(J=1.0, h=0.5)
    print(f"TFI MPO张量数量: {len(mpo.tensors)}")
    print(f"第一个MPO张量形状: {mpo.tensors[0].shape}")
    
    return mpo

if __name__ == "__main__":
    print("开始测试自动微分张量网络实现...")
    
    # 测试MPO
    mpo = test_mpo()
    
    # 测试MPS
    mps, mpo = test_mps()
    
    # 测试MERA
    mera = test_mera()
    
    # 测试PEPS
    peps = test_peps()
    
    print("\n所有核心功能测试完成！")
