import torch as tc
import numpy as np
from AD_MPS_fixed import MPO

# 设置计算设备
device = tc.device('cuda' if tc.cuda.is_available() else 'cpu')

def test_mpo_generation():
    """
    测试MPO生成函数是否正确实现了横场Ising模型
    """
    # 设置参数
    N = 3
    d = 2
    J = 1.0
    h = 0.5
    
    # 创建MPO
    mpo = MPO(length=N, dim=d, device=device)
    mpo.generate_TFI_MPO(J=J, h=h)
    
    print("MPO生成测试：")
    print(f"系统大小：{N}")
    print(f"物理维度：{d}")
    print(f"MPO张量数量：{len(mpo.tensors)}")
    
    # 打印每个MPO张量的形状
    for i, tensor in enumerate(mpo.tensors):
        print(f"MPO张量 {i} 形状：{tensor.shape}")
    
    # 检查MPO的物理维度是否正确
    assert mpo.dim == d, f"MPO物理维度错误：{mpo.dim}，预期：{d}"
    assert mpo.length == N, f"MPO长度错误：{mpo.length}，预期：{N}"
    
    print("\nMPO生成测试通过！")

def test_mpo_contract():
    """
    测试MPO与简单MPS的收缩是否合理
    """
    # 设置参数
    N = 2
    d = 2
    J = 1.0
    h = 0.5
    
    # 创建MPO
    mpo = MPO(length=N, dim=d, device=device)
    mpo.generate_TFI_MPO(J=J, h=h)
    
    print("\nMPO收缩测试：")
    
    # 创建一个简单的MPS（全1张量）
    from AD_MPS_fixed import MPS
    
    # 创建全1 MPS
    tensors = []
    for i in range(N):
        if i == 0:
            # 左边边界
            tensor = tc.ones((1, d, 2), device=device, dtype=tc.complex128, requires_grad=True)
        elif i == N-1:
            # 右边边界
            tensor = tc.ones((2, d, 1), device=device, dtype=tc.complex128, requires_grad=True)
        else:
            # 中间张量
            tensor = tc.ones((2, d, 2), device=device, dtype=tc.complex128, requires_grad=True)
        tensors.append(tensor)
    
    mps = MPS(tensors=tensors, boundary='open')
    
    # 计算能量
    energy, _ = mps.energy_with_MPO(mpo)
    print(f"简单MPS与MPO的能量：{energy.item()}")
    
    # 检查能量是否为实数
    print(f"能量是否为实数：{tc.isreal(energy).item()}")
    
    print("\nMPO收缩测试完成！")

if __name__ == "__main__":
    test_mpo_generation()
    test_mpo_contract()
