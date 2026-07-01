import torch as tc
import numpy as np
from AD_MPS_fixed import MPS, MPO, DMRGnet

# 测试归一化方法与DMRG的兼容性
def test_dmrg_compatibility():
    print("测试归一化方法与DMRG的兼容性...")
    
    # 测试参数
    length = 5  # 较短的长度，快速测试
    dim = 2     # 自旋-1/2系统
    chi = 4     # 键维度
    device = 'cpu'
    dtype = tc.complex128
    
    print(f"\n创建长度 {length} 的MPS和TFI模型MPO...")
    
    # 创建MPS
    mps = MPS(length=length, dim=dim, chi=chi, device=device, dtype=dtype, boundary='open')
    
    # 创建TFI模型MPO
    mpo = MPO(length=length, dim=dim, device=device, dtype=dtype, name='TFI')
    mpo.generate_TFI_MPO(J=1.0, h=1.0)
    
    print("\n初始化DMRGnet...")
    dmrg_net = DMRGnet(mps, mpo)
    
    print("\n前向传播测试...")
    try:
        energy = dmrg_net.forward()
        print(f"  ✓ 前向传播成功，初始能量: {energy}")
    except Exception as e:
        print(f"  ✗ 前向传播失败: {e}")
        return
    
    print("\n优化步骤测试（前10步）...")
    try:
        # 使用Adam优化器，只运行10步
        optimizer = tc.optim.Adam(mps.parameters(), lr=1e-3)
        
        for step in range(10):
            optimizer.zero_grad()
            energy = dmrg_net.forward()
            energy.backward()
            optimizer.step()
            
            # 测试归一化
            if step % 5 == 0:
                with tc.no_grad():
                    normalized_tensors = mps.normalize(return_mps=False)
                    for i in range(mps.length):
                        mps.tensors[i].data = normalized_tensors[i]
        
        print(f"  ✓ 10步优化成功，最终能量: {energy}")
    except Exception as e:
        print(f"  ✗ 优化失败: {e}")
        return
    
    print("\n=== 兼容性测试通过！ ===")

if __name__ == "__main__":
    test_dmrg_compatibility()
