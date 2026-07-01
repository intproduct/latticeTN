import torch as tc
from AD_MPS_fixed import MPS, MPO

def test_energy_with_MPO():
    """
    测试energy_with_MPO方法是否正确实现
    1. 当MPO是单位算符时，结果应该与inner_product一致
    2. 验证开边界和周期边界条件
    """
    
    # 测试参数
    length = 2  # 使用更短的长度，更容易验证
    dim = 2
    chi = 2
    
    # 设备设置
    device = 'cpu'
    dtype = tc.complex128
    
    print("=== 测试 energy_with_MPO 方法 ===")
    
    # 测试开边界条件
    print("\n1. 测试开边界条件：")
    mps_open = MPS(length=length, dim=dim, chi=chi, device=device, dtype=dtype, boundary='open')
    
    # 创建单位算符MPO
    mpo_open = MPO(length=length, dim=dim, device=device, dtype=dtype)
    
    # 生成正确的单位算符MPO张量
    # 单位算符MPO的每个张量都是单位算符，MPO键维度为1
    # 形状：
    # - 左边界: [1×1×dim×dim]
    # - 中间: [1×1×dim×dim]
    # - 右边界: [1×1×dim×dim]
    mpo_open.tensors = []
    I = tc.eye(dim, dtype=dtype, device=device)
    
    # 左边界MPO张量
    mpo_left = I.unsqueeze(0).unsqueeze(0)  # [1×1×dim×dim]
    mpo_open.tensors.append(mpo_left)
    
    # 中间MPO张量（如果有的话）
    for i in range(1, length-1):
        mpo_middle = I.unsqueeze(0).unsqueeze(0)  # [1×1×dim×dim]
        mpo_open.tensors.append(mpo_middle)
    
    # 右边界MPO张量
    mpo_right = I.unsqueeze(0).unsqueeze(0)  # [1×1×dim×dim]
    mpo_open.tensors.append(mpo_right)
    
    # 使用energy_with_MPO计算能量
    energy_mpo, _ = mps_open.energy_with_MPO(mpo_open)
    
    # 使用inner_product计算内积
    energy_inner = mps_open.inner_product(mps_open, boundary='open')
    
    print(f"   energy_with_MPO: {energy_mpo}")
    print(f"   inner_product: {energy_inner}")
    print(f"   相对误差: {abs((energy_mpo - energy_inner).item() / energy_inner.item()):.6e}")
    
    # 归一化MPS，确保结果更容易比较
    print("\n2. 测试归一化后开边界条件：")
    mps_open_normalized = mps_open.normalize()
    energy_mpo_normalized, _ = mps_open_normalized.energy_with_MPO(mpo_open)
    energy_inner_normalized = mps_open_normalized.inner_product(mps_open_normalized, boundary='open')
    
    print(f"   energy_with_MPO: {energy_mpo_normalized}")
    print(f"   inner_product: {energy_inner_normalized}")
    print(f"   相对误差: {abs((energy_mpo_normalized - energy_inner_normalized).item() / energy_inner_normalized.item()):.6e}")
    
    print("\n=== 测试完成 ===")

if __name__ == "__main__":
    test_energy_with_MPO()