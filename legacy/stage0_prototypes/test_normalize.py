import torch as tc
import numpy as np
from AD_MPS_fixed import MPS, MPO

# 设置计算设备
device = tc.device('cuda' if tc.cuda.is_available() else 'cpu')

def test_normalize_returns_mps():
    """
    测试normalize方法返回MPS实例
    """
    print("测试1: normalize方法返回MPS实例")
    
    # 创建MPS
    N = 4
    d = 2
    chi = 5
    mps = MPS(length=N, dim=d, chi=chi, device=device, boundary='open')
    
    # 调用normalize方法，返回MPS实例
    normalized_mps = mps.normalize()
    
    # 检查返回类型
    assert isinstance(normalized_mps, MPS), f"normalize方法应返回MPS实例，实际返回: {type(normalized_mps)}"
    
    # 检查形状一致性
    assert normalized_mps.length == mps.length, f"长度不匹配: {normalized_mps.length} vs {mps.length}"
    assert normalized_mps.dim == mps.dim, f"物理维度不匹配: {normalized_mps.dim} vs {mps.dim}"
    assert normalized_mps.chi == mps.chi, f"键维度不匹配: {normalized_mps.chi} vs {mps.chi}"
    
    print("✓ normalize方法正确返回MPS实例")
    return normalized_mps

def test_normalize_preserves_graph():
    """
    测试normalize方法保持计算图
    """
    print("\n测试2: normalize方法保持计算图")
    
    # 创建MPS
    N = 2
    d = 2
    chi = 3
    mps = MPS(length=N, dim=d, chi=chi, device=device, boundary='open')
    
    # 调用normalize方法，获取张量列表
    normalized_tensors = mps.normalize(return_mps=False)
    
    # 检查第一个张量是否是计算图的一部分
    # 即是否是原始张量的视图
    first_tensor = normalized_tensors[0]
    
    # 计算简单的函数，检查是否可以自动微分
    # 这里我们简单地计算第一个张量的范数
    norm = tc.norm(first_tensor)
    
    # 尝试反向传播
    norm.backward()
    
    # 检查原始MPS的参数是否有梯度
    has_grad = any(p.grad is not None for p in mps.parameters())
    assert has_grad, "normalize方法未保持计算图，梯度不存在"
    
    print("✓ normalize方法正确保持计算图")

def test_normalize_accuracy():
    """
    测试normalize方法的准确性
    """
    print("\n测试3: normalize方法的准确性")
    
    # 创建MPS
    N = 4
    d = 2
    chi = 5
    mps = MPS(length=N, dim=d, chi=chi, device=device, boundary='open')
    
    # 归一化前内积
    overlap_before = mps.inner_product(mps)
    print(f"归一化前内积: {overlap_before.item()}")
    
    # 归一化
    normalized_mps = mps.normalize()
    
    # 归一化后内积
    overlap_after = normalized_mps.inner_product(normalized_mps)
    print(f"归一化后内积: {overlap_after.item()}")
    
    # 检查归一化效果
    assert abs(overlap_after.item() - 1.0) < 1e-6, f"归一化不准确: {overlap_after.item()}"
    
    print("✓ normalize方法准确归一化MPS")

def test_chain_calls():
    """
    测试链式调用
    """
    print("\n测试4: 链式调用支持")
    
    # 创建MPS和MPO
    N = 4
    d = 2
    chi = 5
    
    mps = MPS(length=N, dim=d, chi=chi, device=device, boundary='open')
    mpo = MPO(length=N, dim=d, device=device)
    mpo.generate_TFI_MPO(J=1.0, h=0.5)
    
    # 测试链式调用: normalize().inner_product()
    overlap = mps.normalize().inner_product(mps.normalize())
    print(f"链式调用内积: {overlap.item()}")
    
    # 测试链式调用: normalize().energy_with_MPO()
    energy, _ = mps.normalize().energy_with_MPO(mpo)
    print(f"链式调用能量: {energy.item()}")
    
    print("✓ 链式调用正常工作")

def test_normalize_without_mps():
    """
    测试return_mps=False选项
    """
    print("\n测试5: return_mps=False选项")
    
    # 创建MPS
    N = 4
    d = 2
    chi = 5
    mps = MPS(length=N, dim=d, chi=chi, device=device, boundary='open')
    
    # 调用normalize方法，返回张量列表
    normalized_tensors = mps.normalize(return_mps=False)
    
    # 检查返回类型
    assert isinstance(normalized_tensors, list), f"return_mps=False时应返回列表，实际返回: {type(normalized_tensors)}"
    assert len(normalized_tensors) == N, f"张量列表长度不匹配: {len(normalized_tensors)} vs {N}"
    
    print("✓ return_mps=False选项正常工作")

if __name__ == "__main__":
    print("开始测试normalize方法修改...")
    
    # 运行所有测试
    test_normalize_returns_mps()
    test_normalize_preserves_graph()
    test_normalize_accuracy()
    test_chain_calls()
    test_normalize_without_mps()
    
    print("\n所有测试通过！normalize方法修改成功！")
