import torch as tc
import numpy as np
from AD_MPS_fixed import MPS, MPO

def test_mps():
    """测试修复后的MPS实现"""

    N = 8
    d = 2
    chi = 10
    
    # 创建MPS
    mps = MPS(length=N, dim=d, chi=chi, boundary='open')

    #测试归一化输出
    norm_mps = mps.normalize()
    
    

if __name__ == "__main__":
    test_mps()