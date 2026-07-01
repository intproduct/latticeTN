from AD_MPS_fixed import MPS

# 创建MPS实例
mps = MPS(length=4, dim=2, chi=2, boundary='open')

# 打印MPS张量形状
print('MPS tensors shapes:')
for i, tensor in enumerate(mps.tensors):
    print(f'Site {i}: {tensor.shape}')
