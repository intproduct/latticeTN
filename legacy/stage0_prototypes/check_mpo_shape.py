from AD_MPS_fixed import MPO

# 创建MPO实例
mpo = MPO(length=4, dim=2)
mpo.generate_TFI_MPO(J=1.0, h=1.0)

# 打印MPO张量形状
print('MPO tensors shapes:')
for i, tensor in enumerate(mpo.tensors):
    print(f'Site {i}: {tensor.shape}')
