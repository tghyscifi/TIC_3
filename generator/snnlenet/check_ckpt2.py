import torch

# 原始 LeNet checkpoint
ckpt1 = torch.load(r'E:\Tsinghua\tianji\tianjic_x1_software_0814\generator\LeNet\lenet_model\lenet_mnist_final_99.3.pth', map_location='cpu')
print('=== lenet_mnist_final_99.3.pth ===')
print('type:', type(ckpt1))
print('keys:', list(ckpt1.keys()) if isinstance(ckpt1, dict) else 'N/A')

if isinstance(ckpt1, dict) and 'state_dict' in ckpt1:
    sd = ckpt1['state_dict']
    for k, v in sd.items():
        print(f'  {k}: shape={v.shape}, dtype={v.dtype}')
elif isinstance(ckpt1, dict):
    for k, v in ckpt1.items():
        if hasattr(v, 'shape'):
            print(f'  {k}: shape={v.shape}, dtype={v.dtype}')
        else:
            print(f'  {k}: {type(v).__name__}')

print()
print('=== lenet_snn_sim.pth ===')
ckpt2 = torch.load(r'E:\Tsinghua\tianji\tianjic_x1_software_0814\generator\snnlenet\lenet_snn_sim.pth', map_location='cpu')
for k, v in ckpt2.items():
    if hasattr(v, 'shape'):
        print(f'  {k}: shape={v.shape}, dtype={v.dtype}')
