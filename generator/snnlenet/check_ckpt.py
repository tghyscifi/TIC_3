import torch
ckpt = torch.load(r'E:\Tsinghua\tianji\tianjic_x1_software_0814\generator\snnlenet\lenet_snn_sim.pth', map_location='cpu')
print('type:', type(ckpt))
if isinstance(ckpt, dict):
    for k, v in ckpt.items():
        if hasattr(v, 'shape'):
            print(f'  {k}: shape={v.shape}, dtype={v.dtype}')
        else:
            print(f'  {k}: {type(v).__name__} = {v}')
print()
print('Total keys:', len(ckpt))
# Check if it has state_dict wrapper
if 'state_dict' in ckpt:
    print('Has state_dict wrapper!')
    sd = ckpt['state_dict']
    for k, v in sd.items():
        print(f'  state_dict.{k}: shape={v.shape}, dtype={v.dtype}')
