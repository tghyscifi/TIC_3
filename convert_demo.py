import sys, os
# ensure workspace root on path
sys.path.append(r'C:\Users\14394\Desktop\tianjic_x1_software_0814')

from onnx2map import _convert_pth_to_onnx, generate_map_config_from_onnx

# convert pth -> onnx
pth = r'onnx2prim\lenet.pth'
onnx_tmp = r'onnx2prim\lenet_temp.onnx'
_convert_pth_to_onnx(pth, 'generator.LeNet.lenet_model.lenet.LeNet', (1,1,28,28), onnx_tmp)
print('converted', onnx_tmp)

# debugging: inspect parse_onnx results
from onnx2prim.onnx_parser import parse_onnx
seq = parse_onnx(onnx_tmp)
print('input_shape', seq.input_shape)
print('layers count', len(seq.layers))
for idx,(layer, has_relu) in enumerate(seq.layers):
    print(idx, type(layer).__name__, getattr(layer,'cin',None), getattr(layer,'cout',None), getattr(layer,'px',None), getattr(layer,'py',None))

# generate map_config
mcfg = generate_map_config_from_onnx(onnx_tmp, g2_cores=8)
outdir = r'onnx2prim\lenet_test'
os.makedirs(outdir, exist_ok=True)
import json
with open(os.path.join(outdir, 'map_config.json'), 'w', encoding='utf-8') as f:
    json.dump(mcfg, f, indent=2)
print('map_config generated in', outdir)

print('directory contents:')
print(os.listdir(outdir))