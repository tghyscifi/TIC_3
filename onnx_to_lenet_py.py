# -*- coding: utf-8 -*-
"""
ONNX / .pth -> lenet.py 风格单文件转换器

支持两种输入：
  1) ONNX 模型（如 lenet.onnx）
  2) PyTorch 权重 .pth（需配合 --model-class 与 --input-shape，会先导出为 ONNX 再生成）

生成与 lenet.py 结构一致的单文件 Python 模块，包含：量化/截断、模型类、
QuantizationConfig、DataHandler、generate_data、gen_0/gen_1/gen_2 映射配置（支持 Tianjic 16×10 核）、check 与 main。

Tianjic X1 单芯片为 16×10 核，通过 size_x、size_y 配置；存储与原语参见文档 TianjicX1.docx。

用法:
    # 从 ONNX 生成
    python onnx_to_lenet_py.py lenet.onnx -o generated_lenet.py
    # 从 .pth 生成（会先导出 ONNX 到输出目录）
    python onnx_to_lenet_py.py lenet.pth -o generated_lenet.py --model-class generator.LeNet.lenet_model.lenet.LeNet --input-shape 1,1,28,28
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

# 使用项目内 onnx2prim
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from onnx2prim.onnx_parser import (
    ConvLayer,
    FCLayer,
    LayerSeq,
    MaxPoolLayer,
    parse_onnx,
)
from onnx2prim.address_allocator import AddrAssign, allocate_addresses
from onnx2prim.codegen import (
    build_cut_config,
    generate_g0,
    generate_g1_1core,
    generate_g2,
    _align_ceil as _cin_align_inner,
)


def _align_ceil(c: int, align: int = 32) -> int:
    return (c + align - 1) // align * align


def _convert_pth_to_onnx(
    pth_path: str,
    model_class: str,
    input_shape: Tuple[int, ...],
    onnx_path: str,
) -> None:
    """从 PyTorch .pth 加载模型并导出为 ONNX。需要安装 torch。"""
    try:
        import importlib
        import torch
    except ImportError as e:
        raise ImportError("从 .pth 转换需要安装 PyTorch: pip install torch") from e

    module_name, class_name = model_class.rsplit(".", 1)
    mod = importlib.import_module(module_name)
    ModelCls = getattr(mod, class_name)
    model = ModelCls()
    ckpt = torch.load(pth_path, map_location="cpu")
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]
    model.load_state_dict(ckpt, strict=False)
    model.eval()
    dummy = torch.randn(tuple(input_shape))
    torch.onnx.export(
        model,
        dummy,
        onnx_path,
        opset_version=11,
        input_names=["input"],
        output_names=["output"],
    )


def _build_weight_mapping(seq: LayerSeq, graph_initializers: List[str]) -> List[Tuple[str, Optional[str], Optional[str]]]:
    """构建 (module_key, onnx_weight_name, onnx_bias_name) 列表，用于从 ONNX 加载权重。"""
    mapping = []
    conv_idx = fc_idx = 0
    for layer, _ in seq.layers:
        if isinstance(layer, ConvLayer):
            conv_idx += 1
            key = f"conv{conv_idx}"
            mapping.append((key, getattr(layer, "weight_name", None), getattr(layer, "bias_name", None)))
        elif isinstance(layer, FCLayer):
            fc_idx += 1
            key = f"fc{fc_idx}"
            mapping.append((key, getattr(layer, "weight_name", None), getattr(layer, "bias_name", None)))
    return mapping


def _gen_model_class(seq: LayerSeq, class_name: str = "Model") -> str:
    """根据 LayerSeq 生成 PyTorch 模型类源码。"""
    lines = [
        f"class {class_name}(torch.nn.Module):",
        "    def __init__(self, batch_norm=True, clamp=False, aware_test=None, cut_en=False, aware=False):",
        "        super().__init__()",
        "        self.batch_norm = batch_norm",
        "        self.clamp = clamp",
        "        self.cut_en = cut_en",
        "        self.aware = aware",
        "        self.aware_test = aware_test",
        "",
    ]
    conv_idx = fc_idx = 0
    prev_c = seq.input_shape[0]
    prev_h, prev_w = seq.input_shape[1], seq.input_shape[2]
    for layer, has_relu in seq.layers:
        if isinstance(layer, ConvLayer):
            conv_idx += 1
            pad = (layer.pad_top, layer.pad_down, layer.pad_left, layer.pad_right)
            has_pad = any(p != 0 for p in pad)
            lines.append(f"        self.conv{conv_idx} = nn.Conv2d({layer.cin}, {layer.cout}, kernel_size=({layer.kx}, {layer.ky}), stride=({layer.sx}, {layer.sy}), padding=({layer.pad_top + layer.pad_down or 0}, {layer.pad_left + layer.pad_right or 0}), bias=not batch_norm)")
            if has_pad and (layer.pad_top or layer.pad_down or layer.pad_left or layer.pad_right):
                # padding 在 Conv2d 里用 padding=(ph, pw)，这里用 (pad_top+pad_down, pad_left+pad_right) 简化
                pass  # 已写在上面 padding=
            lines.append(f"        if self.batch_norm:")
            lines.append(f"            self.bn{conv_idx} = nn.BatchNorm2d({layer.cout})")
            lines.append(f"        self.cut{conv_idx} = Cut(en=True)")
            lines.append(f"        self.relu{conv_idx} = nn.ReLU(inplace=False)")
            idx_cur = seq.layers.index((layer, has_relu))
            next_layer = seq.layers[idx_cur + 1][0] if idx_cur + 1 < len(seq.layers) else None
            if isinstance(next_layer, MaxPoolLayer):
                lines.append(f"        self.max_pool{conv_idx} = nn.MaxPool2d(kernel_size=({next_layer.kx}, {next_layer.ky}), stride=({next_layer.sx}, {next_layer.sy}))")
                prev_h, prev_w = next_layer.px, next_layer.py
            else:
                lines.append(f"        self.max_pool{conv_idx} = nn.Identity()  # no pool in graph")
                prev_h, prev_w = layer.px_out, layer.py_out
            prev_c = layer.cout
        elif isinstance(layer, MaxPoolLayer):
            # 紧跟 Conv 的 MaxPool 已在上面与 Conv 一起建层，跳过
            pass
        elif isinstance(layer, FCLayer):
            fc_idx += 1
            cin_fc = layer.cin
            cout_fc = layer.cout
            num_fc = sum(1 for l, _ in seq.layers if isinstance(l, FCLayer))
            lines.append(f"        self.fc{fc_idx} = nn.Linear({cin_fc}, {cout_fc}, bias=not batch_norm)")
            if fc_idx < num_fc:  # 与 lenet.py 一致：最后一层 FC 不加 BatchNorm
                lines.append(f"        if self.batch_norm:")
                lines.append(f"            self.fc_bn{fc_idx} = nn.BatchNorm1d({cout_fc})")
            lines.append(f"        self.fc_cut{fc_idx} = Cut(en=True)")
            if fc_idx < sum(1 for l, _ in seq.layers if isinstance(l, FCLayer)):
                lines.append(f"        self.fc_relu{fc_idx} = nn.ReLU(inplace=False)")
            prev_c = cout_fc
            prev_h = prev_w = 1

    # forward
    lines.append("")
    lines.append("    def forward(self, x):")
    lines.append("        if self.clamp:")
    lines.append("            x = x.clamp(-1, 1)")
    lines.append("        if self.aware_test is not None:")
    lines.append("            x = FakeQuantizeWithoutLimit.apply(x, self.aware_test)")
    lines.append("        if self.aware:")
    lines.append("            x = FakeQuantize.apply(x, -1, 1, 128)")
    lines.append("        if self.cut_en:")
    lines.append("            x = x.mul(128).round().clamp(-128, 127)")
    lines.append("")
    conv_idx = fc_idx = 0
    for layer, has_relu in seq.layers:
        if isinstance(layer, ConvLayer):
            conv_idx += 1
            lines.append(f"        x = self.conv{conv_idx}(x)")
            lines.append(f"        if self.batch_norm:")
            lines.append(f"            x = self.bn{conv_idx}(x)")
            lines.append(f"        if self.cut_en:")
            lines.append(f"            x = self.cut{conv_idx}(x)")
            lines.append(f"        if self.clamp:")
            lines.append(f"            x = x.clamp(-1, 1)")
            lines.append(f"        if self.aware_test is not None:")
            lines.append(f"            x = FakeQuantizeWithoutLimit.apply(x, self.aware_test)")
            lines.append(f"        if self.aware:")
            lines.append(f"            x = FakeQuantize.apply(x, -1, 1, 128)")
            lines.append(f"        x = self.relu{conv_idx}(x)")
            lines.append(f"        x = self.max_pool{conv_idx}(x)")
            lines.append("")
        elif isinstance(layer, MaxPoolLayer):
            pass
        elif isinstance(layer, FCLayer):
            fc_idx += 1
            num_fc = sum(1 for l, _ in seq.layers if isinstance(l, FCLayer))
            if fc_idx == 1:
                lines.append("        x = x.permute(0, 2, 3, 1)")
                lines.append("        x = x.reshape(x.shape[0], -1)")
                lines.append("")
            lines.append(f"        x = self.fc{fc_idx}(x)")
            if fc_idx < num_fc:
                lines.append(f"        if self.batch_norm:")
                lines.append(f"            x = self.fc_bn{fc_idx}(x)")
            lines.append(f"        if self.cut_en:")
            lines.append(f"            x = self.fc_cut{fc_idx}(x)")
            lines.append(f"        if self.clamp:")
            lines.append(f"            x = x.clamp(-1, 1)")
            lines.append(f"        if self.aware_test is not None:")
            lines.append(f"            x = FakeQuantizeWithoutLimit.apply(x, self.aware_test)")
            lines.append(f"        if self.aware:")
            lines.append(f"            x = FakeQuantize.apply(x, -1, 1, 128)")
            if fc_idx < sum(1 for l, _ in seq.layers if isinstance(l, FCLayer)):
                lines.append(f"        x = self.fc_relu{fc_idx}(x)")
            lines.append("")
    lines.append("        return x")
    return "\n".join(lines)


def _gen_quantization_config(seq: LayerSeq) -> str:
    """生成 QuantizationConfig 类源码。"""
    weight_params = {}
    cut_params = {}
    conv_idx = fc_idx = 0
    for layer, _ in seq.layers:
        if isinstance(layer, ConvLayer):
            conv_idx += 1
            n = max(256, layer.cout * layer.cin * layer.kx * layer.ky)
            weight_params[f"conv{conv_idx}"] = min(4096, (1 << (int(math.log(n, 4)) * 2)) or 256)
            cut_params[f"cut{conv_idx}"] = f"int(math.log(self.__weight_parameters['conv{conv_idx}'], 4))"
        elif isinstance(layer, FCLayer):
            fc_idx += 1
            n = max(256, layer.cout * layer.cin)
            weight_params[f"fc{fc_idx}"] = min(4096, (1 << (int(math.log(n, 4)) * 2)) or 256)
            if fc_idx == sum(1 for l, _ in seq.layers if isinstance(l, FCLayer)):
                cut_params[f"fc_cut{fc_idx}"] = f"int(math.log(self.__weight_parameters['fc{fc_idx}'] * 4, 4))"
            else:
                cut_params[f"fc_cut{fc_idx}"] = f"int(math.log(self.__weight_parameters['fc{fc_idx}'], 4))"
    wp_str = repr(weight_params)
    cp_lines = ["        self.__cut_parameters = {"]
    for k, v in cut_params.items():
        cp_lines.append(f"            '{k}': {v},")
    cp_lines.append("        }")
    return f'''class QuantizationConfig:
    def __init__(self, name='in_cut_start'):
        self.__name = name
        self.__weight_parameters = {wp_str}
        {" ".join(cp_lines)}
    def __getitem__(self, item):
        if self.__name == 'in_cut_start':
            result = self.__cut_parameters.get(item)
        else:
            assert self.__name == 'weight'
            result = self.__weight_parameters.get(item)
        if result is None:
            warnings.warn('Module \\'{{}}\\' do not have in_cut_start parameter!'.format(item))
        return result
'''


def _gen_data_handler_and_generate_data(
    seq: LayerSeq,
    handler_class_name: str,
    model_class_name: str,
    onnx_default_basename: str,
    assignments: List[AddrAssign],
) -> Tuple[str, str]:
    """生成 DataHandler 子类与 generate_data 函数。onnx_default_basename 为与 .py 同目录的 .onnx 文件名，用于默认权重路径。"""
    # 收集各层 shape 用于 tensor_split
    conv_idx = fc_idx = 0
    layer_shapes = {}  # name -> (in_shape, out_shape) or weight/bias shape
    c, h, w = seq.input_shape[0], seq.input_shape[1], seq.input_shape[2]
    block_idx = 0
    for i, (layer, has_relu) in enumerate(seq.layers):
        if isinstance(layer, MaxPoolLayer) and i > 0 and isinstance(seq.layers[i - 1][0], ConvLayer):
            continue
        if isinstance(layer, ConvLayer):
            conv_idx += 1
            key = f"conv{conv_idx}"
            layer_shapes[key] = {
                "input": (c, h, w),
                "weight": (layer.cout, layer.cin, layer.ky, layer.kx),
                "bias": (layer.cout,),
                "cut_out": (layer.cout, layer.px_out, layer.py_out),
            }
            next_layer = seq.layers[i + 1][0] if i + 1 < len(seq.layers) else None
            if isinstance(next_layer, MaxPoolLayer):
                layer_shapes[f"max_pool{conv_idx}"] = {"output": (layer.cout, next_layer.px, next_layer.py)}
                h, w = next_layer.px, next_layer.py
            else:
                layer_shapes[f"max_pool{conv_idx}"] = {"output": (layer.cout, layer.px_out, layer.py_out)}
                h, w = layer.px_out, layer.py_out
            c = layer.cout
        elif isinstance(layer, FCLayer):
            fc_idx += 1
            key = f"fc{fc_idx}"
            layer_shapes[key] = {"weight": (layer.cout, layer.cin), "bias": (layer.cout,), "output": (layer.cout,)}
            if fc_idx == sum(1 for l, _ in seq.layers if isinstance(l, FCLayer)):
                layer_shapes[f"fc_cut{fc_idx}"] = {"output": (layer.cout,)}
            c = layer.cout
            h = w = 1

    # DataHandler 源码
    handler_src = f'''
class {handler_class_name}(DataHandler):
    def __init__(self, seed=7, onnx_path=None):
        super().__init__()
        torch.manual_seed(seed)
        c, h, w = {seq.input_shape[0]}, {seq.input_shape[1]}, {seq.input_shape[2]}
        x = torch.randn((1, c, h, w))
        onnx_file = onnx_path or os.path.join(os.path.dirname(os.path.abspath(__file__)), {repr(onnx_default_basename)})
        self.__model = {model_class_name}(batch_norm=False, clamp=False, aware_test=None, cut_en=True, aware=False)
        load_weights_from_onnx(self.__model, onnx_file)
        quantize = QuantizationConfig()
        for name, module in self.__model.named_modules():
            if isinstance(module, Cut):
                module.en = True
                try:
                    module.in_cut_start = quantize[name]
                except Exception:
                    pass
        DataHandler.check_model(self.__model)
        internal_data_input = []
        internal_data_output = []
        internal_name = []
        self.__internal = OrderedDict()
        def hook_fn(module, input, output):
            assert (type(input) is tuple and len(input) == 1)
            internal_data_output.append(output.clone().detach())
            internal_data_input.append(input[0].clone().detach())
        for name, module in self.__model.named_modules():
            if not isinstance(module, (nn.Conv2d, nn.MaxPool2d, nn.ReLU, nn.Linear, nn.AdaptiveAvgPool2d, nn.BatchNorm2d, nn.BatchNorm1d, Cut)):
                continue
            module.register_forward_hook(hook_fn)
            internal_name.append(name)
        y = self.__model(x)
        for name, data_in, data_out in zip(internal_name, internal_data_input, internal_data_output):
            self.__internal[name] = {{'input': np.array(data_in.squeeze(0), dtype=np.int32), 'output': np.array(data_out.squeeze(0), dtype=np.int32)}}
        for name, p in self.__model.named_parameters():
            layer_name, p_name = '.'.join(name.split('.')[:-1]), name.split('.')[-1]
            if self.__internal.get(layer_name) is not None:
                if p_name == 'bias':
                    self.__internal[layer_name][p_name] = np.array(p.clone().detach(), dtype=np.int32)
                elif p_name == 'weight':
                    self.__internal[layer_name][p_name] = np.array(p.clone().detach(), dtype=np.int8)
        self.__internal = {{k: v for k, v in self.__internal.items() if v}}
    @property
    def names(self):
        return self.__internal.keys()
    @property
    def parameters(self):
        return self.__internal
'''

    # generate_data 源码：根据 layer_shapes 生成 tensor_split 调用
    g_data_keys = []
    conv_idx = fc_idx = 0
    for i, (layer, _) in enumerate(seq.layers):
        if isinstance(layer, MaxPoolLayer) and i > 0 and isinstance(seq.layers[i - 1][0], ConvLayer):
            continue
        if isinstance(layer, ConvLayer):
            conv_idx += 1
            g_data_keys.append(("conv", conv_idx, layer, None))
        elif isinstance(layer, MaxPoolLayer):
            g_data_keys.append(("max_pool", conv_idx, layer, None))
        elif isinstance(layer, FCLayer):
            fc_idx += 1
            g_data_keys.append(("fc", fc_idx, layer, None))

    g_data_entries = ["    g_data = {}"]
    conv_idx = fc_idx = 0
    for i, (layer, _) in enumerate(seq.layers):
        if isinstance(layer, MaxPoolLayer) and i > 0 and isinstance(seq.layers[i - 1][0], ConvLayer):
            continue
        if isinstance(layer, ConvLayer):
            conv_idx += 1
            sh = layer_shapes.get(f"conv{conv_idx}", {})
            in_sh = sh.get("input", (1, 28, 28))
            w_sh = sh.get("weight", (6, 1, 5, 5))
            b_sh = sh.get("bias", (6,))
            cut_sh = sh.get("cut_out", (6, 24, 24))
            g_data_entries.append(f"    g_data['conv{conv_idx}'] = {{}}")
            if conv_idx == 1:
                g_data_entries.append(f"    g_data['conv{conv_idx}']['input'] = {handler_class_name}.tensor_split(")
                g_data_entries.append(f"        raw_data=handler.parameters['conv{conv_idx}']['input'].astype(np.int8),")
                g_data_entries.append(f"        split_dict={{(0, 0): ((0, {in_sh[0]}), (0, {in_sh[1]}), (0, {in_sh[2]}))}},")
                g_data_entries.append(f"        data_type=1, alignment=(None, None, 16), dims=(2, 1, 0))")
            g_data_entries.append(f"    g_data['conv{conv_idx}']['weight'] = {handler_class_name}.tensor_split(")
            g_data_entries.append(f"        raw_data=handler.parameters['conv{conv_idx}']['weight'].astype(np.int8),")
            g_data_entries.append(f"        split_dict={{(0, 0): ((0, {w_sh[0]}), (0, {w_sh[1]}), (0, {w_sh[2]}), (0, {w_sh[3]}))}},")
            g_data_entries.append(f"        data_type=1, alignment=(32, None, None, None), dims=(0, 3, 2, 1), is_weight=True)")
            g_data_entries.append(f"    g_data['conv{conv_idx}']['bias'] = {handler_class_name}.tensor_split(")
            g_data_entries.append(f"        raw_data=handler.parameters['conv{conv_idx}']['bias'].astype(np.int32),")
            g_data_entries.append(f"        split_dict={{(0, 0): ((0, {b_sh[0]}),)}}, data_type=0, alignment=(32,), dims=(0,))")
            g_data_entries.append(f"    g_data['cut{conv_idx}'] = {{}}")
            g_data_entries.append(f"    g_data['cut{conv_idx}']['output'] = {handler_class_name}.tensor_split(")
            g_data_entries.append(f"        raw_data=handler.parameters['cut{conv_idx}']['output'].astype(np.int8),")
            g_data_entries.append(f"        split_dict={{(0, 0): ((0, {cut_sh[0]}), (0, {cut_sh[1]}), (0, {cut_sh[2]}))}},")
            g_data_entries.append(f"        data_type=1, alignment=(16, None, None), dims=(0, 2, 1))")
            pool_sh = layer_shapes.get(f"max_pool{conv_idx}", {}).get("output", (6, 12, 12))
            g_data_entries.append(f"    g_data['max_pool{conv_idx}'] = {{}}")
            g_data_entries.append(f"    g_data['max_pool{conv_idx}']['output'] = {handler_class_name}.tensor_split(")
            g_data_entries.append(f"        raw_data=handler.parameters['max_pool{conv_idx}']['output'].astype(np.int8),")
            g_data_entries.append(f"        split_dict={{(0, 0): ((0, {pool_sh[0]}), (0, {pool_sh[1]}), (0, {pool_sh[2]}))}},")
            g_data_entries.append(f"        data_type=1, alignment=(16, None, None), dims=(0, 2, 1))")
        elif isinstance(layer, FCLayer):
            fc_idx += 1
            sh = layer_shapes.get(f"fc{fc_idx}", {})
            w_sh = sh.get("weight", (120, 400))
            b_sh = sh.get("bias", (120,))
            g_data_entries.append(f"    g_data['fc{fc_idx}'] = {{}}")
            g_data_entries.append(f"    g_data['fc{fc_idx}']['weight'] = {handler_class_name}.tensor_split(")
            g_data_entries.append(f"        raw_data=handler.parameters['fc{fc_idx}']['weight'].astype(np.int8),")
            g_data_entries.append(f"        split_dict={{(0, 0): ((0, {w_sh[0]}), (0, {w_sh[1]}))}},")
            g_data_entries.append(f"        data_type=1, alignment=(32, None), dims=(0, 1), is_weight=True)")
            g_data_entries.append(f"    g_data['fc{fc_idx}']['bias'] = {handler_class_name}.tensor_split(")
            g_data_entries.append(f"        raw_data=handler.parameters['fc{fc_idx}']['bias'].astype(np.int32),")
            g_data_entries.append(f"        split_dict={{(0, 0): ((0, {b_sh[0]}),)}}, data_type=0, alignment=(32,), dims=(0,))")
            if fc_idx == sum(1 for l, _ in seq.layers if isinstance(l, FCLayer)):
                out_sh = sh.get("output", (10,))
                g_data_entries.append(f"    g_data['fc_cut{fc_idx}'] = {{}}")
                g_data_entries.append(f"    g_data['fc_cut{fc_idx}']['output'] = {handler_class_name}.tensor_split(")
                g_data_entries.append(f"        raw_data=handler.parameters['fc_cut{fc_idx}']['output'].astype(np.int8),")
                g_data_entries.append(f"        split_dict={{(0, 0): ((0, {out_sh[0]}),)}}, data_type=1, alignment=(16,), dims=(0,))")
    g_data_entries.append("    return g_data")
    generate_data_src = "\n".join(g_data_entries)

    return handler_src, "def generate_data(handler):\n" + generate_data_src


def _gen_load_weights_from_onnx(mapping: List[Tuple[str, Optional[str], Optional[str]]]) -> str:
    """生成从 ONNX 加载权重到 state_dict 的函数。"""
    lines = [
        "def load_weights_from_onnx(model, onnx_path):",
        "    try:",
        "        import onnx",
        "        from onnx import numpy_helper",
        "    except ImportError:",
        "        raise ImportError('pip install onnx')",
        "    if not os.path.exists(onnx_path):",
        "        return",
        "    m = onnx.load(onnx_path)",
        "    init_names = {init.name: numpy_helper.to_array(init) for init in m.graph.initializer}",
        "    state = model.state_dict()",
        "    name_map = [  # (state_key_prefix, onnx_weight_name, onnx_bias_name)",
    ]
    for key, wname, bname in mapping:
        lines.append(f"        ({repr(key)}, {repr(wname)}, {repr(bname)}),")
    lines.extend([
        "    ]",
        "    for prefix, wname, bname in name_map:",
        "        if wname and wname in init_names:",
        "            w = torch.from_numpy(init_names[wname]).float()",
        "            if w.dim() == 4:",
        "                w = w.permute(0, 1, 2, 3)",
        "            key = prefix + '.weight'",
        "            if key in state and state[key].shape == w.shape:",
        "                state[key].copy_(w.mul(128).round().clamp(-128, 127).to(state[key].dtype))",
        "        if bname and bname in init_names:",
        "            b = torch.from_numpy(init_names[bname]).float()",
        "            key = prefix + '.bias'",
        "            if key in state and state[key].shape == b.shape:",
        "                state[key].copy_(b.mul(128*128).round().clamp(-0x80000000, 0x7fffffff).to(state[key].dtype))",
        "    return",
    ])
    return "\n".join(lines)


def generate_full_lenet_style_file(
    onnx_path: str,
    output_path: str,
    model_class_name: str = "Model",
    handler_class_name: str = "ModelDataHandler",
    case_file_name: Optional[str] = None,
    default_size_x: int = 1,
    default_size_y: int = 1,
    g2_size_x: int = 8,
    g2_size_y: int = 1,
) -> None:
    """
    从 ONNX 生成完整 lenet.py 风格单文件。

    支持 Tianjic 16×10 核：在 main 中通过 size_x/size_y 传入，默认 1×1 计算核、8×1 输出组。
    """
    seq = parse_onnx(onnx_path)
    cut_config = build_cut_config(seq)
    assignments, weight_end, final_out_addr = allocate_addresses(seq)
    mapping = _build_weight_mapping(seq, [])

    case_name = case_file_name or os.path.splitext(os.path.basename(onnx_path))[0]
    out_base = os.path.splitext(os.path.basename(output_path))[0]
    onnx_default_basename = out_base + ".onnx"

    model_src = _gen_model_class(seq, model_class_name)
    quant_src = _gen_quantization_config(seq)
    load_src = _gen_load_weights_from_onnx(mapping)
    handler_src, generate_data_src = _gen_data_handler_and_generate_data(
        seq, handler_class_name, model_class_name, onnx_default_basename, assignments
    )

    g0_code = generate_g0(
        os.path.splitext(os.path.basename(onnx_path))[0],
        seq.input_shape,
        seq.input_shape[0] * seq.input_shape[1] * seq.input_shape[2],
    )
    # 从 g0 中提取 gen_0_map_config 的 body，改用 data['conv1']['input'] 等通用 key
    g1_code = generate_g1_1core(
        os.path.splitext(os.path.basename(onnx_path))[0],
        seq,
        assignments,
        cut_config,
        final_out_addr,
    )
    last_fc = sum(1 for l, _ in seq.layers if isinstance(l, FCLayer))
    output_data_key = f"data['fc_cut{last_fc}']['output']"
    g1_code = g1_code.replace("data['output']", output_data_key)
    g2_code = generate_g2(
        os.path.splitext(os.path.basename(onnx_path))[0],
        final_out_addr,
        seq.output_size,
        num_cores=g2_size_x * g2_size_y,
    )
    g2_code = g2_code.replace("data['output']", output_data_key)

    # 组装完整文件：先公共 + 模型 + Config + load + DataHandler + generate_data，再内联 g0/g1/g2 与 check、main
    file_lines = [
        '"""',
        "由 onnx_to_lenet_py 从 " + onnx_path + " 自动生成。",
        "Tianjic X1 支持 16×10 核，通过 gen_*_map_config(..., size_x=, size_y=) 配置。",
        '"""',
        "import numpy as np",
        "import sys",
        "import os",
        "import torch",
        "import torch.nn as nn",
        "import warnings",
        "import math",
        "from itertools import product",
        "from collections import OrderedDict",
        "sys.path.append(os.getcwd())",
        "",
        "from generator.mapping_utils.prims import pX5, p02, p03, p83, p07, p04, p41, p06, p26, p09, p81",
        "from generator.mapping_utils.map_config_gen import MapConfigGen",
        "from generator.mapping_utils.data_handler import DataHandler",
        "",
        "# ================ 量化与截断 ================",
        "class FakeQuantize(torch.autograd.Function):",
        "    @staticmethod",
        "    def forward(ctx, x, x_min, x_max, x_range):",
        "        ctx.save_for_backward(x, torch.tensor(x_min), torch.tensor(x_max))",
        "        x = x.mul(x_range).round().clamp(-128, 127).mul(1 / x_range)",
        "        return x",
        "    @staticmethod",
        "    def backward(ctx, grad_output):",
        "        x, x_min, x_max = ctx.saved_tensors",
        "        zeros = torch.zeros_like(x)",
        "        ones = torch.ones_like(x)",
        "        mask0 = torch.where(x < x_min, zeros, ones)",
        "        mask1 = torch.where(x > x_max, zeros, ones)",
        "        mask = mask0 * mask1",
        "        grad = grad_output * mask",
        "        return grad, None, None, None",
        "",
        "class FakeQuantizeWithoutLimit(torch.autograd.Function):",
        "    @staticmethod",
        "    def forward(ctx, x, x_range):",
        "        ctx.save_for_backward(x)",
        "        x = x.mul(x_range).round().mul(1 / x_range)",
        "        return x",
        "    @staticmethod",
        "    def backward(ctx, grad_output):",
        "        return grad_output, None",
        "",
        "class Cut(nn.Module):",
        "    def __init__(self, en: bool = False, in_cut_start: int = 4, type_out: int = 1):",
        "        super(Cut, self).__init__()",
        "        self.en = en",
        "        self.in_cut_start = in_cut_start",
        "        self.type_out = type_out",
        "        if type_out == 0:",
        "            self.max, self.min = 0x7fffffff, -0x80000000",
        "        elif type_out == 1:",
        "            self.max, self.min = 0x7f, -0x80",
        "        else:",
        "            raise ValueError",
        "    def forward(self, x):",
        "        if self.en:",
        "            return x.div(2 ** (2 * self.in_cut_start)).floor().clamp(min=self.min, max=self.max)",
        "        else:",
        "            return x",
        "    def __repr__(self):",
        "        return '{}(en={}, in_cut_start={:d}, type_out={:d})'.format(self.__class__.__name__, self.en, self.in_cut_start, self.type_out)",
        "",
        "# ================ 模型（由 ONNX 结构生成）================",
        model_src,
        "",
        load_src,
        "",
        "# ================ 量化配置 ================",
        quant_src,
        "",
        "# ================ DataHandler 与 generate_data ================",
        handler_src,
        "",
        generate_data_src,
        "",
        "# ================ Group 0 (FPGA/输入) ================",
    ]

    # 与 generate_data 一致：第一层输入为 data['conv1']['input']
    g0_str = g0_code.replace("data['input'][(0, 0)]", "data['conv1']['input'][(0, 0)]")
    file_lines.append(g0_str.strip())
    file_lines.append("")
    file_lines.append("# ================ Group 2 (多核输出) ================")
    file_lines.append(g2_code.strip())
    file_lines.append("")
    file_lines.append("# ================ Group 1 (计算) ================")
    file_lines.append(g1_code.strip())
    file_lines.append("")
    file_lines.append("# ================ 校验占位与 check ================")
    file_lines.append("class ResultCompareWithClockSpecificSimulator:")
    file_lines.append("    def __init__(self, data_file_name, save_ref_data_en, phase_en, print_matched, step):")
    file_lines.append("        self.data = {}")
    file_lines.append("    def add_ref_data(self, data): pass")
    file_lines.append("    def run(self): pass")
    file_lines.append("    def show_result(self): pass")
    file_lines.append("")
    file_lines.append("def check(data, compare, empty_offset=0):")
    file_lines.append("    data_1 = {}")
    file_lines.append("    for i in range(32):")
    file_lines.append("        data_1[i] = {}")
    file_lines.append("    chip = (0, 0)")
    last_fc = sum(1 for l, _ in seq.layers if isinstance(l, FCLayer))
    file_lines.append(f"    data_1[empty_offset + 5][(chip, (0, 0))] = [{{'data': data['fc_cut{last_fc}']['output'][(0, 0)], 'addr_start': 0x2000 >> 2, 'type': 1}}]")
    file_lines.append("    compare.add_ref_data(data_1)")
    file_lines.append("")
    file_lines.append("# ================ 主程序 ================")
    file_lines.append("if __name__ == '__main__':")
    file_lines.append("    chip = (0, 0)")
    file_lines.append("    send_to_fpga = True")
    file_lines.append("    # Tianjic X1: 最多 16×10 核，此处用 1×1 计算、8×1 输出组")
    file_lines.append(f"    size_x, size_y = {default_size_x}, {default_size_y}")
    file_lines.append(f"    g2_size_x, g2_size_y = {g2_size_x}, {g2_size_y}")
    file_lines.append("")
    file_lines.append("    handler = " + handler_class_name + "(onnx_path=None)")
    file_lines.append("    data = generate_data(handler)")
    file_lines.append("    cuts = QuantizationConfig()")
    file_lines.append("    config = MapConfigGen()")
    file_lines.append("")
    file_lines.append("    phase = np.ones(50).astype(int)")
    file_lines.append("    clock_in_phase = 20_0000")
    file_lines.append("")
    file_lines.append("    config_0 = gen_0_map_config(phase, clock_in_phase, size_x=size_x, size_y=size_y, data=data, out_data_en=True, chip=chip, in_data_en=not send_to_fpga)")
    file_lines.append("    config.add_config(config_0, core_offset=(1, 0))")
    file_lines.append("")
    file_lines.append("    config_1 = gen_1_map_config(phase, clock_in_phase=clock_in_phase, size_x=size_x, size_y=size_y, in_cut_start_dict=cuts, in_data_en=True, out_data_en=True, chip=chip, data=data, init_data=True)")
    file_lines.append("    config.add_config(config_1, core_offset=(0, 0))")
    file_lines.append("")
    file_lines.append("    config_2 = gen_2_map_config(phase, clock_in_phase, size_x=g2_size_x, size_y=g2_size_y, data=data, in_data_en=True, out_data_en=True, chip=chip, init_data=True, send_to_fpga=send_to_fpga)")
    file_lines.append("    config.add_config(config_2, core_offset=(0, 1))")
    file_lines.append("")
    file_lines.append("    MapConfigGen.add_router_info(config.map_config, core_x_num=16, core_y_num=10)")
    file_lines.append("    config.map_config['sim_clock'] = 10_0000")
    file_lines.append("    config.map_config['step_clock'] = {((0, 0), 0): (10_0000 - 1, 10_0000)}")
    file_lines.append("")
    file_lines.append("    from generator.test_engine import TestMode, TestEngine")
    file_lines.append("    from generator.test_engine.test_config import HardwareDebugFileSwitch")
    file_lines.append(f"    case_file_name = '{case_name}'")
    file_lines.append("    out_files_path = os.getcwd() + '/simulator/Out_files/' + case_file_name + '/'")
    file_lines.append("    if os.path.exists(out_files_path):")
    file_lines.append("        c_path = os.getcwd()")
    file_lines.append("        os.chdir(out_files_path)")
    file_lines.append("        del_command = 'rd/s/q cmp_out' if sys.platform.startswith('win') else 'rm -rf cmp_out'")
    file_lines.append("        os.system(del_command)")
    file_lines.append("        os.chdir(c_path)")
    file_lines.append("    test_config = {'tb_name': case_file_name, 'test_mode': TestMode.MEMORY_STATE,")
    file_lines.append("                   'debug_file_switch': HardwareDebugFileSwitch().close_all.dict, 'test_group_phase': [(0, 1)]}")
    file_lines.append("    tester = TestEngine(config.map_config, test_config)")
    file_lines.append("    assert tester.run_test()")

    out_text = "\n".join(file_lines)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(out_text)
    print(f"已生成: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="ONNX / .pth -> lenet.py 风格单文件（支持 Tianjic 16×10 核）"
    )
    parser.add_argument(
        "input_file",
        nargs="?",
        default=None,
        help="输入模型路径：.onnx 或 .pth 文件（.pth 时需同时指定 --model-class 与 --input-shape）",
    )
    parser.add_argument("--onnx", "-i", dest="onnx_file", help="输入 ONNX 路径（与 input_file 二选一）")
    parser.add_argument("--pth", "-p", dest="pth_file", help="输入 .pth 路径（与 input_file 二选一）")
    parser.add_argument("-o", "--output", default=None, help="输出 .py 文件路径，默认与输入同目录且同名_generated.py")
    parser.add_argument("--case-name", default=None, help="用例名，默认取自输入文件名")
    parser.add_argument("--model-class", default=None, help="[.pth 必填] 模型类完整路径，如 generator.LeNet.lenet_model.lenet.LeNet")
    parser.add_argument("--input-shape", default=None, help="[.pth 必填] 输入 shape 逗号分隔，如 1,1,28,28")
    parser.add_argument("--size-x", type=int, default=1, help="计算组 core 的 size_x（Tianjic 单芯片最大 16）")
    parser.add_argument("--size-y", type=int, default=1, help="计算组 core 的 size_y（Tianjic 单芯片最大 10）")
    parser.add_argument("--g2-size-x", type=int, default=8, help="输出组 Group2 的 size_x")
    parser.add_argument("--g2-size-y", type=int, default=1, help="输出组 Group2 的 size_y")
    args = parser.parse_args()

    # 确定输入文件：位置参数 > --onnx / --pth
    input_path = args.input_file or args.onnx_file or args.pth_file
    if not input_path:
        parser.error("请指定输入文件：位置参数、--onnx 或 --pth 之一")
    if args.onnx_file and args.pth_file:
        parser.error("--onnx 与 --pth 不能同时指定")

    # 根据扩展名或显式参数判断是否为 .pth
    lower = input_path.lower()
    if args.pth_file is not None and input_path == args.pth_file:
        is_pth = True
    elif args.onnx_file is not None and input_path == args.onnx_file:
        is_pth = False
    else:
        is_pth = lower.endswith(".pth")

    if is_pth:
        if not args.model_class or not args.input_shape:
            parser.error("输入为 .pth 时必须指定 --model-class 和 --input-shape")
        try:
            input_shape = tuple(int(x.strip()) for x in args.input_shape.split(","))
        except ValueError:
            parser.error("--input-shape 格式错误，应为逗号分隔整数，如 1,1,28,28")
        if len(input_shape) != 4:
            parser.error("--input-shape 应为 4 维 (N,C,H,W)，如 1,1,28,28")

    if not os.path.isfile(input_path):
        print(f"错误: 文件不存在 {input_path}", file=sys.stderr)
        sys.exit(1)

    base_name = os.path.splitext(os.path.basename(input_path))[0]
    output_path = args.output
    if output_path is None:
        output_path = os.path.join(os.path.dirname(os.path.abspath(input_path)), base_name + "_generated.py")
    output_dir = os.path.dirname(os.path.abspath(output_path))
    out_base = os.path.splitext(os.path.basename(output_path))[0]

    if is_pth:
        onnx_path = os.path.join(output_dir, out_base + ".onnx")
        print(f"正在将 .pth 导出为 ONNX: {onnx_path}")
        _convert_pth_to_onnx(input_path, args.model_class, input_shape, onnx_path)
        print("ONNX 导出完成，开始生成 .py 文件")
    else:
        onnx_path = input_path

    generate_full_lenet_style_file(
        onnx_path,
        output_path,
        case_file_name=args.case_name or base_name,
        default_size_x=args.size_x,
        default_size_y=args.size_y,
        g2_size_x=args.g2_size_x,
        g2_size_y=args.g2_size_y,
    )


if __name__ == "__main__":
    main()
