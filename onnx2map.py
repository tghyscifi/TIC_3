"""
简单脚本/模块：从 ONNX 模型（或 PyTorch `.pth`）生成对应的 map_config 字典。

支持传入 `.pth` 文件并指定网络类与输入尺寸，会将其导出为临时 ONNX
然后再执行后续转换。

用法示例（脚本模式）：
    python onnx2map.py --onnx lenet.onnx --outdir ./generated
或
    python onnx2map.py --pth lenet.pth --model-class generator.LeNet.lenet.LeNet \
                       --input-shape 1,1,28,28 --outdir ./generated

也可直接在代码中调用 generate_map_config_from_onnx() 得到字典数据并进一步处理。

使用 `.pth` 时需安装 `torch`（见 README）。

本工具基于 workspace 中已有的 onnx2prim 模块实现。
"""

import os
import argparse
import numpy as np
from typing import Optional, Tuple

from onnx2prim.onnx_parser import parse_onnx
from onnx2prim.address_allocator import allocate_addresses, allocate_addresses_for_stage
from onnx2prim.codegen import (
    generate_g0,
    generate_g1_1core,
    generate_g2,
    generate_gi_compute,
    generate_g_out,
    generate_top_py,
    build_cut_config,
)
from onnx2prim.hardware_config import DEFAULT_HARDWARE, HardwareConfig
from onnx2prim.memory_planner import plan_stages
from generator.mapping_utils.map_config_gen import MapConfigGen


def _exec_code(code: str, namespace: dict) -> None:
    """在给定 namespace 中执行字符串形式的 python 代码。"""
    exec(code, namespace)


def _convert_pth_to_onnx(pth_path: str, model_class: str, input_shape: Tuple[int, ...], onnx_path: str) -> None:
    """Load a PyTorch checkpoint and export the model to ONNX.

    *model_class* should be a full import path including the class name,
    e.g. ``generator.LeNet.lenet.LeNet``. The class is instantiated with no
    arguments; adjust externally if initialization requires settings.
    """
    import importlib
    import torch

    module_name, class_name = model_class.rsplit('.', 1)
    mod = importlib.import_module(module_name)
    ModelCls = getattr(mod, class_name)
    model = ModelCls()
    # load state dict (handles files that wrap in {'state_dict':...})
    ckpt = torch.load(pth_path, map_location='cpu')
    if isinstance(ckpt, dict) and 'state_dict' in ckpt:
        ckpt = ckpt['state_dict']
    model.load_state_dict(ckpt, strict=False)
    model.eval()

    dummy = torch.randn(tuple(input_shape))
    torch.onnx.export(model, dummy, onnx_path, opset_version=11,
                      input_names=['input'], output_names=['output'])


def generate_map_config_from_onnx(
    onnx_path: str,
    g2_cores: int = 8,
    phase_length: int = 50,
    clock_in_phase: int = 20_0000,
    mem_limit: Optional[int] = None,
    force_single_group: bool = False,
) -> dict:
    """读取 ONNX 文件并构建 map_config 字典。

    如果网络较大，会根据 hardware_config 和 mem_limit 自动拆分到多个
    计算组（stage），并生成对应的 g0/g1..gK/g_out/top 结构。

    参数
    -------
    onnx_path: ONNX 模型路径
    g2_cores: 单组模式下 g2 核心数量
    phase_length: 生成的 phase 向量长度
    clock_in_phase: 时钟周期
    mem_limit: 每核可用字节上限；若为 None，使用 DEFAULT_HARDWARE
    force_single_group: 若为 True，则不拆分，多核也只生成单组

    返回值
    -------
    map_config 字典（已包含路由信息），不包含 data 或量化配置。
    """
    seq = parse_onnx(onnx_path)

    # 硬件/内存设定
    hw = DEFAULT_HARDWARE if mem_limit is None else HardwareConfig(per_core_usable_bytes=mem_limit)
    stages = plan_stages(seq, hw=hw)
    use_multi_group = (not force_single_group) and (len(stages) > 1 or stages[0].size_x * stages[0].size_y > 1)

    cut_config = build_cut_config(seq)

    # 统一生成 g0 代码并获取函数
    model_name = os.path.splitext(os.path.basename(onnx_path))[0]
    g0_code = generate_g0(model_name, seq.input_shape,
                          seq.input_shape[0] * seq.input_shape[1] * seq.input_shape[2])
    ns: dict = {}
    _exec_code(g0_code, ns)
    gen_0_map_config = ns['gen_0_map_config']

    # prepare phase vector
    phase = np.zeros(phase_length, dtype=int)
    phase[:] = 1

    config = MapConfigGen()
    # always add g0
    config_0 = gen_0_map_config(phase, clock_in_phase, size_x=1, size_y=1,
                                data=None, out_data_en=False, chip=(0, 0), in_data_en=False)
    config.add_config(config_0, core_offset=(0, 0))

    if use_multi_group:
        # multi-stage generation; groups named by their core_offset
        core_offsets = []
        stage_sizes = []
        group_names = []
        final_out_addr = 0x2000
        for stage in stages:
            assignments, _, final_out_addr = allocate_addresses_for_stage(
                seq, stage.block_indices, seq.input_shape
            )
            stage_sizes.append((stage.size_x, stage.size_y))
            core_offsets.append(stage.core_offset)
            ox, oy = stage.core_offset
            name = f"g_{ox}_{oy}"
            group_names.append(name)

            gi_code = generate_gi_compute(
                model_name, seq, stage.block_indices, assignments,
                cut_config, final_out_addr, stage.size_x, stage.size_y,
                group_name=name,
            )
            _exec_code(gi_code, ns)
            gen_gi = ns[f"gen_{name}_map_config"]
            cfgi = gen_gi(phase, clock_in_phase, size_x=stage.size_x, size_y=stage.size_y,
                           in_cut_start_dict=cut_config, data=None,
                           in_data_en=False, out_data_en=False, chip=(0, 0), init_data=False)
            config.add_config(cfgi, core_offset=stage.core_offset)

        # add output group
        g_out_code = generate_g_out(seq.output_size, final_out_addr, 1, 1)
        _exec_code(g_out_code, ns)
        cfg_out = ns['gen_g_out_map_config']
        cfgo = cfg_out(phase, clock_in_phase, size_x=1, size_y=1,
                        data=None, in_data_en=False, out_data_en=False,
                        chip=(0, 0), init_data=False, send_to_fpga=False)
        config.add_config(cfgo, core_offset=(0, 0))

        # top-level assembly (optional)
        top_code = generate_top_py(
            os.path.basename(os.path.dirname(os.path.abspath(onnx_path))),
            group_names, core_offsets, stage_sizes, case_file_name=model_name)
        # top code is not executed here, but the map_config produced above is already complete

    else:
        # single-group mode
        assignments, weight_end, final_out_addr = allocate_addresses(seq)
        g1_code = generate_g1_1core(model_name, seq, assignments, cut_config, final_out_addr)
        g2_code = generate_g2(model_name, final_out_addr, seq.output_size, num_cores=g2_cores)
        _exec_code(g1_code, ns)
        _exec_code(g2_code, ns)
        gen_1_map_config = ns['gen_1_map_config']
        gen_2_map_config = ns['gen_2_map_config']

        cfg1 = gen_1_map_config(phase, clock_in_phase, size_x=1, size_y=1,
                                 in_cut_start_dict=cut_config, data=None,
                                 in_data_en=False, out_data_en=False, chip=(0, 0), init_data=False)
        config.add_config(cfg1, core_offset=(0, 0))

        cfg2 = gen_2_map_config(phase, clock_in_phase, size_x=g2_cores, size_y=1,
                                 data=None, in_data_en=False, out_data_en=False,
                                 chip=(0, 0), init_data=False, send_to_fpga=False)
        config.add_config(cfg2, core_offset=(0, 0))

    MapConfigGen.add_router_info(config.map_config)
    return config.map_config


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="从 ONNX (或 PyTorch .pth) 自动生成 map_config 字典并保存为 json")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--onnx", "-i", help="ONNX 模型路径")
    group.add_argument("--pth", help="PyTorch checkpoint 文件 (.pth) 路径")
    parser.add_argument("--model-class", help="用于导入 .pth 的模型类路径，例如 generator.LeNet.lenet.LeNet")
    parser.add_argument("--input-shape", help="Dummy 输入 shape，用逗号分隔，例如 1,1,28,28")
    parser.add_argument("--outdir", "-o", default=".", help="输出目录（用于保存生成的 map_config.json）")
    parser.add_argument("--g2-cores", type=int, default=8, help="[单组模式] g2 核数")
    parser.add_argument("--mem-limit", type=int, default=None,
                        help="每核可用字节上限；影响自动拆分阶段")
    parser.add_argument("--single-group", action="store_true",
                        help="强制单组生成（忽略多阶段拆分）")
    args = parser.parse_args()

    # prepare onnx file path
    if args.pth:
        if not args.model_class or not args.input_shape:
            parser.error("--pth requires --model-class and --input-shape")
        tmp_onnx = os.path.join(args.outdir, "_temp.onnx")
        shape = tuple(int(x) for x in args.input_shape.split(','))
        _convert_pth_to_onnx(args.pth, args.model_class, shape, tmp_onnx)
        onnx_path = tmp_onnx
    else:
        onnx_path = args.onnx

    mcfg = generate_map_config_from_onnx(
        onnx_path,
        g2_cores=args.g2_cores,
        mem_limit=args.mem_limit,
        force_single_group=args.single_group,
    )
    import json

    os.makedirs(args.outdir, exist_ok=True)
    json_path = os.path.join(args.outdir, "map_config.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(mcfg, f, indent=2)
    print(f"已生成 map_config 并保存到 {json_path}")
