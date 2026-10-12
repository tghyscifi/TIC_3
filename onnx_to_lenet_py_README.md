# ONNX / .pth → lenet.py 风格单文件转换器

## 功能

`onnx_to_lenet_py.py` 支持两种输入，生成与项目内 `lenet.py` 结构一致的单文件 Python 模块：

- **ONNX**：直接读入 `.onnx` 文件
- **PyTorch 权重**：读入 `.pth` 文件，需指定 `--model-class` 与 `--input-shape`，会先导出为 ONNX 再生成 .py（需安装 `torch`）

生成内容包含：

- **量化与截断**：`FakeQuantize`、`FakeQuantizeWithoutLimit`、`Cut`
- **模型类**：由 ONNX 拓扑生成的 PyTorch 模型（Conv / MaxPool / FC）
- **QuantizationConfig**：各层权重量化与 cut 参数
- **DataHandler 子类**：从 ONNX 加载权重、跑前向并抓取各层输入/输出
- **generate_data()**：按 Tianjic 存储与对齐要求做 `tensor_split`
- **gen_0_map_config / gen_1_map_config / gen_2_map_config**：Group0（FPGA/输入）、Group1（计算）、Group2（多核输出）映射配置
- **check()** 与 **main**：仿真对比与主入口

## Tianjic X1 核与存储

- 单芯片为 **16×10** 核，通过 `size_x`、`size_y` 配置。
- 存储布局与原语（p06/p09/p41/p81/pX5 等）参见文档 **TianjicX1.docx**。
- 生成文件中 `MapConfigGen.add_router_info(..., core_x_num=16, core_y_num=10)` 已按 16×10 核配置。

## 用法

```bash
# ---------- 从 ONNX 生成 ----------
# 基本：读入 lenet.onnx，生成 lenet_generated.py（默认与 ONNX 同目录）
python onnx_to_lenet_py.py lenet.onnx -o lenet_generated.py

# 指定输出路径与用例名
python onnx_to_lenet_py.py lenet.onnx -o ./generated/mymodel.py --case-name mymodel

# ---------- 从 .pth 生成 ----------
# 需指定模型类与输入 shape，会先导出 ONNX 到输出目录，再生成 .py（需 pip install torch）
python onnx_to_lenet_py.py lenet.pth -o lenet_generated.py --model-class generator.LeNet.lenet_model.lenet.LeNet --input-shape 1,1,28,28

# 或用 --pth 显式指定
python onnx_to_lenet_py.py --pth lenet.pth -o out.py --model-class generator.LeNet.lenet_model.lenet.LeNet --input-shape 1,1,28,28

# 配置核数：计算组 1×1，输出组 8×1（默认）
python onnx_to_lenet_py.py lenet.onnx -o out.py --size-x 1 --size-y 1 --g2-size-x 8 --g2-size-y 1
```

## 参数说明

| 参数 | 说明 | 默认 |
|------|------|------|
| `input_file` | 输入文件路径（.onnx 或 .pth），也可用 `--onnx` / `--pth` | 可选 |
| `-o`, `--output` | 输出 .py 文件路径 | 与输入同目录、同名 _generated.py |
| `--case-name` | 用例名（仿真输出目录等） | 输入文件名 |
| `--model-class` | [.pth 必填] 模型类完整路径，如 `generator.LeNet.lenet_model.lenet.LeNet` | - |
| `--input-shape` | [.pth 必填] 输入 shape，逗号分隔，如 `1,1,28,28` | - |
| `--size-x`, `--size-y` | 计算组 Group1 的 core 网格 | 1, 1 |
| `--g2-size-x`, `--g2-size-y` | 输出组 Group2 的 core 网格 | 8, 1 |

## 依赖

- 项目内 `onnx2prim`（onnx_parser、address_allocator、codegen）
- **ONNX 输入**：`onnx`、`numpy`
- **.pth 输入**：还需 `torch`（`pip install torch`），用于加载权重并导出 ONNX
- 运行生成文件时还需 `generator.mapping_utils` 等工程路径

## 如何得到 ONNX

- **直接使用 .pth**：运行本脚本时传入 `.pth` 及 `--model-class`、`--input-shape`，会自动导出 ONNX 并生成 .py。
- 若已有 ONNX，可直接传入 `.onnx` 路径。
- 或使用 PyTorch `torch.onnx.export()` 从现有模型导出 ONNX。

## 限制

- 仅支持线性拓扑：Conv、MaxPool、Gemm/MatMul（FC），以及可忽略的 Relu/Flatten 等。
- 权重从 ONNX initializer 按层名映射加载；若 ONNX 中权重名与生成模型 `conv1.weight`、`fc1.bias` 等不一致，需在 ONNX 或生成代码中做名称对齐。
