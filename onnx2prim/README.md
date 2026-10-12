# onnx2prim 原型：使用与精修入口

当前版本是一个**经过计算原语行为验证的 ANN/SNN 单核原型**，不是完整部署编译器。它把静态 ONNX 拓扑转成显式 IR，完成张量布局、地址分配和原语生成，再逐阶段对照独立整数参考实现。默认线性入口保留 LeNet 路径；`--graph` 支持微型残差网络（第七节），`--snn` 支持显式 p08 神经元契约与逐时间步状态验证（第八节）。

SNN 状态推进目前由 Python 验证器实现。尚未实现芯片端时间步输入/重放驱动、G0/G2 路由、多核拆分、周期级时序或实芯片验证；这些能力不包含在“原语行为验证通过”的结论中。

## 1. 已完成的范围

```text
ONNX → LayerSeq → ComputeBlock → 显式量化契约
     → 张量打包 / 单核 SRAM 地址分配 / 静态校验
     → p81 / p41 + pX5 等效原语对象 → map_config
     → 原项目 JSON 序列化器 → hardware_config.json
     → 原项目 Memory/Axon/Soma 执行 ↔ 独立整数参考
```

支持的编译子集：batch=1、静态 NCHW、单条无分支计算链；入口为无 padding 的卷积（cin≤16），后续普通卷积与 FC；ReLU；卷积后融合无 padding 的 2×2、stride=2 MaxPool；CHW 或显式 NHWC 展平。

解析器还可以表示部分更宽的拓扑，例如独立 FC 或其他池化尺寸；**解析成功不等于后端支持**。编译器会进一步拒绝尚未验证的配置。Conv→FC 时若通道补齐引入额外数据，当前会拒绝，尚未实现压紧原语。

上述默认线性入口不支持分支/残差。新增 DAG 入口支持限定的分支/Add，但仍不支持 BN、任意动态形状/时间循环、未知/自定义算子、group/dilation 卷积、自动 PTQ、ONNX 内嵌 Div/Floor/Clip 量化子图。不能把 SNN 当 ANN 静默编译。

## 2. 实际运行环境

本机完整测试使用：`D:\Anaconda\envs\tianjic\python.exe`，Python 3.10.20、NumPy 1.26.4、PyTorch 2.12.0+cpu。

ONNX 1.16.2 和 protobuf 5.29.6 已隔离安装到项目 `.prototype_deps_tianjic`，没有修改原有 conda 环境。`.prototype_deps` 是另一个 Python 3.12 的解析器测试依赖目录，不能混入 Python 3.10 环境。

原始原语中存在依赖 NumPy 1.x 的越界 int8 转换；NumPy 2.x 不适用于当前行为后端。新解析/布局模块并不依赖 PyTorch，实际原语执行依赖。

在项目根目录打开 PowerShell：

```powershell
$env:PYTHONPATH = "$PWD\.prototype_deps_tianjic;$PWD"
& D:\Anaconda\envs\tianjic\python.exe -m unittest discover -s tests/prototype -v
```

在另一台机器上，可以向独立环境安装 `requirements-prototype.txt`，再提供与原工程兼容的 PyTorch。不要把本机隔离目录的二进制直接复制到不同 Python 版本中使用。

## 3. 复现 LeNet 最小闭环

下面的目录名必须尚不存在；重复运行时换成新的名字。示例生成器使用确定性随机整数参数，只验证拓扑和编译正确性，**不代表已训练 LeNet 的识别精度**。

```powershell
$env:PYTHONPATH = "$PWD\.prototype_deps_tianjic;$PWD"
& D:\Anaconda\envs\tianjic\python.exe examples/prototype/make_lenet_fixture.py prototype_runs/my_fixture
& D:\Anaconda\envs\tianjic\python.exe -m onnx2prim prototype_runs/my_fixture/lenet.onnx --input prototype_runs/my_fixture/input.npy --quantization prototype_runs/my_fixture/quantization.json --output prototype_runs/my_result --verify
```

这里的 `--verify` 执行的是原项目 `Memory`、Axon、Soma 的逐阶段原语行为，与独立 NumPy 整数参考逐元素比较。它不是 `TestEngine.run_test()`，没有运行整个 ChipArray 的路由/时序，也没有调用 `TianjicX1_SIM.exe`。

本实现不会调用旧测试入口的全局清理逻辑，不删除已有仿真结果或 `simulator/Debug_files`。输出目录已存在时直接报错。

### 输出文件

| 文件 | 内容与用途 |
|---|---|
| `model.onnx` | 导入模型的自包含快照，供生成入口重新编译 |
| `input.npy` | 已量化的 CHW、int8 范围整数输入，不是原始浮点输入 |
| `quantization.json` | 完整 Cut 字典、参数量化模式及显式尺度 |
| `mapping.py` | 可重建 `map_config` 的 Python 入口；使用类型化 builder，不插入 ONNX 名称作为代码 |
| `hardware_config.json` | 通过原工程序列化器生成的 ChipArray 配置；不是原始含 tuple-key 的 map_config；尚未用 exe/芯片验证 |
| `manifest.json` | 编译范围、字地址分配、各项验证是否实际执行；用于审查和对比 |
| `reference_outputs.npz` | 独立整数参考的每个计算 block 输出，CHW 或向量布局 |
| `behavior_outputs.npz` | 仅在 `--verify` 成功后生成；原项目原语执行的同布局输出 |

目前已有一份与最终回归代码对应的运行结果：`prototype_runs/lenet_run_002`。其目录被 gitignore 排除，源代码、测试、生成器才是可复现资产。

## 4. 量化契约：必须人工决定的部分

模型的浮点 ONNX 输出，不应直接与此整数编译结果声称相等。这里验证的是“显式整数模型 → 原语”的一致性。

配置示例：

```json
{
  "cuts": {"cut1": 4, "cut2": 4, "fc_cut1": 5, "fc_cut2": 4, "fc_cut3": 5},
  "parameter_mode": "integer"
}
```

- `integer`：ONNX 参数虽然可以是 float 存储，但数值必须已经是整数，权重落在 int8、偏置落在 int32 范围。
- `scaled`：必须由使用者选定尺度，权重按 `weight_scale`、偏置按 `bias_scale` 乘后取最近整数；默认值为 128 和 16384，但不会自动开启此模式。
- 不会默默裁剪超范围参数，不会从张量尺寸猜 Cut。每个 block 必须指定对应 Cut。
- 激活量化为 `floor(x / 2^(2*cut))`，再饱和到 int8；含 ReLU 时下限为 0，否则为 -128。
- 缺少配置、超范围、未验证的累计溢出都会报错。此原型没有校准数据采集、自动选尺度或量化精度评估。

示例配置借用了手写 LeNet 的 Cut，但仅用于明确的测试用例，不能直接套给任意模型。

## 5. 按什么顺序人工精修

| 顺序 | 先读/改的模块 | 验收方式 |
|---|---|---|
| 1 | `ir.py`、`onnx_parser.py` | 新增非正方形、展平、分支/动态形状用例；检查来源与逻辑维度 |
| 2 | `quantization.py`、`reference.py` | 换成实际模型与真实量化配置；分清浮点误差与编译误差 |
| 3 | `layout.py`、`address_allocator.py`、`validate.py` | 校验补齐维度、word 地址、每个 SRAM bank 边界、共享 scratch |
| 4 | `codegen/runtime.py`、`tests/prototype/test_pipeline.py` | 对照手写 LeNet 的 primitive 字段，逐阶段比较结果 |
| 5 | `compiler.py` | 审查生成文件、输入快照和验证报告，保证错误不伪装成成功 |
| 6 | `snn_ir.py`、`snn_reference.py`、`snn_compiler.py` | 微型 LIF 时间轨迹 oracle 与 p08 已接通；继续核对训练库阈值、reset/leak 顺序和状态量化，不能直接抄参数 |
| 7 | 后续路由/阶段模块 | 单独验证 G0/G2、时序、再多核拆分，最后才评价吞吐与芯片结果 |

主代理完成的接口与语义审查见 `docs/原型审查.md`。

## 6. 与旧入口的关系

旧 `onnx2map.py`、`onnx_to_lenet_py.py` 没有被改写。它们的 imports 已有对应符号，但 G0/G2、多组/top 生成仍明确报 Unsupported；**旧完整流程没有恢复可用**。

`generate_g1_1core` 提供了可执行的单核 compute adapter，要求 packed data 按 IR 层名称组织。主流程推荐 `python -m onnx2prim` 或 `compile_onnx()`，不依赖旧共享命名空间 `exec` 管线。底层 `compile_compute()` 是给已经验证的 IR/分配/打包数据使用的 builder；使用者应走主编译入口。

现存根目录 `lenet_temp.onnx` 在本次读取中 protobuf 解码失败；`lenet_generated.onnx` 可解码，但含当前白名单之外的 BN 和量化/shape 子图。原文件均未修改，不能把它们的失败当成合成用例通过了真实 checkpoint 部署。

## 7. 新增 DAG / 残差计算入口

第一轮扩展加入 `GraphIR`、独立 DAG parser、按节点量化、按消费者生命周期复用激活内存，以及真实原语 Add/Tanh lowering。默认 LeNet 入口未替换。

```powershell
$env:PYTHONPATH = "$PWD\.prototype_deps_tianjic;$PWD"
& D:\Anaconda\envs\tianjic\python.exe examples/prototype/make_residual_fixture.py prototype_runs/my_residual_fixture
& D:\Anaconda\envs\tianjic\python.exe -m onnx2prim prototype_runs/my_residual_fixture/residual.onnx --graph --input prototype_runs/my_residual_fixture/input.npy --quantization prototype_runs/my_residual_fixture/quantization.json --output prototype_runs/my_residual_result --verify
```

示例是 H=6、W=8 的三个残差块合成网络，输入 4 通道、内部 32 通道、输出 6 通道和 Tanh；包含 19 个逻辑节点、26 个 primitive phase。可以设置 `--height`、`--width`，但超出 SRAM 或 PI 容量会报错，不会自动多核切分。最终验证结果目录见 `docs/ANN扩展进度.md`。

### DAG 支持边界

- 支持普通 Conv 的显式首层 padding、相同 CHW 形状的 Add、Relu、无 padding 的 floor MaxPool、显式量化的 Tanh。
- 所有激活采用 **HWC/C32** 物理布局。卷积输入通道和权重相应补零，不能与默认 LeNet 的 p81/C16 路径混用。C32 是首版兼容策略，有额外 SRAM 开销，不是最优布局。
- Add 会生成两个 p06 搬运阶段，使两支在临时区连续排列，再使用 p02 求和、pX5 截断；skip tensor 保留到所有消费者用完。
- Conv 和 Add 各自必须有 Cut，以**节点名称**为键；不从形状推断，不随意改变 Cut 的位置。`scaled` 模式支持每个 Conv 独立 weight/bias scale，但要求显式声明激活尺度；不同尺度的 Add 输入被拒绝。
- `activation_scales` 是声明和分支一致性检查，不是自动量化校准或完整浮点等价证明。省略它仅允许整数模式，表示预先声明的公共整数单位。
- Tanh 需指定 `tanh_input_scales` 与 `tanh_output_scales`，数学契约为 `floor(clip(tanh(x/input_scale)*output_scale,-128,127))`，目标使用 p07 的 256 项表。例如输入尺度 32、输出尺度 128 对应工程残差示例的整数 Tanh 约定；不能声称任意原始浮点 Tanh 与其相等。
- 有 padding 的 MaxPool 暂时拒绝：原 primitive 的补零可能改变全负边界结果，不能把 ONNX 的负无穷边界语义忽略。
- DAG 后端尚未支持 FC/Flatten、AvgPool/GlobalAveragePool、BN、复杂量化子图、多核/多芯片及路由。解析成功不等于完整模型兼容。

### 结果与测试

输出仍包含 model、input、quantization、mapping、hardware_config、manifest 和两套逐节点输出。DAG manifest 为 schema v2，额外记录张量生命周期、物理地址与节点对应的 primitive phase。`reference_outputs.npz` 以 ONNX value 名为键，包含输入；行为输出包含全部计算节点。

测试覆盖微型残差的负值、饱和与独立 Cut，256 个有符号 Tanh 输入，无 padding 非正方形 MaxPool，三块残差网络的逐节点行为比较，以及导出 mapping.py 的重新编译。未运行全芯片/路由仿真或外部 exe。

后续实际 ResidualBlocks 与 ResNet 的兼容性进度、审查及下一阶段见 `docs/ANN扩展进度.md`。

## 8. SNN：显式神经元契约与时间轨迹

新增接口：`parse_onnx_snn()`、`compile_snn()`、`compile_onnx_snn()`、`verify_snn_behavior()`、`export_snn_artifacts()`。`compile_ir()` 可接受 `SNNModel`；对任意旧 `ModelIR.states/time` 仍明确拒绝，不能据此推断它们具有同一种 LIF 语义。

```text
单时间步 ONNX（Conv/Gemm → com.tianjic::LIF）
  → SNNModel / LIFSpec / 显式膜电位、不应期、窗口状态
  → 独立整数参考逐步推进 ↔ 原项目 Memory + p41 + p08 逐步推进
  → 单帧 map_config / JSON 模板 + 完整 T 步结果及状态轨迹
```

### 输入形式与支持边界

- 支持单条 Conv/Gemm + LIF 链，可有多层卷积、纯 FC，或卷积 LIF 后 `Flatten(axis=1)` 再 FC。Conv 支持显式 padding、普通 stride，拒绝 group/dilation；Flatten 的 CHW 权重会显式转换到物理 HWC/C32，补齐通道对应权重置零。
- ONNX 描述**一帧**运算，静态 batch=1 输入为 `[1,N]` 或 `[1,C,H,W]`；实际输入文件为 `T×N` 或 `T×C×H×W` 的 int8 范围整数数组。`binary-spikes` 仅允许 0/1，`int8-current` 允许 -128..127。不支持批量多样本，重新调用验证器会从同一初态重新开始一个样本。
- 自定义节点为 `com.tianjic::LIF`，domain opset 为 1。这是本项目的显式导入约定，**不是标准 ONNX 或 NIR 神经元格式**。标准 ONNX opset 接受 11..18；LIF 未被拆解/猜测成 Add/Where/Loop 子图。
- 必需 metadata：`snn_profile=tianjic-p08-v1`、`snn_dt=正有限数`、`snn_encoding=int8-current` 或 `binary-spikes`。dt 只描述步长，不会自动换算成 chip clock。
- LIF 必须显式声明 `threshold`、`leak_enabled`、`leak_alpha`；其他字段及默认值见 `snn_ir.LIFSpec`。浮点阈值、未知属性、阈值自适应、随机阈值、外部状态输入、递归/分支/Loop 都被拒绝。神经元契约不能从普通 `.pth` 权重自动恢复。
- 支持 reset 0（回到 reset_value）、1（减去阈值）、2（减去 reset_decrement）；fire_type 4（int8 0/1 脉冲）、2（复位/泄漏前膜电位截断读出）、3（新膜电位截断读出）。fire_type 5 的 packed 2-bit 读出尚未实现。
- 量化仍人工显式指定；SNN 配置必须为 `{"cuts":{},"parameter_mode":"integer"}` 或显式 scaled 模式。仿射电流保持 int32；`readout_cut` 位于 LIF，**fire_type=4 时不会改变电流或发放阈值**。不会把 ANN 的 Cut 字典静默套到 LIF 电流上。
- 单核最多 31 个计算 block + 1 个初始空 phase；独立常驻膜电位/参数区不可与激活或 current scratch 重叠。超出 Mem0/Mem1 或需要权重跨 bank 的大层直接拒绝，尚未空间/通道切分。

### 神经元语义：对照实际 p08，而非模型名称

```text
v_integrated = clip28(v + (ref == 0 ? current : 0))
threshold = max(threshold, floor)
spike = v_integrated > threshold
v_reset = 按 reset_mode 复位 v_integrated
v_next = window >= window_length ? initial_v
         : (leak_enabled ? floor(v_reset * leak_alpha / 256) : v_reset) + leak_beta
v_next = max(clip28(v_next), floor)
ref_next = spike ? refractory : max(ref - 1, 0)
window_next = window_enabled ? (window == window_length ? 0 : window + 1) : window
```

`Tw_finish` 的判断与 `Tw_en` 无关，因此 `window_length=0` 即使禁用计数仍每次复位膜电位。本版保留并测试这个行为，不把它解释成持续积分；持续状态推荐 `window_enabled=false, window_length=1, window_initial=0`。alpha 的分母是 **256**；不能直接当作训练代码中 `decay/255`。负数泄漏采用向下取整。

窗口暂限 0..255：原 p08 写回 Addr_para 时仅保留 `Tw_cnt & 0xff`，SHD 手写代码中的 1000 尚不能据此宣布芯片可表示。`p08` 原 Python 实现重放多组输出还有形状问题，本版关闭行流水，将空间 HWC 神经元展成一组；没有修改原始 primitive。

### 运行与结果文件

在项目根目录运行，目录名必须尚不存在：

```powershell
$env:PYTHONPATH = "$PWD\.prototype_deps_tianjic;$PWD"
& D:\Anaconda\envs\tianjic\python.exe examples/prototype/make_snn_fixture.py --output prototype_runs/my_snn_fixture
& D:\Anaconda\envs\tianjic\python.exe -m onnx2prim prototype_runs/my_snn_fixture/model.onnx --snn --input prototype_runs/my_snn_fixture/input.npy --quantization prototype_runs/my_snn_fixture/quantization.json --output prototype_runs/my_snn_result --verify
```

给生成器增加 `--spatial` 可生成 2×3×5 输入的 Conv-LIF→Flatten→FC-LIF。两个夹具都含 6 步，使用合成整数权重，不代表训练模型准确率。`--snn` 与 `--graph` 互斥，普通 ANN 入口仍拒绝自定义 LIF。

| 输出 | SNN 版本的含义 |
|---|---|
| `model.onnx` | 含显式 LIF 属性与 metadata 的单帧计算模型 |
| `input.npy` | T 帧输入；不是已部署的芯片输入包 |
| `quantization.json` | 参数整数化契约；cut 字典为空，读出 cut 在 LIF 属性中 |
| `mapping.py` | 重建单帧 PI 和状态初值的 Python 入口；可再调用验证器运行 T 步 |
| `hardware_config.json` | 单帧 compute 模板及状态初值；未生成芯片输入、时间循环或 G0/G2，不可直接宣称 T 步部署 |
| `manifest.json` | schema v3；逐层 LIF 属性、状态和字地址、dt、T、实际验证状态与部署限制 |
| `reference_outputs.npz` | 每层 T 步读出及 `.membrane`、`.refractory`、`.window` 轨迹 |
| `behavior_outputs.npz` | 原 Memory/p08 的同布局轨迹；仅 `--verify` 成功后生成 |

每个时间步、每一层的上述四种轨迹均逐元素比较。原语对象中的窗口计数会变化，所以验证器深拷贝 PI，避免验证后导出末态当成初态；膜电位/不应期从真实 Memory 解码，样本内不重复初始化。

手写 SNN 的逐项兼容性与问题清单见 [SNN手写映射兼容性](docs/SNN手写映射兼容性.md)。本节所述 v0.3.0 单核 `--snn` 入口尚未完整兼容大型、混合与分支模型；v0.4.0 的独立分阶段计算扩展见下一节。

2026-10-12 验收：99 项 unittest 全部通过（原 ANN 64 项 + 新 SNN 35 项），compileall 通过。除 NumPy 独立参考外，还对照原项目 LIFFCNeuron 的整数无泄漏配置验证八步脉冲一致；未加载训练 checkpoint。实际 CLI 结果为 `prototype_runs/snn_fc_run_001` 与 `prototype_runs/snn_spatial_run_001`，各 6 步；原项目文件与历史仿真结果未改动。

## 9. v0.4.0：全尺寸、混合与分支 SNN 计算

新增 `--temporal`：显式整数 temporal DAG，支持 Conv/FC/Cut/Relu/MaxPool/Add/Concat/Cast/LIF；大层按输入/输出通道及空间分片，部分和由真实 p02 在 int32 中累加，持久膜状态按核隔离。Sound、HNN、snntorch 延迟复位分别 lowering，新增 packed ternary 输入与 fire_type5 输出，以及 sum/mean/last 时间读出。

项目扫描得到11个源码模型族，均有原尺寸预设。可复现入口：

```powershell
$env:PYTHONPATH = "$PWD\.prototype_deps_tianjic;$PWD"
& D:\Anaconda\envs\tianjic\python.exe examples/prototype/run_project_snn_suite.py --output prototype_runs/my_snn_suite --steps 3 --export-stages SoundSNN shd
& D:\Anaconda\envs\tianjic\python.exe -m onnx2prim prototype_runs/my_snn_suite/00_SoundSNN/model.json --temporal --input prototype_runs/my_snn_suite/00_SoundSNN/input.npy --quantization prototype_runs/my_snn_suite/00_SoundSNN/quant.json --output prototype_runs/my_sound_recompiled --verify
```

这些结果使用**合成整数参数与显式量化契约**，不是全部真实 checkpoint 的兼容证明。分片输入、halo、部分和拼装、宽窗口及时间读出由主机控制；导出的 stage JSON 是零动态输入模板，不能单独完成网络。SHD 的原路由选择关系尚未验证；浮点 LeNet 需明确选择近似量化策略；未折叠 BN 被拒绝。**自主路由部署和硬件 exe/芯片验证没有完成。**

逐模型结构、神经元方程、参数导入、输出解读与验收边界见 [SNN全模型扩展进度](docs/SNN全模型扩展进度.md)。旧 `--snn` 仍为受限单核接口，不自动获得这些能力。
