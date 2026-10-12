# SNN 手写映射与 onnx2prim v0.3.0

> 本文保留 v0.3.0 单核 `--snn` 后端的历史边界。v0.4.0 新增独立的 `--temporal` 分片计算入口，最新进展见 [SNN全模型扩展进度](SNN全模型扩展进度.md)；旧入口没有因此获得自动路由部署能力。

本轮实现可运行的确定性 Conv/FC + LIF 单核计算链，不把所有名为 SNN 的目录视为完整、正确或具有相同神经元语义。原模型、权重、手写映射与 primitive 文件均未修改。实际芯片端时间步输入/重放、多核路由与周期级验证仍不在本轮范围。

## 1. 手写映射核对结果

| 对象 | 当前代码实际做了什么 | 本轮覆盖 | 仍不兼容/需修正 |
|---|---|---|---|
| `generator/shd` | 700 路输入，多核/路由重排；隐藏组分四支 p41→p08（fire_type=2、cut=2），再 p02 求和→p08（fire_type=5）；输出 p41→p08（threshold=60000、fire_type=2、cut=11） | p41 FC、严格阈值、alpha/256 泄漏、reset、fire_type=2/3 int8 膜电位读出与持久状态 | 不是简单 700→64→64→20 的三层链：分支/求和、packed 2-bit 输入/输出、A2S2、路由、随机逐神经元初态和 1000-step 窗口尚不支持；不能宣称完整 SHD 已导入 |
| `generator/SoundSNN` | 模型代码有四层 LIFFCNeuron，训练版输入 784，输出按时间取均值；但 `g1_1core.py` 目前只有 **cin=16→128 的 p41 + pX5**，没有 p08，也没有其余三层 | 四层 FC-LIF 合成缩减网络可运行；阈值与状态从显式契约获得，而非猜训练对象 | 原四层 784→128→128→128→10 第一层权重超过本版单 bank 分配；需权重切分/多核。输出 mean 不在芯片计算链中。训练版 decay/255 与目标 alpha/256 不相同 |
| `generator/snnlenet/snnlenet.py` | ANN LeNet 的大部分阶段保留 pX5；FC2 120→84 后换为 p08，threshold=10、fire_type=4、Tw_en=False/Tw_len=0 | 独立 FC-LIF 的阈值、无泄漏与零窗口每步复位可表示 | ANN Pool/Cut 与 SNN LIF 混合图尚不能一次编译；地址单位/输入输出地址有错配风险，需先修正旧映射，不应将它当 golden oracle |
| `generator/full.py` | 两个卷积和 FC 后均接 p08；模型与映射尺寸/地址存在不一致 | 修正尺寸后的 Conv-LIF→Conv-LIF→Flatten→FC-LIF 小型链路可表示 | 旧原语首层缺 padding 却按 14×14 注释计数；FC soma 配 12 而模型输出 10；状态和输出重叠、多层状态共用地址、零窗口，以及真实模型的大尺寸/权重问题；未宣称完整权重部署 |

来源：

- [SHD hidden](../../generator/shd/shd_hidden.py)：292–411 附近包含四支 fire_type=2 与末端 fire_type=5；[SHD output](../../generator/shd/shd_output.py)：134 附近为输出 LIF。
- [SoundSNN 映射](../../generator/SoundSNN/g1_1core.py)：约 45–67 行仅第一层；[训练模型](../../generator/SoundSNN/sound_snn.py)：LIFFCNeuron 的 integrate→strict-threshold→reset→decay，以及四层模型和时间均值；[硬件参数](../../generator/SoundSNN/snn_config.py) 是 threshold=64、decay=128，而训练文件内另有 threshold=1.0、decay=255 的同名类。
- [混合 snnlenet](../../generator/snnlenet/snnlenet.py)：约 615–695 行。p41 输出是 `0x400 >> 2`，p08 读取 `0x400`；p08 写 `0x600`，后续 FC3 却读取 `0x600 >> 2`；注释为“字节”的状态地址也未除四。新编译器从统一 word plan 生成地址，不沿用这些常量。
- [full 映射](../../generator/full.py)：约 282–345 行。首层 28、kernel5、stride2、无 padding，按几何应为 12×12，而 soma 注释为 14×14；末层 FC 的 soma `addr_v == addr_s`，且前两层共享膜状态起始地址。

`SoundSNN/data_handler.py` 的 pretrained 路径指向 `quantized_model.pth`，与目录中训练 `.pt` 的模型状态封装也不一样。本轮没有加载或转换这些 checkpoint，不评估识别精度。根目录 snntorch 脚本是训练代码，不等于已有完整、语义等价的手写映射。

## 2. p08 的实际契约与风险

以 [Prim_08_lif_new.py](../../primitive/Prim_08_lif_new.py) 的 `prim_execution()` 为准：

1. 从 Addr_V 读 packed int32：低 28 位为有符号膜电位，高 4 位为不应期。
2. 仅在不应期为 0 时累加 int32 current；先做 28-bit 饱和。
3. 使用严格 `V > Vth`；Vth 至少为 VL。不能替换成 `>=`。
4. 发放后执行 reset（回到 VR / 减阈值 / 减 dV），然后执行泄漏；乘法分母为 256，负数向下取整。
5. `Tw_cnt >= Tw_len` 会覆盖泄漏结果、回到 Vinit，这个判断不检查 Tw_en。因此 Tw_len=0 不是“无限时间”，而是每次调用后复位。
6. fire_type=2 输出复位/泄漏前膜电位，3 输出新膜电位，4 输出 0/1。只有 2/3 的读出使用 in_cut_start，4 不用其缩放 current。
7. 更新不应期、窗口计数，写回膜状态、Vtheta 和四字参数区。

本版关闭随机阈值与阈值自适应，VM/Vtheta 初始化为零。阈值等参数限制在明确的 28-bit 区间，以免依赖原实现 `Vth.clip()` 未赋回结果的行为。

原实现还有两个影响时间轨迹的细节：

- `out_reg` 在每次执行后原地 reshape 为 (group_num, neu_num)。重放多组时下一次给它赋扁平 spike 会失败。新 lowering 的行流水关闭、空间神经元展成一个 HWC 平序组，保持数据顺序，无需改原文件。
- 参数写回把窗口计数压入 `Tw_cnt & 0xff`。虽然手写 SHD 配置 Tw_len=1000，但不能从 Python 对象的大整数计数推断芯片同样支持。v0.3.0 保守拒绝大于 255 的窗口；后续要根据硬件文档/RTL 确认真实窗口位宽及参数重载方式。

## 3. 扩展模块与验证方法

| 模块 | 作用 |
|---|---|
| `snn_ir.py` | SNNBlock/SNNModel/LIFSpec，显式 dt、编码、初态、reset/leak/window、StateSpec；拒绝歧义 |
| `snn_parser.py` | 单帧 Conv/Gemm→com.tianjic::LIF，允许 LIF 后 CHW Flatten，验证 metadata、属性、维度和线性连接 |
| `snn_reference.py` | 不依赖原语/打包的逻辑整数电流与 LIF 独立 oracle |
| `snn_planner.py` | 常驻膜状态/VM/Vtheta/para、激活与电流 scratch、参数分 bank；重算覆盖和区域防止伪造地址元数据 |
| `snn_codegen.py` | p41+p08，C32/HWC 与 CHW-FC 权重重排，统一 word 地址 |
| `snn_compiler.py` | 逐时间步真实 Memory/原语执行、四类轨迹比较、样本级重置、初始 PI 导出 |

验收不只比较最后一层 spike：对每一层、每一时间步比较读出、膜电位、不应期、窗口计数。每个验证样本初始化一次 SRAM，时间步内只更新输入。PI 深拷贝后推进，防止原语内部计数污染导出初态；再次验证同一 compilation 必须得到相同结果。

实际例子：

- `prototype_runs/snn_fc_run_001`：4→3→2，6 个时间步、2 个 LIF block；8 项（两层 × 四种）时间轨迹。
- `prototype_runs/snn_spatial_run_001`：2×3×5→Conv3×3×5→Flatten→FC3，6 个时间步；检查非正方形、padding、补齐通道和展平次序。
- 额外回归包含四层 256→128→128→128→10 的合成整数网络和参数 Mem0 spill，及 SHD 风格的 192→32、fire_type=2、alpha128/256、cut2（窗口缩至可表示的 200 步）。这些均不是实际 checkpoint 或完整 SHD/SoundSNN 的等价证明。

复现命令和八个输出文件含义见 [README 第八节](../README.md#8-snn显式神经元契约与时间轨迹)。文件中会明确标注 `hardware_temporal_driver=NOT implemented`。JSON 是单帧 compute 模板；T 步输入由 Python 验证器供给，不会自动在芯片上重放。

## 4. 本轮实际验收

2026-10-12：**99 项 unittest 全部通过**（原 ANN 64 项 + SNN 新增 35 项），`compileall` 通过。两个 CLI 结果目录各有八个文件，两层 × 四种轨迹的 NPZ 数组全部一致；生成的 `mapping.py` 重新编译并验证通过。

除了独立 NumPy oracle 与原 Memory/p08，还直接实例化项目 `SoundSNN.sound_snn.LIFFCNeuron` 作第三套 oracle：小尺寸随机整数参数、threshold=4、decay=255/255（对应目标禁用泄漏）时，八步脉冲输出逐元素一致。这不证明其他 decay/255、浮点权重或完整训练 checkpoint 可无损转换。

子代理按用户要求尝试使用 Sol 6.1 / low，但三个子任务均因模型容量不足未启动；本轮代码、独立参考、导入器和审查实际由主代理完成，没有引用未获得的子代理报告。原工程 NumPy 1.x 越界转换产生的弃用警告仍存在，原文件未修改。

## 5. 下一步应优先补什么

1. 先从训练框架导出显式 neuron metadata：证明 threshold/reset/leak 顺序和量化单位一致；snntorch beta=0.9 不能直接写 alpha=0.9 或未经声明地取整成 230。
2. 核对窗口计数位宽、PI 参数区重载、膜状态初始化/样本 reset，再实现芯片端时间步输入与重放；这些不能用 Python 连续调用替代验收。
3. 加 packed 2-bit I/O、p02 求和的有状态 DAG 与 A2S2 调度，才接近完整 SHD 的手写策略。
4. 加权重/通道切分、状态核归属与跨核路由，处理真实 SoundSNN 首层和大尺寸卷积。
5. 混合 ANN/SNN 图需要显式 Cut/Pool 与 LIF 电流尺度连接，而非把 ANN cut 参数塞进 fire_type=4。最后再用真实 checkpoint 评价精度。
