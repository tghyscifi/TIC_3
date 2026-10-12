# ANN 兼容扩展：第一轮实现与后续边界

## 本轮交付

目标是先实现 GraphIR、微型残差块和缩小版 ResidualBlocks 的计算闭环，不把全部 ResNet 或大尺寸多核网络标为已兼容。

主代理负责 IR 契约、生命周期规划、原语 lowering、编译/导出、集成验收。子代理以 Sol 6.1 低思考强度并行完成 DAG parser、独立整数参考、测试夹具，后续另派子代理检查 planner 边界。原有模型、checkpoint、mapping_utils、primitive 和行为模拟器未改动。

2026-10-12 验收：**64 项 unittest 全部通过**，包括上一版全部 30 项回归；compileall 通过。CLI 实际生成并验证 `prototype_runs/residual_run_003`，包含 19 个计算节点、26 个 phase 和八个结果文件。该用例为合成参数，不是训练精度评估。

另在小尺寸输入上直接实例化原项目 `QuantizNet`，使用随机量化参数与原始 Cut 配置作为第三套 oracle：最终输出及七处 ReLU 输出与新 primitive 后端逐元素一致。该测试不加载 checkpoint，也不代表原项目量化 ONNX 子图已能自动导入。

## 能力和模型状态

| 对象 | 当前状态 | 尚缺能力 |
|---|---|---|
| 既有 LeNet 合成参数闭环 | 原默认入口保留，继续回归 | 实际 checkpoint 精度及完整路由/芯片验证 |
| 微型残差块 | GraphIR、Add 和逐节点整数行为支持 | 自动浮点量化与复杂尺度转换 |
| 缩小版 ResidualBlocks 合成参数 | 三个残差块及末端 Tanh 支持，逐节点验证 | 实际 checkpoint、原始 H=560/W=60、多核数据交换 |
| 原始 ResidualBlocks | 不完整兼容，大尺寸超 SRAM 明确拒绝 | 空间切片/halo、跨核路由、阶段与时序规划 |
| ResNet 单芯片定制版 | 尚未兼容 | padded MaxPool、4×4 Pool、GAP、FC、量化子图、通道/空间切分及完整调度 |
| ResNet 双芯片定制版 | 尚未兼容 | 上述能力加跨芯片通信与同步 |
| 标准 ResNet-50 五芯片版 | 尚未兼容 | 完整 Bottleneck 图、更多组/阶段及多芯片资源规划 |

ResNet 单/双芯片目录内是定制缩减网络，不能仅根据目录名推断它们均为标准 50 层模型。其他算法类或演示文档不代表已有完整手写映射。

## 模块和实现路径

```text
graph_parser → GraphIR / topological_nodes
             → graph_reference：显式量化与独立数学 oracle
             → graph_planner：最后消费者、内存复用、临时区与常驻参数
             → graph_codegen：Conv/Relu/Pool/Add/Tanh 对应真实 primitive
             → graph_compiler：逐节点行为比较和报告
```

逻辑形状统一 CHW，物理激活布局统一 HWC/C32，地址统一 4-byte word。参数、激活、Add 拼接临时区和 INT32 scratch 分开；weight/LUT 跨阶段常驻。该策略先保证正确性，不以布局最优或性能最优为目标。

Add 不是将 Python 数组相加后冒充硬件结果：两支分别用 p06 复制到连续临时区，p02 的非平均求和模式读入，pX5 实现 Add 的独立 Cut。比较参考输出时在每个逻辑节点的最后 phase 读取实际 Memory。

Tanh 的参考直接计算数学函数，目标读 p07 LUT；全 256 个 int8 输入的测试同时与 PyTorch 数学函数比较。Tanh scales 为外部显式约定，未实现任意 ONNX 量化子图的自动识别。

## 审查发现与修复

子代理新增的 planner 边界测试发现初稿过于信任分配元数据。已修复：校验器从图重新计算 born/last_use，复核 concat/scratch 的必需长度，并根据参数 shape/布局重新核对 weight/bias/LUT 地址、长度及持久区登记。伪造生命周期、缺失参数、清空持久区或缩短临时区现在都会报错。实际 H=560/W=60 的单核超限也有明确拒绝测试。

## 必须保留的拒绝行为

- 分支 shape 或激活尺度不一致时不自动广播/重标定。
- 循环、缺失输入、重复 value、死节点和未知算子不静默丢弃。
- 超 Mem0/Mem1 或保守 32-phase 单核限制时不放宽硬件边界，也不输出“多核成功”。
- padded MaxPool 不用零边界替代负无穷边界。
- 原工程的 Div/Floor/Clip、BN、AvgPool、FC 图不能靠删除节点来假装支持。
- `hardware_config.json` 的生成不代表外部模拟器接受，也不代表路由或芯片测试通过。

## 后续实施顺序

1. 针对实际 ResidualBlocks 的量化模型制定严格 ONNX 归一化契约，接入 checkpoint/真实 Cut；分清数学模型、整数模型和 primitive 三层误差。
2. 补输入 halo、空间分块及 p06/p09 数据交换，先验证两个相邻 tile 的边界与尾块，再扩大到原输入尺寸。
3. 实现负边界 MaxPool、GAP 的求和/除数/Cut 契约、FC，完成微型 Bottleneck 与投影 shortcut。
4. 面向单芯片定制 ResNet 做通道部分和归并、偏置去重、PI/RHead 校验和全图时序；再推进双芯片与完整五芯片。

运行说明见 `../README.md`。验证只针对这里列出的测试子集，不覆盖全部硬件寄存器字段范围或时间同步约束。
