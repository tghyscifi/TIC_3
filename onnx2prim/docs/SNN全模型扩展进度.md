# SNN 全模型扩展：v0.4.0

## 结论与验收层次

项目中发现的 **11 个源码模型族**已纳入原尺寸、显式整数计算图预设。新增大层分片、ANN/SNN 混合 DAG、不同神经元更新顺序、packed 2-bit 输入/输出与时间读出。新入口 `--temporal` 与旧 ANN、`--graph`、单核 `--snn` 并存。

**这不等于“全部原始 SNN 已无修改导入且能部署到芯片”。**

| 验收层次 | 本次状态 | 不代表 |
|---|---|---|
| 源码模型族、原尺寸拓扑 | 11 个预设；源码仅通过 AST 只读扫描 | 任意 checkpoint、任意参数变体都兼容 |
| 显式整数契约 → 原项目原语 | 合成权重下逐帧、逐节点、状态与读出比较 | 无损浮点等价或训练准确率保持 |
| 阶段/状态核/搬运描述及 JSON | 可导出分片模板，主机负责调度 | JSON 能独立完成网络或每帧可重载全部状态 |
| 原路由、自主时间步、硬件 | 尚未实现/验证 | SHD 旧路由等价、周期或上板结果 |

全尺寸测试不是缩小通道的玩具模型，但**参数和输入仍是合成整数**。测试长度不等于完整数据集评测。

## 1. 覆盖清单

下表支持范围统一为 **显式整数、主机分阶段搬运的计算路径**。

| 模型 ID | 实际尺寸与结构 | 实现与边界 |
|---|---|---|
| `SoundSNN` | 784→128→128→128→10，每层 LIF，时间均值 | 完整四层，首层通道分片、int32 部分和与 bias一次；不是旧16→128映射。训练 threshold1.0、decay/255须与整数契约区分 |
| `snn` | Conv1：12×24²→LIF→Pool12²；Conv2：32×8²→LIF→Pool4²；FC512→10 | snntorch延迟复位与空间状态。两次调用都传同一旧膜，不是连续积分两次 |
| `snn_mnist_s` | stride2、pad2：12×14²→32×7²→FC1568→10 | 三层LIF、真实padding/stride；当前源码FC宽度为1568，旧笔记中的512已不适用 |
| `full` | 同上述stride2网络的 Python 模型 | 使用源码snntorch的整数近似；旧p08映射的阈值、泄漏、padding、状态布局与模型不一致 |
| `snnlenet` | LeNet 1→6→16→400→120→84→10，FC2 p08 | 混合Cut/ReLU/Pool/LIF，统一word地址、独立状态；表示旧混合映射，不冒充Python ReLU与该LIF相同 |
| `HNN` | 256→512→512→10 | 上一脉冲门控旧膜，先泄漏再积分，时间求和；源码bias未参与计算，原IO矩阵须转为API的OI |
| `SNN_forward` | 4→128→128→128→128→2 | 同HNN更新，但最后一帧读出；不是HNN的同构副本 |
| `shd` | 700 ternary输入，四支192→64选择投影/LIF→求和→soma→64→32输出 | 独立分支膜、fire2/5、逐神经元初膜、1000-step逻辑窗口；合成选择表不是旧router证明，输出32按映射定义 |
| `snnlenet/train` | LeNet，FC2 Cut后 `x>=10`，再FC3 | Cast int8→int32；p08严格`x>9`精确表示整数`x>=10`，零窗口保持无状态 |
| `snnlenet/train_lif_float` | 无Cut的浮点LeNet+阈值函数 | 需明确选择整数近似策略；添加Cut会改变原模型，不是无损浮点importer |
| `snnlenet/snnpredict` | LeNet，FC2返回`spk*mem` | 当前beta0/threshold0/下限-10/延迟zero-reset等价于ReLU，不误当二值脉冲；仅BN=False当前配置 |

另有 `full/mapping`：修正空间形状的旧p08契约（threshold1、无泄漏、零窗口），用于比较源码模型/手写映射差异，不计为额外模型族。

清单来自 [project_snn_inventory.py](../project_snn_inventory.py)，预设见 [project_snn_models.py](../project_snn_models.py)。没有 import 原训练文件、下载数据或覆盖 checkpoint。

## 2. 神经元更新顺序

独立参考侧使用整数方程；目标侧执行原 `Memory` 与原语。`floor` 对负数也向下取整。

| 契约 | 方程/顺序 | lowering |
|---|---|---|
| p08 | 积分→严格阈值→复位→泄漏→窗口/下限 | p08；reset0/1/2，fire2/3/4/5 |
| Sound | `u=v+current; s=u>th; v'=floor((s?0:u)*beta/256)+offset` | p08；beta256时关闭泄漏，避免写入8位alpha |
| HNN | `u=floor(v*(1-s_prev)*beta/256)+current; s=u>th; v'=u` | 先p08门控/泄漏，再p08积分；新膜不提前zero-reset |
| snntorch delayed subtract | `reset=(v>th); u=floor(v*beta/256)+current-reset*th; s=u>th; v'=u` | p08比较旧膜、p08泄漏、p83阈值校正、p02累加、p08读出；zero/none另行lowering |

snntorch复位来自**上一膜电位**，不同于输入积分后立即复位。重复调用的约定单独记录。beta0.9采用显式230/256近似；阈值64是示例整数声明，不能从原浮点阈值1.0/0.3自动推断。即使beta可精确表示，也不证明所有原浮点膜与floor整数膜相同。

源码profile超过保守signed28界限会拒绝，不悄悄饱和；原生p08保留其真实clip/reset行为。

状态输出：`.membrane`为signed28膜；`.refractory`为真正packed高4位计数；源码profile额外提供`.previous_spike`，不把它称为不应期；`.window`是主机逻辑计数。

## 3. 分片与状态

```text
逻辑张量 → 输出通道/空间/输入通道分片
  → 主机收集输入与卷积halo
  → p41 int32部分和（每个输出片只有首个输入片加bias）
  → p02 int32累加，不提前int8截断
  → Cut/Pool或持久LIF状态片
  → 主机拼回逻辑张量，供下一节点使用
```

卷积默认4×4输出空间、最多32输出通道；FC最多512输入分块，卷积输入块还受kernel/SRAM限制。状态按最多512神经元切片分配不同核，样本内不重复初始化；临时计算复用 `(15,9)`，状态核最多159个。超出预算明确失败。

目标实际读写检查 word地址、SRAM bank、初始化区和长度。“核分配”描述状态所有权与模板，**没有建立片间p09路由**。权重加载、halo、拼接、阶段搬运和时间读出仍需主机。

原p08计数写回只有 `Tw_cnt & 0xff`。新入口用主机逻辑计数表示1000-step窗口，每帧设置真实p08 `Tw_cnt=0`，按窗口结束选择 `Tw_len=0/1`，结束动作仍由p08执行。manifest显式记录主机依赖；不声称原硬件自主计数器支持1000。旧`--snn`仍拒绝大于255的窗口。

## 4. 可复现运行

在项目根目录运行，输出目录必须不存在：

```powershell
$env:PYTHONPATH = "$PWD\.prototype_deps_tianjic;$PWD"
& D:\Anaconda\envs\tianjic\python.exe examples/prototype/run_project_snn_suite.py --output prototype_runs/my_snn_suite --steps 3 --export-stages SoundSNN shd
& D:\Anaconda\envs\tianjic\python.exe -m onnx2prim prototype_runs/my_snn_suite/00_SoundSNN/model.json --temporal --input prototype_runs/my_snn_suite/00_SoundSNN/input.npy --quantization prototype_runs/my_snn_suite/00_SoundSNN/quant.json --output prototype_runs/my_sound_recompiled --verify
```

默认验证全部11族；float LeNet明确选用了添加Cuts的近似策略。可用 `--models SoundSNN HNN` 选子集。`--export-stages`选择需要完整JSON的模型，避免生成大量无用模板。结果不代表训练精度。

### 自己的参数

```python
from onnx2prim.project_snn_models import build_project_graph
from onnx2prim.temporal_compiler import compile_temporal, verify_temporal_behavior
from onnx2prim.quantization import QuantizationConfig

# 全部仿射层 -> {'weight': int8范围整数数组, 'bias': int32范围整数数组}
# Conv OIHW，FC OI；LeNet fc1传源码HWC列顺序，adapter转换为CHW。
graph = build_project_graph('SoundSNN', parameters=parameters)
# 审核/设置节点neuron的整数阈值、衰减和reset，再使用已量化T×shape输入。
c = compile_temporal(graph, QuantizationConfig({}), frames)
verify_temporal_behavior(c)
```

神经元整数契约通过 `NeuronContract`/`P08Contract` 设置。预设的threshold64、beta230/256不是校准结果；使用真实参数还必须评估精度。参数字典须完整，未知层、浮点参数、范围越界、未折叠BN均拒绝，没有自动`.pth`加载或恢复超参数。

SHD真实参数须提供 `shd_selectors (4,2,192)`：0..699选择源列，-1补零；各branch权重 `(64,192)`。折叠到 `(64,700)` 时重复列累加，越界int8则拒绝。**能够表达已声明选择表，不证明它等于旧路由**；默认选择表与输出初膜都是合成fixture，可在图契约中替换。

### ONNX约定

`--temporal`也接受静态batch1、单输入单输出ONNX，standard opset11..18，custom `com.tianjic` version1。必需metadata `snn_profile=tianjic-temporal-v1`、`snn_dt`；可指定 `snn_input_dtype=int8/int2`、`snn_readout=spikes/sum/mean/last`。

白名单：Conv/Gemm、Add、channel Concat、Relu、无padding floor MaxPool、CHW Flatten别名、`com.tianjic::Cut/LIF/Cast`。LIF profile可为p08/sound/snntorch/hnn；Cast显式把int8/int2升到int32。动态形状、Loop、未知属性、任意浮点神经元展开子图等拒绝，不猜语义。**这不是NIR或标准通用ONNX SNN格式。**

## 5. 输出解读

| 文件 | 意义 |
|---|---|
| suite `report.json` | 每个模型尺寸、验证状态、分片数、时间、近似与未完成项；不是部署成功证明 |
| 各模型 `model.json`、`weights.npz` | 白名单typed DAG与参数；参数使用安全编号p000000等，不把模型名当路径或代码 |
| `input.npy`、`quant.json` | 合成T帧、整数参数契约；不是芯片数据包/校准结果 |
| 两套`*_outputs.npz`、`trace_index.json` | 参考/行为逐帧节点、状态、读出；安全编号对应逻辑名在索引中 |
| `compiled/manifest.json` | schema4：stage来源、字地址、状态初始化边界、主机搬运/窗口依赖、真实验证状态 |
| `compiled/stage_configs/stageNNNNNN.json` | 每片零动态输入模板、真实PIC和初始参数；不能单独代表网络，也不能每帧重新加载所有状态 |
| `compiled/mapping.py` | 从快照重建阶段，显式 `host_staging_required=True`，不是路由或chip时间驱动 |
| `compiled/*_outputs.npz` | 导出轨迹的安全编号对应表在manifest的output_archives中 |

每次新验证调用从同一初态重新开始样本。验证深拷贝原语，防止把末态作为初态导出。只调用map-only序列化，不调用会删除历史目录的 `TestEngine.run_test()`。

## 6. 验收与仍未完成的兼容范围

三帧原尺寸11族逐元素计算测试已通过；最终回归数字与CLI记录随后补充。可复现结果见 [套件报告](../../prototype_runs/snn_full_suite_001/report.json)。

测试覆盖：复位/泄漏顺序、负电流、延迟复位、beta0/128/230/256、阈值256乘法边界、padding/stride halo、784输入/bias一次、packed ternary、分支求和、readout、逐神经元初态、窗口999→1000→0、样本复位、可搬迁导出、非法形状/参数/路径。

称为“全部原模型端到端芯片兼容”前，还须：

1. 导入真实checkpoint与输入，明确量化尺度，并验证近似误差与任务准确率。
2. 核实SHD p09/p06选择/重排，验证旧映射等价，不能用合成选择表代替证明。
3. 实现片间p09、权重换入、芯片时间输入驱动，以及宽窗口/时间读出安排。
4. 按需折叠BN=True等变体，逐项验收不同参数配置。
5. 运行硬件exe/周期仿真，再上板验证资源、吞吐与真实数据集。

原generator、primitive、训练文件、权重与历史仿真结果未改动。不自动修复原映射以使它与新参考“看起来一致”。
