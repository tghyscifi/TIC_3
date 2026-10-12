# Training SoundSNN on MNIST Dataset

整体来说，在模型方面，训练 SNN 只需要添加一个代理梯度函数，以便模型能够正常进行反向传播与训练。除这一点外，网络结构本身没有进行修改，仍然保持为原始的四层全连接 SNN 结构。

## 1. 模型训练

训练脚本为：

`python sound_snn_training.py`

在训练过程中，我们主要做了以下几件事：

* 为脉冲神经元添加了代理梯度函数，使模型能够正常训练。
* 保持原始 SoundSNN 的四层全连接结构不变。
* 在训练时使用 MNIST 数据集，并将输入处理为模型所需的时序张量格式。

对于 MNIST 输入，我们的处理方式是：

1. 将每张输入图片从 `[1, 28, 28]` 展平为 `[784]`
2. 将一个 batch 的输入整理为 `[B, 784]`
3. 在时间维上增加一个维度，变为 `[B, 1, 784]`
4. 沿时间维重复 `T = 10` 次，最终得到 `[B, 10, 784]`（所以 sequence_length=10）

这样就适配了当前模型所要求的输入格式 `[Batch_size, T, D]`。

## 2. 并行调参

并行调参使用的脚本为：

`python manager.py`

该脚本用于并行搜索超参数。最终选出的最优超参数为：

* `thresh = 1.0`
* `decay = 255`
* `beta = 0`

并且四个全连接层都使用这一组参数。

## 3. 模型测试

测试脚本为：

`python sound_snn.py`

该脚本会直接加载训练完成后保存的模型权重，并在 MNIST 的 test set 上进行测试。

在测试脚本中，我只对 `sound_snn.py` 做了一处必要修改，即将 `LIFFCNeuron` 的 `forward` 函数中的 `synaptic_input` 初始化改为与输入张量保持相同的 device 和 dtype：

`torch.zeros(self.batch_size, self.time_windows, self.hidden_size, device=input_data.device, dtype=input_data.dtype)`

这一步只是为了确保模型在 GPU 上测试时能够正常运行，不会影响模型本身的结构和功能。

此外，在测试时我将所有 `summary` 记录相关的部分都注释掉了。这不会影响模型的前向推理和测试结果，只是去除了与测试无关的中间记录操作。

## 4. 结果保存

调参结果和最终训练得到的模型权重都保存在： `./results` 目录下。而最优的模型权重为：`./results/thresh_1.0_best_model.pt`

## 5. 使用说明

训练：

`python sound_snn_training.py`

并行调参：

`python manager.py`

测试：

`python sound_snn.py`
