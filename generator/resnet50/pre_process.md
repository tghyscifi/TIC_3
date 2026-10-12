# ImageNet 预处理

验证集的图像数据进行预处理的Pytorch代码如下：
$$

$$

$$
\begin{aligned}
y=&clamp\left({round}\left(\cfrac{\cfrac{x}{128} - \mu}{\sigma} \times 128\right)\right)\\
=&clamp\left(round\left(\cfrac{x}{\sigma}-128 \times \cfrac{\mu}{\sigma}\right)\right)
\end{aligned}
$$

