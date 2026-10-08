# RMSProp with Nesterov Momentum 实验

## 1. 实验任务

本实验基于给定的 NumPy 全连接神经网络完成以下任务：

1. 修改 NN，实现更新策略（RMSProp + Nesterov momentum）
2. 在兵王问题上测试你写出的算法，写出运行结果。
3. 与 Momentum 方法对比，总结此方法的优势和劣势。

## 2. 算法原理

设模型参数为 $\theta$，速度为 $v$，平方梯度的指数移动平均为 $r$，全局学习率为 $\epsilon$，RMSProp 衰减率为 $\rho$，动量系数为 $\alpha$。每个 mini-batch 的更新过程为：

1. 计算 Nesterov 前瞻参数：

   $$
   \tilde{\theta}_t=\theta_t+\alpha v_{t-1}.
   $$

2. 在前瞻位置计算梯度：

   $$
   g_t=\frac{1}{m}\nabla_{\tilde{\theta}_t}
   \sum_{i=1}^{m}L\left(f(x^{(i)};\tilde{\theta}_t),y^{(i)}\right).
   $$

3. 使用 RMSProp 累积梯度平方：

   $$
   r_t=\rho r_{t-1}+(1-\rho)g_t\odot g_t.
   $$

4. 更新速度：

   $$
   v_t=\alpha v_{t-1}
   -\frac{\epsilon}{\sqrt{r_t}+\delta}\odot g_t.
   $$

5. 更新参数：

   $$
   \theta_{t+1}=\theta_t+v_t.
   $$

其中 $\delta$ 是防止除零的数值稳定项。本实验使用：

```python
RMSPROP_DECAY = 0.9       # rho
NESTEROV_MOMENTUM = 0.9  # alpha
OPTIMIZER_EPSILON = 1e-8 # delta
```

RMSProp 根据每个参数近期的梯度尺度自适应调整步长；Nesterov momentum 则先沿历史速度向前观察，再使用前瞻位置的梯度修正更新方向。

## 3. 程序修改

### 3.1 `NN.py`：初始化优化器状态

RMSPropNesterov 同时需要速度 $v$ 和平方梯度累积量 $r$。因此在原有初始化条件中加入 `RMSPropNesterov`：

```python
if method in ("Momentum", "RMSPropNesterov"):
    self.vW[k] = np.zeros((height, width), dtype=float)
    self.vb[k] = np.zeros((height, 1), dtype=float)

if method in ("AdaGrad", "RMSProp", "Adam", "RMSPropNesterov"):
    self.rW[k] = np.zeros((height, width), dtype=float)
    self.rb[k] = np.zeros((height, 1), dtype=float)
```

启用 Batch Normalization 时，`Gamma` 和 `Beta` 也是可训练参数，需要相应的速度和累积量：

```python
if method == "RMSPropNesterov":
    self.vGamma[k] = 0
    self.vBeta[k] = 0

if method in ("AdaGrad", "RMSProp", "Adam", "RMSPropNesterov"):
    self.rGamma[k] = 0
    self.rBeta[k] = 0
```

### 3.2 `nn_train.py`：在前瞻位置计算梯度

Nesterov 与普通 Momentum 的关键区别是梯度计算位置。必须在 `nn_forward()` 之前将所有可训练参数移动到前瞻位置：

```python
from nn_applygradient import (
    nn_applygradient,
    NESTEROV_MOMENTUM,
)


def nn_train(nn, train_x, train_y):
    batch_size = nn.batch_size
    m = train_x.shape[0]
    num_batches = m / batch_size
    kk = np.random.permutation(m)

    for l in range(int(num_batches)):
        indices = kk[l * batch_size : (l + 1) * batch_size]
        batch_x = train_x[indices, :]
        batch_y = train_y[indices, :]

        if nn.optimization_method == "RMSPropNesterov":
            alpha = NESTEROV_MOMENTUM
            for k in range(nn.depth - 1):
                nn.W[k] = nn.W[k] + alpha * nn.vW[k]
                nn.b[k] = nn.b[k] + alpha * nn.vb[k]

                if nn.batch_normalization:
                    nn.Gamma[k] = nn.Gamma[k] + alpha * nn.vGamma[k]
                    nn.Beta[k] = nn.Beta[k] + alpha * nn.vBeta[k]

        nn = nn_forward(nn, batch_x, batch_y)
        nn = nn_backward(nn, batch_y)
        nn = nn_applygradient(nn)

    return nn
```

### 3.3 `nn_applygradient.py`：加入更新分支

首先在模块顶部定义常量：

```python
RMSPROP_DECAY = 0.9
NESTEROV_MOMENTUM = 0.9
OPTIMIZER_EPSILON = 1e-8
```

未启用 Batch Normalization 时，在原优化器循环中、`Adam` 分支之后加入：

```python
elif method == "RMSPropNesterov":
    rho = RMSPROP_DECAY
    alpha = NESTEROV_MOMENTUM

    nn.rW[k] = rho * nn.rW[k] + (1 - rho) * nn.W_grad[k] ** 2
    nn.rb[k] = rho * nn.rb[k] + (1 - rho) * nn.b_grad[k] ** 2

    # 当前保存的是 theta + alpha*v_old，先恢复 theta。
    nn.W[k] = nn.W[k] - alpha * nn.vW[k]
    nn.b[k] = nn.b[k] - alpha * nn.vb[k]

    nn.vW[k] = alpha * nn.vW[k] - nn.learning_rate * nn.W_grad[k] / (
        np.sqrt(nn.rW[k]) + OPTIMIZER_EPSILON
    )
    nn.vb[k] = alpha * nn.vb[k] - nn.learning_rate * nn.b_grad[k] / (
        np.sqrt(nn.rb[k]) + OPTIMIZER_EPSILON
    )

    nn.W[k] = nn.W[k] + nn.vW[k]
    nn.b[k] = nn.b[k] + nn.vb[k]
```

这里出现减号，是因为 `nn_train.py` 已经执行：

$$
\tilde{\theta}_t=\theta_t+\alpha v_{t-1}.
$$

正式更新必须从原参数 $\theta_t$ 出发，所以先执行：

$$
\tilde{\theta}_t-\alpha v_{t-1}=\theta_t,
$$

再应用新速度 $v_t$。如果不撤销前瞻位移，旧动量会被重复加入一次。

启用 Batch Normalization 时，使用同样的公式更新 `Gamma` 和 `Beta`，代码结构类似。

## 4. 兵王问题实验

两种优化器使用相同的数据、网络、初始化种子和训练轮数：

| 项目 | 设置 |
|---|---|
| 网络结构 | `[6, 20, 20, 20, 20, 2]` |
| 隐藏层激活函数 | Sigmoid |
| 输出层 | Softmax |
| 损失函数 | Cross Entropy |
| Batch Normalization | 开启 |
| Batch size | 100 |
| Epoch | 100 |
| NumPy 随机种子 | 0 |
| 模型选择 | 验证集准确率最高的 epoch |

各优化器参数如下：

| 优化器 | 学习率 | 其他参数 |
|---|---:|---|
| Momentum | 0.01 | 原程序动量系数 0.1 |
| RMSPropNesterov | 0.001 | $\rho=0.9$，$\alpha=0.9$，$\delta=10^{-8}$ |

使用不同学习率是因为 RMSProp 会对梯度进行自适应缩放，直接沿用 Momentum 的学习率容易产生过大的实际更新。本实验比较的是两种方法各自合理配置下的实际效果，不是仅改变优化器名称的单变量消融实验。

### 4.3 运行结果

固定 NumPy 随机种子为 0，运行结果如下：

| 优化器 | 最佳 epoch | 最佳验证准确率 | 测试准确率 | 测试错误数 |
|---|---:|---:|---:|---:|
| Momentum | 86 | 99.8515% | 99.5396% | 62 |
| RMSPropNesterov | 83 | **99.9109%** | **99.7995%** | **27** |

以 `draw` 为类别 0，测试集混淆矩阵为：

```text
Momentum
[[ 1362,   21],
 [   41,12043]]

RMSPropNesterov
[[ 1368,   15],
 [   12,12072]]
```

其中行表示真实类别、列表示预测类别。

## 5. 与 Momentum 的比较

### 5.1 RMSPropNesterov 的优势

1. **能够适应不同参数的梯度尺度。** RMSProp 为每个参数维护平方梯度移动平均。梯度长期较大的方向会得到较小步长，梯度较小的方向则保留较大步长。
2. **能更早修正动量方向。** Nesterov 在 $\theta+\alpha v$ 处计算梯度，可以在参数按照惯性继续移动之前观察前方损失曲面，降低过冲风险。

### 5.2 RMSPropNesterov 的劣势

1. **需要更多内存。** Momentum 每个参数只需额外保存速度 $v$；RMSPropNesterov 还需保存平方梯度累积量 $r$，优化器状态约增加一倍。
2. **单次更新计算量更大。** 它需要前瞻操作、逐元素平方、开平方和除法。
3. **参数更多。** 除学习率外，还需要选择 RMSProp 衰减率 $\rho$、动量系数 $\alpha$ 和数值稳定项 $\delta$。

## 6. 运行方法

创建环境并安装依赖：

```bash
python3 -m venv .venv
pip install -r requirements.txt
```

运行原有兵王实验：

```bash
cd NeuralNetworks
python3 testChess.py
```

公平比较 Momentum 与 RMSPropNesterov：

```bash
.venv/bin/python NeuralNetworks/testOptimizerComparison.py
```

该脚本不会加载旧模型，会自动验证两组初始模型完全一致，并为每个 epoch 设置相同的 mini-batch 随机种子。最佳模型分别保存为：

```text
storedChess_Momentum.pkl
storedChess_RMSPropNesterov.pkl
```

快速检查而不保存模型：

```bash
.venv/bin/python NeuralNetworks/testOptimizerComparison.py --epochs 2 --no-save
```

测试 Momentum 时使用：

```python
learning_rate=0.01,
optimization_method="Momentum",
```

测试 RMSPropNesterov 时使用：

```python
learning_rate=0.001,
optimization_method="RMSPropNesterov",
```

两组实验应使用不同模型文件名，或者在运行前确认不会加载上一组实验的 `storedChess.pkl`，否则测试结果可能来自错误的优化器模型。
