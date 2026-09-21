# 线性可分性判定程序介绍

## 1. 程序功能

程序文件为 `whetherLinearSeparable.py`，其中实现了题目要求的函数：

```python
def whetherLinearSeparable(X):
    ...
    return Y
```

输入 `X` 是一个二维数组。每一行的最后一个数是类别标签，只能取 `+1` 或 `-1`；其余数值是该样本的特征坐标，因此程序可以处理二维、三维及更高维数据。若数据线性可分，函数返回 `1`；否则返回 `-1`。

## 2. 算法介绍

程序采用线性规划判断是否存在分离超平面。设第 `i` 个 `d` 维样本的特征为 `x_i = [x_i1, ..., x_id]`，标签为 `y_i`，希望找到权重向量 `w` 和偏置 `b`，使所有样本都满足：

```text
y_i * (w · x_i + b) >= 1
```

当 `y_i = +1` 时，该条件要求样本位于超平面的正侧；当 `y_i = -1` 时，要求样本位于负侧。右侧常数取 `1` 不会改变可分性：如果有限点集存在严格分离超平面，就可以同时放大 `w` 和 `b`，使最小间隔达到 `1`。

为了把偏置也作为未知量求解，程序在每个特征向量末尾添加常数 `1`，令 `theta = [w, b]`。上述条件可转换为 SciPy 线性规划函数所需的形式：

```text
-y_i * ([x_i, 1] · theta) <= -1
```

线性规划的目标函数设为全零，因为这里只需判断约束是否存在可行解，不需要寻找某个特定的最优超平面。若 `scipy.optimize.linprog` 找到可行解，则返回 `1`；若判定约束不可行，则返回 `-1`。

根据 `linprog` 所需要的形式
`A_ub · theta <= b_ub`
对比得到 

```python
A_ub = -(labels[:, np.newaxis] * augmented_features)
b_ub = -np.ones(n_samples)
```

## 3. 运行程序

`_validate_labelled_data()` 函数会检查输入是否为非空二维数值数组、是否包含至少一列特征和一列标签、是否含有 `NaN` 或无穷值，以及最后一列是否全部为 `+1/-1`。

在项目根目录运行：

```bash
python3 whetherLinearSeparable.py
```

运行结果为：
```bash
Example 1 (expected 1): 1
Example 2 (expected -1): -1
Example 3 (expected -1): -1
Example 4 (expected 1): 1
Example 5 (expected -1): -1
Saved figure: Example_1.pdf
Saved figure: Example_2.pdf
Saved figure: Example_3.pdf
Saved figure: Example_4.pdf
Saved figure: Example_5.pdf
```