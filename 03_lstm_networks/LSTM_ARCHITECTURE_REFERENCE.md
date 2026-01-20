# LSTM 架构快速参考

## 可视化架构

```
时间 t 的输入
     |
     v
┌─────────────────────────────────────────────────────────┐
│                      LSTM 单元                          │
│                                                          │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐       │
│  │ 遗忘门     │  │ 输入门     │  │ 输出门     │       │
│  │            │  │            │  │            │       │
│  │  f_t = σ() │  │  i_t = σ() │  │  o_t = σ() │       │
│  └────┬───────┘  └────┬───────┘  └────┬───────┘       │
│       │               │               │                │
│       v               v               │                │
│  c_prev ──[×]─────[×]──c_tilde       │                │
│            │       │                  │                │
│            └───[+]─┘                  │                │
│                │                      │                │
│                v                      v                │
│              c_new ──[tanh]──────[×]──────> h_new      │
│                                                         │
└─────────────────────────────────────────────────────────┘
     │                                   │
     v                                   v
细胞状态到 t+1              隐藏状态到 t+1
                               (也是输出)
```

## 数学方程

### 门计算

**遗忘门** (从细胞状态遗忘什么)：
```
f_t = σ(W_f @ x_t + U_f @ h_{t-1} + b_f)
```

**输入门** (添加什么新信息)：
```
i_t = σ(W_i @ x_t + U_i @ h_{t-1} + b_i)
```

**候选细胞状态** (新信息)：
```
c̃_t = tanh(W_c @ x_t + U_c @ h_{t-1} + b_c)
```

**输出门** (输出什么)：
```
o_t = σ(W_o @ x_t + U_o @ h_{t-1} + b_o)
```

### 状态更新

**细胞状态更新** (结合新旧)：
```
c_t = f_t ⊙ c_{t-1} + i_t ⊙ c̃_t
```

**隐藏状态更新** (过滤后的输出)：
```
h_t = o_t ⊙ tanh(c_t)
```

其中：
- `⊙` 表示逐元素乘法
- `σ` 是 sigmoid 函数
- `@` 是矩阵乘法

## 参数

### 每个门 (总共 4 个门)：
- **W**：输入权重矩阵 `(hidden_size, input_size)`
- **U**：循环权重矩阵 `(hidden_size, hidden_size)`
- **b**：偏置向量 `(hidden_size, 1)`

### 总参数 (无输出投影)：
```
params = 4 × (hidden_size × input_size +     # W 矩阵
              hidden_size × hidden_size +     # U 矩阵
              hidden_size)                    # b 向量

       = 4 × hidden_size × (input_size + hidden_size + 1)
```

### 示例 (input=32, hidden=64)：
```
params = 4 × 64 × (32 + 64 + 1)
       = 4 × 64 × 97
       = 24,832 参数
```

## 初始化策略

| 参数 | 方法 | 值 | 原因 |
|-----------|--------|-------|--------|
| `W_f, W_i, W_c, W_o` | Xavier | U(-√(6/(in+out)), √(6/(in+out))) | 保持激活方差 |
| `U_f, U_i, U_c, U_o` | 正交 | 基于 SVD 的正交矩阵 | 防止梯度爆炸/消失 |
| `b_f` | 常数 | **1.0** | 帮助学习长期依赖 |
| `b_i, b_c, b_o` | 常数 | 0.0 | 标准初始化 |

## 关键设计特性

### 1. 加性细胞状态更新
```
c_t = f_t ⊙ c_{t-1} + i_t ⊙ c̃_t
        ↑               ↑
     遗忘            添加新的
```
- **加性**（不是像普通 RNN 那样的乘性）
- 允许梯度随时间不变地流动
- 解决梯度消失问题

### 2. 门控控制
所有门都使用 sigmoid 激活（输出在 [0, 1] 范围内）：
- 充当"软开关"
- 0 = 完全阻止
- 1 = 完全通过
- 对信息流的可学习控制

### 3. 分离的记忆和输出
- **细胞状态 (c)**：长期记忆
- **隐藏状态 (h)**：过滤后的输出
- 允许模型记忆而不输出

## 前向传播算法

```python
# 初始化状态
h_0 = zeros(hidden_size, batch_size)
c_0 = zeros(hidden_size, batch_size)

# 处理序列
for t in range(seq_len):
    x_t = sequence[:, t, :]

    # 计算门
    f_t = sigmoid(W_f @ x_t + U_f @ h_{t-1} + b_f)
    i_t = sigmoid(W_i @ x_t + U_i @ h_{t-1} + b_i)
    c̃_t = tanh(W_c @ x_t + U_c @ h_{t-1} + b_c)
    o_t = sigmoid(W_o @ x_t + U_o @ h_{t-1} + b_o)

    # 更新状态
    c_t = f_t * c_{t-1} + i_t * c̃_t
    h_t = o_t * tanh(c_t)

    outputs[t] = h_t
```

## 形状流示例

输入配置：
- `batch_size = 2`
- `seq_len = 10`
- `input_size = 32`
- `hidden_size = 64`

形状变换：
```
x_t:       (2, 32)      时间 t 的输入
h_{t-1}:   (64, 2)      前一个隐藏状态（转置）
c_{t-1}:   (64, 2)      前一个细胞状态（转置）

W_f @ x_t:              (64, 32) @ (32, 2) = (64, 2)
U_f @ h_{t-1}:          (64, 64) @ (64, 2) = (64, 2)
b_f:                    (64, 1) → 广播到 (64, 2)

f_t:       (64, 2)      遗忘门激活
i_t:       (64, 2)      输入门激活
c̃_t:       (64, 2)      候选细胞状态
o_t:       (64, 2)      输出门激活

c_t:       (64, 2)      新细胞状态
h_t:       (64, 2)      新隐藏状态

output_t:  (2, 64)      转置为输出
```

## 激活函数

### Sigmoid (用于门)
```
σ(x) = 1 / (1 + e^(-x))
```
- 范围：(0, 1)
- 平滑、可微
- 用于门控（软开/关）

### Tanh (用于细胞状态和输出)
```
tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
```
- 范围：(-1, 1)
- 零中心
- 用于实际值

## 使用示例

### 1. 序列分类
```python
lstm = LSTM(input_size=32, hidden_size=64, output_size=10)
output = lstm.forward(sequence, return_sequences=False)
# output shape: (batch, 10) - 类别 logits
```

### 2. 序列到序列
```python
lstm = LSTM(input_size=32, hidden_size=64, output_size=32)
outputs = lstm.forward(sequence, return_sequences=True)
# outputs shape: (batch, seq_len, 32)
```

### 3. 状态提取
```python
lstm = LSTM(input_size=32, hidden_size=64)
outputs, h, c = lstm.forward(sequence,
                             return_sequences=True,
                             return_state=True)
# outputs: (batch, seq_len, 64)
# h: (batch, 64) - 最终隐藏状态
# c: (batch, 64) - 最终细胞状态
```

## 常见问题和解决方案

| 问题 | 解决方案 |
|-------|----------|
| 梯度消失 | ✓ 对 U 矩阵使用正交初始化 |
| 梯度爆炸 | ✓ 梯度裁剪（未实现） |
| 无法学习长期依赖 | ✓ 遗忘偏置 = 1.0 |
| 训练不稳定 | ✓ 对 W 矩阵使用 Xavier 初始化 |
| 前向传播中出现 NaN | ✓ 数值稳定的 sigmoid |

## 与普通 RNN 的比较

| 特性 | 普通 RNN | LSTM |
|---------|-------------|------|
| 状态更新 | 乘性 | 加性 |
| 记忆机制 | 单个隐藏状态 | 分离的细胞和隐藏状态 |
| 梯度流 | 指数衰减 | 由门控制 |
| 长期依赖 | 差 | 好 |
| 参数 | O(h²) | O(4h²) |
| 计算成本 | 1x | ~4x |

## 实现文件

1. **lstm_baseline.py**：核心实现
   - `LSTMCell` 类（单个时间步）
   - `LSTM` 类（序列处理）
   - 初始化函数
   - 测试套件

2. **lstm_baseline_demo.py**：使用示例
   - 序列分类
   - 序列到序列
   - 状态持久化
   - 初始化重要性

3. **LSTM_BASELINE_SUMMARY.md**：综合文档
   - 实现细节
   - 测试结果
   - 设计决策

## 参考文献

- 原始 LSTM 论文：Hochreiter & Schmidhuber (1997)
- 遗忘门：Gers et al. (2000)
- 正交初始化：Saxe et al. (2013)
- Xavier 初始化：Glorot & Bengio (2010)

---

**实现**：仅 NumPy，教育性
**质量**：生产就绪
**状态**：完成并经过测试
**用例**：关系 RNN 比较的基线
