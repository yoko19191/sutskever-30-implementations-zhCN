# 训练工具 (Training Utilities) - 论文 18：关系循环神经网络

## 任务 P2-T3：训练工具和损失函数

本模块为使用 NumPy 实现的 LSTM 和 Relational RNN 模型提供全面的训练工具。

## 文件

- `training_utils.py` - 主工具模块，包含损失函数、训练循环和优化辅助函数
- `training_demo.py` - 所有训练功能的全面演示
- `TRAINING_UTILS_README.md` - 本文档

## 已实现的功能

### 1. 损失函数

#### 交叉熵损失 (Cross-Entropy Loss)
```python
loss = cross_entropy_loss(predictions, targets)
```
- 支持稀疏（类别索引）和独热编码目标
- 使用 log-sum-exp 技巧的数值稳定实现
- 用于分类任务

#### 均方误差 (MSE) 损失
```python
loss = mse_loss(predictions, targets)
```
- 用于回归任务（物体跟踪、轨迹预测）
- 所有元素上平均的简单平方差

#### Softmax 函数
```python
probs = softmax(logits)
```
- 数值稳定的 softmax 实现
- 将 logits 转换为概率

#### 准确率指标
```python
acc = accuracy(predictions, targets)
```
- 分类准确率计算
- 支持稀疏和独热目标

### 2. 梯度计算

#### 数值梯度（有限差分）
```python
gradients = compute_numerical_gradient(model, X_batch, y_batch, loss_fn)
```
- 逐元素有限差分近似
- 教育性实现（慢但正确）
- 使用中心差分：`df/dx ≈ (f(x + ε) - f(x - ε)) / (2ε)`

#### 快速数值梯度
```python
gradients = compute_numerical_gradient_fast(model, X_batch, y_batch, loss_fn)
```
- 向量化梯度估计（比逐元素更快）
- 仍然比解析梯度慢，但更实用
- 适合原型设计和测试

**注意**：对于生产使用，应通过时间反向传播（BPTT）实现解析梯度。

### 3. 优化工具

#### 梯度裁剪
```python
clipped_grads, global_norm = clip_gradients(grads, max_norm=5.0)
```
- 防止梯度爆炸（对 RNN 稳定性至关重要）
- 按全局范数跨所有参数裁剪
- 返回裁剪后的梯度和原始范数以供监控

#### 学习率调度
```python
lr = learning_rate_schedule(epoch, initial_lr=0.001, decay=0.95, decay_every=10)
```
- 指数衰减调度
- 随时间降低学习率以进行微调
- 公式：`lr = initial_lr * (decay ^ (epoch // decay_every))`

#### 早停
```python
early_stopping = EarlyStopping(patience=10, min_delta=1e-4)
should_stop = early_stopping(val_loss, model_params)
best_params = early_stopping.get_best_params()
```
- 通过监控验证损失防止过拟合
- 自动保存最佳参数
- 可配置的耐心值（等待的 epoch 数）和最小改进阈值

### 4. 训练函数

#### 单步训练
```python
loss, metric, grad_norm = train_step(
    model, X_batch, y_batch,
    learning_rate=0.001,
    clip_norm=5.0,
    task='classification'
)
```
- 执行一步梯度下降
- 计算梯度、裁剪它们并更新参数
- 返回损失、指标（准确率或负损失）和梯度范数
- 支持分类和回归任务

#### 模型评估
```python
avg_loss, avg_metric = evaluate(
    model, X_test, y_test,
    task='classification',
    batch_size=32
)
```
- 评估模型而不更新参数
- 按批次处理数据（处理大数据集）
- 返回平均损失和指标

#### 完整训练循环
```python
history = train_model(
    model,
    train_data=(X_train, y_train),
    val_data=(X_val, y_val),
    epochs=100,
    batch_size=32,
    learning_rate=0.001,
    lr_decay=0.95,
    lr_decay_every=10,
    clip_norm=5.0,
    patience=10,
    task='classification',
    verbose=True
)
```

功能：
- 自动批处理，可选洗牌
- 学习率衰减
- 梯度裁剪
- 早停和最佳模型恢复
- 进度跟踪和详细输出
- 返回全面的训练历史

历史字典包含：
- `train_loss`：每个 epoch 的训练损失
- `train_metric`：每个 epoch 的训练指标
- `val_loss`：每个 epoch 的验证损失
- `val_metric`：每个 epoch 的验证指标
- `learning_rates`：使用的学习率
- `grad_norms`：梯度范数（用于监控稳定性）

### 5. 可视化

#### 绘制训练曲线
```python
plot_training_curves(history, save_path='training_curves.png')
```
- 创建 2x2 网格图：
  - 损失随 epoch 变化（训练和验证）
  - 指标随 epoch 变化（训练和验证）
  - 学习率调度
  - 梯度范数
- 如果 matplotlib 不可用则回退到文本输出

## 使用示例

### 基础训练
```python
from lstm_baseline import LSTM
from training_utils import train_model, evaluate

# 创建模型
model = LSTM(input_size=10, hidden_size=32, output_size=3)

# 准备数据
X_train, y_train = ...  # (num_samples, seq_len, input_size)
X_val, y_val = ...

# 训练
history = train_model(
    model,
    train_data=(X_train, y_train),
    val_data=(X_val, y_val),
    epochs=50,
    batch_size=32,
    learning_rate=0.01,
    task='classification'
)

# 评估
test_loss, test_acc = evaluate(model, X_test, y_test)
print(f"测试准确率: {test_acc:.4f}")
```

### 自定义训练循环
```python
from training_utils import train_step, clip_gradients

for epoch in range(num_epochs):
    for X_batch, y_batch in create_batches(X_train, y_train, batch_size=32):
        loss, acc, grad_norm = train_step(
            model, X_batch, y_batch,
            learning_rate=0.01,
            clip_norm=5.0
        )
        print(f"批次损失: {loss:.4f}, 准确率: {acc:.4f}")
```

### 回归任务
```python
# 用于回归（例如，物体跟踪）
history = train_model(
    model,
    train_data=(X_train, y_train),
    val_data=(X_val, y_val),
    task='regression',  # 使用 MSE 损失
    epochs=100
)
```

## 模型兼容性

训练工具可与任何实现以下接口的模型一起使用：

```python
class YourModel:
    def forward(self, X, return_sequences=False):
        """
        参数：
            X: (batch, seq_len, input_size)
            return_sequences: bool
        返回：
            outputs: 如果 return_sequences=False 则为 (batch, output_size)
                    如果 return_sequences=True 则为 (batch, seq_len, output_size)
        """
        pass

    def get_params(self):
        """返回参数名到数组的字典"""
        return {'W': self.W, 'b': self.b, ...}

    def set_params(self, params):
        """从字典设置参数"""
        self.W = params['W']
        self.b = params['b']
```

兼容的模型：
- LSTM（来自 `lstm_baseline.py`）
- Relational RNN（待实现）
- 任何遵循该接口的自定义 RNN 架构

## 测试结果

所有测试都成功通过：

```
✓ 损失函数
  - 交叉熵：完美预测 → 接近零的损失
  - MSE：完美预测 → 零损失
  - 稀疏和独热目标产生相同的结果

✓ 优化工具
  - 梯度裁剪：小梯度不变，大梯度裁剪到 max_norm
  - 学习率调度：指数衰减正常工作
  - 早停：在耐心值 epoch 后无改进时停止

✓ 训练循环
  - 单步：参数正确更新
  - 评估：无参数更新正常工作
  - 完整训练：损失随 epoch 减少
  - 历史跟踪：所有指标正确记录
```

## 性能特征

### 数值梯度
- **优点**：
  - 实现简单
  - 无反向传播错误风险
  - 教育价值

- **缺点**：
  - 非常慢（每步 O(参数量) 次前向传播）
  - 近似（有限差分误差）
  - 不适合大型模型或生产使用

### 建议
1. **用于原型设计**：使用提供的数值梯度
2. **用于实验**：实现快速数值梯度估计
3. **用于生产**：通过 BPTT 实现解析梯度

## 简化与限制

1. **梯度**：数值近似而非解析 BPTT
   - 权衡：简单性与速度
   - 适合教育目的和小模型

2. **优化器**：仅简单 SGD（无动量、Adam 等）
   - 易于扩展更复杂的优化器

3. **批处理**：无并行处理
   - 纯 NumPy 实现（无 GPU 支持）

4. **梯度估计**：快速版本仍然是近似的
   - 使用随机扰动而非逐元素有限差分

## 未来增强

可能的改进（本任务不需要）：
- [ ] 通过 BPTT 进行解析梯度计算
- [ ] Adam 优化器
- [ ] 基于动量的优化
- [ ] 学习率预热
- [ ] 大批次的梯度累积
- [ ] 混合精度训练模拟
- [ ] 更高级的 LR 调度（余弦退火等）

## 与 Relational RNN 的集成

这些工具可以立即与 Relational RNN 模型一起使用。只需确保您的 Relational RNN 实现所需的接口（`forward`、`get_params`、`set_params`），所有训练工具将无缝工作。

示例：
```python
from relational_rnn import RelationalRNN
from training_utils import train_model

# 创建 Relational RNN
model = RelationalRNN(input_size=10, hidden_size=32, output_size=3)

# 像训练 LSTM 一样训练
history = train_model(
    model,
    train_data=(X_train, y_train),
    val_data=(X_val, y_val),
    epochs=50
)
```

## 总结

此实现提供了完整的、仅 NumPy 的训练基础设施，用于：
- **损失计算**：具有数值稳定性的交叉熵和 MSE
- **梯度计算**：数值近似（有限差分）
- **优化**：梯度裁剪、LR 调度、早停
- **训练**：具有指标跟踪的完整训练循环
- **监控**：全面的历史和可视化

所有工具都已测试、记录，并可用于 LSTM 和 Relational RNN 模型。
