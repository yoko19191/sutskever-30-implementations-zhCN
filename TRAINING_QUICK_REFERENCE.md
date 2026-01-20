# 训练工具 (Training Utilities) - 快速参考

## 安装
```python
# 无需安装 - 纯 NumPy 实现
from training_utils import *
```

## 常用工作流程

### 基础分类训练
```python
from lstm_baseline import LSTM
from training_utils import train_model, evaluate

model = LSTM(input_size=10, hidden_size=32, output_size=3)

history = train_model(
    model,
    train_data=(X_train, y_train),
    val_data=(X_val, y_val),
    epochs=50,
    batch_size=32,
    learning_rate=0.01,
    task='classification'
)

test_loss, test_acc = evaluate(model, X_test, y_test)
```

### 回归训练
```python
history = train_model(
    model,
    train_data=(X_train, y_train),
    val_data=(X_val, y_val),
    task='regression',  # 使用 MSE 损失
    epochs=100
)
```

### 使用所有功能
```python
history = train_model(
    model,
    train_data=(X_train, y_train),
    val_data=(X_val, y_val),
    epochs=100,
    batch_size=32,
    learning_rate=0.01,
    lr_decay=0.95,           # 学习率衰减 5%
    lr_decay_every=10,       # 每 10 个 epoch
    clip_norm=5.0,           # 梯度裁剪到范数 5
    patience=10,             # 早停耐心值
    task='classification',
    verbose=True
)
```

## 函数参考

### 损失函数
```python
# 分类
loss = cross_entropy_loss(predictions, targets)  # targets: (batch,) 或 (batch, n_classes)

# 回归
loss = mse_loss(predictions, targets)  # 连续值的 MSE

# 准确率
acc = accuracy(predictions, targets)  # 分类准确率 [0, 1]
```

### 单步训练
```python
loss, metric, grad_norm = train_step(
    model, X_batch, y_batch,
    learning_rate=0.01,
    clip_norm=5.0,
    task='classification'
)
```

### 评估
```python
loss, metric = evaluate(
    model, X_test, y_test,
    task='classification',
    batch_size=32
)
```

### 梯度裁剪
```python
clipped_grads, global_norm = clip_gradients(grads, max_norm=5.0)
```

### 学习率调度
```python
lr = learning_rate_schedule(
    epoch,
    initial_lr=0.001,
    decay=0.95,
    decay_every=10
)
```

### 早停
```python
early_stop = EarlyStopping(patience=10, min_delta=1e-4)

for epoch in range(epochs):
    # ... 训练 ...
    if early_stop(val_loss, model.get_params()):
        print("早停触发！")
        best_params = early_stop.get_best_params()
        model.set_params(best_params)
        break
```

### 可视化
```python
plot_training_curves(history, save_path='training.png')
```

## 历史字典

```python
history = {
    'train_loss': [1.2, 1.1, 1.0, ...],      # 每个 epoch 的训练损失
    'train_metric': [0.3, 0.4, 0.5, ...],    # 每个 epoch 的训练指标
    'val_loss': [1.3, 1.2, 1.1, ...],        # 每个 epoch 的验证损失
    'val_metric': [0.25, 0.35, 0.45, ...],   # 每个 epoch 的验证指标
    'learning_rates': [0.01, 0.01, ...],     # 每个 epoch 使用的学习率
    'grad_norms': [0.5, 0.4, 0.3, ...]       # 每个 epoch 的梯度范数
}
```

## 数据格式

### 输入数据
```python
X_train: (num_samples, seq_len, input_size)  # 序列
y_train: (num_samples,)                       # 类别标签（分类）
         或 (num_samples, output_size)        # 目标值（回归）
```

### 模型接口
```python
class YourModel:
    def forward(self, X, return_sequences=False):
        # X: (batch, seq_len, input_size)
        # return: (batch, output_size) if return_sequences=False
        pass

    def get_params(self):
        return {'W': self.W, 'b': self.b}

    def set_params(self, params):
        self.W = params['W']
        self.b = params['b']
```

## 超参数建议

### 小数据集 (< 1000 样本)
```python
epochs=100
batch_size=16
learning_rate=0.01
lr_decay=0.95
lr_decay_every=10
clip_norm=5.0
patience=10
```

### 中等数据集 (1000-10000 样本)
```python
epochs=50
batch_size=32
learning_rate=0.01
lr_decay=0.95
lr_decay_every=5
clip_norm=5.0
patience=10
```

### 大数据集 (> 10000 样本)
```python
epochs=30
batch_size=64
learning_rate=0.01
lr_decay=0.95
lr_decay_every=5
clip_norm=5.0
patience=5
```

### 过拟合迹象
```python
# 检查训练-验证差距
train_acc = history['train_metric'][-1]
val_acc = history['val_metric'][-1]
gap = train_acc - val_acc

if gap > 0.1:  # 过拟合
    # 解决方案：
    # - 增加耐心值（更多 epochs）
    # - 使用更小的学习率
    # - 添加正则化（未实现）
    # - 获取更多数据
```

### 欠拟合迹象
```python
# 训练和验证准确率都低
if train_acc < 0.6 and val_acc < 0.6:
    # 解决方案：
    # - 增加模型大小（hidden_size）
    # - 训练更长时间（更多 epochs）
    # - 增加学习率
    # - 检查数据质量
```

## 常见问题

### 损失为 NaN
```python
# 可能原因：
# 1. 学习率过高 → 降低学习率
# 2. 梯度爆炸 → 检查 clip_norm
# 3. 数值不稳定 → 损失函数使用稳定实现

# 解决方案：
learning_rate=0.001  # 降低
clip_norm=1.0        # 降低裁剪阈值
```

### 损失不下降
```python
# 可能原因：
# 1. 学习率过低
# 2. 任务类型错误
# 3. 数据/标签不匹配

# 检查：
print(f"损失: {loss}, 指标: {metric}")
print(f"预测: {model.forward(X_batch[:1])}")
print(f"目标: {y_batch[:1]}")
```

### 训练太慢
```python
# 数值梯度很慢
# 加速训练：
# 1. 使用更小的批次
# 2. 减小模型大小
# 3. 使用更少的 epochs
# 4. 实现解析梯度（BPTT）
```

## 测试

### 快速测试
```bash
python3 test_training_utils_quick.py
```

### 完整测试套件
```bash
python3 training_utils.py
```

### 演示
```bash
python3 training_demo.py
```

## 文件

- `training_utils.py` - 主实现 (37KB)
- `training_demo.py` - 演示 (11KB)
- `test_training_utils_quick.py` - 快速测试 (5KB)
- `TRAINING_UTILS_README.md` - 完整文档 (10KB)
- `TRAINING_QUICK_REFERENCE.md` - 本文件 (8KB)
- `TASK_P2_T3_SUMMARY.md` - 任务总结 (9KB)

## 后续步骤

1. 使用相同接口实现 Relational RNN
2. 使用这些工具训练 LSTM 和 Relational RNN
3. 在推理任务上比较性能
4. （可选）实现解析梯度以加速训练
