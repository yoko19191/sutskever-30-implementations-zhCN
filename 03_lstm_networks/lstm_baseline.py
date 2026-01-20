"""
LSTM Baseline Implementation for Relational RNN Comparison

This module implements a standard LSTM (Long Short-Term Memory) network
using NumPy only. The implementation includes:
- Proper parameter initialization (Xavier/He for input weights, orthogonal for recurrent)
- Forget gate bias initialization to 1.0 (standard trick to help learning)
- LSTMCell for single time step processing
- LSTM wrapper for sequence processing with output projection

Paper 18: Relational RNN Comparison Baseline
"""

import numpy as np


def orthogonal_initializer(shape, gain=1.0):
    """
    Initialize weight matrix with orthogonal initialization.
    This helps prevent vanishing/exploding gradients in recurrent connections.

    Args:
        shape: tuple of (rows, cols)
        gain: scaling factor (default 1.0)

    Returns:
        Orthogonal matrix of given shape
    """
    flat_shape = (shape[0], np.prod(shape[1:]))
    a = np.random.normal(0.0, 1.0, flat_shape)
    u, _, v = np.linalg.svd(a, full_matrices=False)
    q = u if u.shape == flat_shape else v
    q = q.reshape(shape)
    return gain * q[:shape[0], :shape[1]]


def xavier_initializer(shape):
    """
    Xavier/Glorot initialization for input weights.
    Helps maintain variance of activations across layers.

    Args:
        shape: tuple of (rows, cols)

    Returns:
        Xavier-initialized matrix
    """
    limit = np.sqrt(6.0 / (shape[0] + shape[1]))
    return np.random.uniform(-limit, limit, shape)


class LSTMCell:
    """
    带有遗忘门、输入门和输出门的标准LSTM单元。

    架构:
        f_t = sigmoid(W_f @ x_t + U_f @ h_{t-1} + b_f)  # 遗忘门
        i_t = sigmoid(W_i @ x_t + U_i @ h_{t-1} + b_i)  # 输入门
        c_tilde_t = tanh(W_c @ x_t + U_c @ h_{t-1} + b_c)  # 候选单元状态
        o_t = sigmoid(W_o @ x_t + U_o @ h_{t-1} + b_o)  # 输出门
        c_t = f_t * c_{t-1} + i_t * c_tilde_t  # 新单元状态
        h_t = o_t * tanh(c_t)  # 新隐藏状态

    参数:
        input_size: 输入特征的维度
        hidden_size: 隐藏状态和单元状态的维度
    """

    def __init__(self, input_size, hidden_size):
        """
        使用适当的初始化策略初始化LSTM参数。

        参数:
            input_size: int, 输入特征的维度
            hidden_size: int, 隐藏状态和单元状态的维度
        """
        self.input_size = input_size
        self.hidden_size = hidden_size

        # 遗忘门参数
        # 输入权重: Xavier初始化
        self.W_f = xavier_initializer((hidden_size, input_size))
        # 循环权重: 正交初始化
        self.U_f = orthogonal_initializer((hidden_size, hidden_size))
        # 偏置: 初始化为1.0(帮助学习长期依赖的标准技巧)
        self.b_f = np.ones((hidden_size, 1))

        # 输入门参数
        self.W_i = xavier_initializer((hidden_size, input_size))
        self.U_i = orthogonal_initializer((hidden_size, hidden_size))
        self.b_i = np.zeros((hidden_size, 1))

        # 单元门参数(候选值)
        self.W_c = xavier_initializer((hidden_size, input_size))
        self.U_c = orthogonal_initializer((hidden_size, hidden_size))
        self.b_c = np.zeros((hidden_size, 1))

        # 输出门参数
        self.W_o = xavier_initializer((hidden_size, input_size))
        self.U_o = orthogonal_initializer((hidden_size, hidden_size))
        self.b_o = np.zeros((hidden_size, 1))

    def forward(self, x, h_prev, c_prev):
        """
        单时间步的前向传播。

        参数:
            x: 输入, 形状 (batch_size, input_size) 或 (input_size, batch_size)
            h_prev: 前一隐藏状态, 形状 (hidden_size, batch_size)
            c_prev: 前一单元状态, 形状 (hidden_size, batch_size)

        返回:
            h: 新隐藏状态, 形状 (hidden_size, batch_size)
            c: 新单元状态, 形状 (hidden_size, batch_size)
        """
        # 处理输入形状: 将 (batch_size, input_size) 转换为 (input_size, batch_size)
        if x.ndim == 2 and x.shape[1] == self.input_size:
            x = x.T  # 转置为 (input_size, batch_size)

        # 确保x是2D的
        if x.ndim == 1:
            x = x.reshape(-1, 1)

        # 确保h_prev和c_prev是2D的
        if h_prev.ndim == 1:
            h_prev = h_prev.reshape(-1, 1)
        if c_prev.ndim == 1:
            c_prev = c_prev.reshape(-1, 1)

        # 遗忘门: 决定从单元状态中丢弃什么信息
        f = self._sigmoid(self.W_f @ x + self.U_f @ h_prev + self.b_f)

        # 输入门: 决定将什么新信息存储到单元状态中
        i = self._sigmoid(self.W_i @ x + self.U_i @ h_prev + self.b_i)

        # 候选单元状态: 可能添加的新信息
        c_tilde = np.tanh(self.W_c @ x + self.U_c @ h_prev + self.b_c)

        # 输出门: 决定输出单元状态的哪些部分
        o = self._sigmoid(self.W_o @ x + self.U_o @ h_prev + self.b_o)

        # 更新单元状态: 遗忘旧的 + 添加新的
        c = f * c_prev + i * c_tilde

        # 更新隐藏状态: 过滤后的单元状态
        h = o * np.tanh(c)

        return h, c

    @staticmethod
    def _sigmoid(x):
        """数值稳定的sigmoid函数。"""
        return np.where(
            x >= 0,
            1 / (1 + np.exp(-x)),
            np.exp(x) / (1 + np.exp(x))
        )


class LSTM:
    """
    处理序列并产生输出的LSTM。

    这个包装器类使用LSTMCell处理输入序列,并可选择性地
    将隐藏状态投影到输出空间。

    参数:
        input_size: 输入特征的维度
        hidden_size: 隐藏状态的维度
        output_size: 输出的维度(None表示无投影)
    """

    def __init__(self, input_size, hidden_size, output_size=None):
        """
        初始化带有可选输出投影的LSTM。

        参数:
            input_size: int, 输入特征的维度
            hidden_size: int, 隐藏状态的维度
            output_size: int或None, 输出的维度
                        如果为None, 输出为隐藏状态
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # 创建LSTM单元
        self.cell = LSTMCell(input_size, hidden_size)

        # 可选的输出投影层
        if output_size is not None:
            self.W_out = xavier_initializer((output_size, hidden_size))
            self.b_out = np.zeros((output_size, 1))
        else:
            self.W_out = None
            self.b_out = None

    def forward(self, sequence, return_sequences=True, return_state=False):
        """
        通过LSTM处理序列。

        参数:
            sequence: 输入序列, 形状 (batch_size, seq_len, input_size)
            return_sequences: bool, 如果为True返回所有时间步的输出,
                            如果为False仅返回最后一个输出
            return_state: bool, 如果为True同时返回最终的(h, c)状态

        返回:
            如果return_sequences=True且return_state=False:
                outputs: 形状 (batch_size, seq_len, output_size或hidden_size)
            如果return_sequences=False且return_state=False:
                output: 形状 (batch_size, output_size或hidden_size)
            如果return_state=True:
                outputs(或output), final_h, final_c
        """
        batch_size, seq_len, _ = sequence.shape

        # 初始化隐藏和单元状态
        h = np.zeros((self.hidden_size, batch_size))
        c = np.zeros((self.hidden_size, batch_size))

        # 存储每个时间步的输出
        outputs = []

        # 处理序列
        for t in range(seq_len):
            # 获取时间t的输入: (batch_size, input_size)
            x_t = sequence[:, t, :]

            # LSTM前向传播
            h, c = self.cell.forward(x_t, h, c)

            # 如果需要,投影到输出空间
            if self.W_out is not None:
                # h形状: (hidden_size, batch_size)
                # output形状: (output_size, batch_size)
                out_t = self.W_out @ h + self.b_out
            else:
                out_t = h

            # 存储输出: 转置为 (batch_size, output_size或hidden_size)
            outputs.append(out_t.T)

        # 堆叠输出
        if return_sequences:
            # 形状: (batch_size, seq_len, output_size或hidden_size)
            result = np.stack(outputs, axis=1)
        else:
            # 仅返回最后一个输出: (batch_size, output_size或hidden_size)
            result = outputs[-1]

        if return_state:
            # 返回输出和最终状态
            # 将h和c转置回 (batch_size, hidden_size)
            return result, h.T, c.T
        else:
            return result

    def get_params(self):
        """
        获取所有模型参数。

        返回:
            参数名到数组的字典
        """
        params = {
            'W_f': self.cell.W_f, 'U_f': self.cell.U_f, 'b_f': self.cell.b_f,
            'W_i': self.cell.W_i, 'U_i': self.cell.U_i, 'b_i': self.cell.b_i,
            'W_c': self.cell.W_c, 'U_c': self.cell.U_c, 'b_c': self.cell.b_c,
            'W_o': self.cell.W_o, 'U_o': self.cell.U_o, 'b_o': self.cell.b_o,
        }

        if self.W_out is not None:
            params['W_out'] = self.W_out
            params['b_out'] = self.b_out

        return params

    def set_params(self, params):
        """
        设置模型参数。

        参数:
            params: 参数名到数组的字典
        """
        self.cell.W_f = params['W_f']
        self.cell.U_f = params['U_f']
        self.cell.b_f = params['b_f']

        self.cell.W_i = params['W_i']
        self.cell.U_i = params['U_i']
        self.cell.b_i = params['b_i']

        self.cell.W_c = params['W_c']
        self.cell.U_c = params['U_c']
        self.cell.b_c = params['b_c']

        self.cell.W_o = params['W_o']
        self.cell.U_o = params['U_o']
        self.cell.b_o = params['b_o']

        if 'W_out' in params:
            self.W_out = params['W_out']
            self.b_out = params['b_out']


def test_lstm():
    """
    使用随机数据测试LSTM实现。
    验证:
    - 正确的输出形状
    - 无NaN或Inf值
    - 适当的状态演化
    """
    print("="*60)
    print("测试LSTM实现")
    print("="*60)

    # 测试参数
    batch_size = 2
    seq_len = 10
    input_size = 32
    hidden_size = 64
    output_size = 16

    # 创建随机序列
    print(f"\n1. 创建随机序列...")
    print(f"   形状: (batch={batch_size}, seq_len={seq_len}, input_size={input_size})")
    sequence = np.random.randn(batch_size, seq_len, input_size)

    # 测试1: 无输出投影的LSTM
    print(f"\n2. 测试无输出投影的LSTM...")
    lstm_no_proj = LSTM(input_size, hidden_size, output_size=None)

    outputs = lstm_no_proj.forward(sequence, return_sequences=True)
    print(f"   输出形状: {outputs.shape}")
    print(f"   期望: ({batch_size}, {seq_len}, {hidden_size})")
    assert outputs.shape == (batch_size, seq_len, hidden_size), "形状不匹配!"
    assert not np.isnan(outputs).any(), "输出中检测到NaN!"
    assert not np.isinf(outputs).any(), "输出中检测到Inf!"
    print(f"   ✓ 形状正确, 无NaN/Inf")

    # 测试2: 带输出投影的LSTM
    print(f"\n3. 测试带输出投影的LSTM...")
    lstm_with_proj = LSTM(input_size, hidden_size, output_size=output_size)

    outputs = lstm_with_proj.forward(sequence, return_sequences=True)
    print(f"   输出形状: {outputs.shape}")
    print(f"   期望: ({batch_size}, {seq_len}, {output_size})")
    assert outputs.shape == (batch_size, seq_len, output_size), "形状不匹配!"
    assert not np.isnan(outputs).any(), "输出中检测到NaN!"
    assert not np.isinf(outputs).any(), "输出中检测到Inf!"
    print(f"   ✓ 形状正确, 无NaN/Inf")

    # 测试3: 仅返回最后一个输出
    print(f"\n4. 测试return_sequences=False...")
    output_last = lstm_with_proj.forward(sequence, return_sequences=False)
    print(f"   输出形状: {output_last.shape}")
    print(f"   期望: ({batch_size}, {output_size})")
    assert output_last.shape == (batch_size, output_size), "形状不匹配!"
    print(f"   ✓ 形状正确")

    # 测试4: 返回状态
    print(f"\n5. 测试return_state=True...")
    outputs, final_h, final_c = lstm_with_proj.forward(sequence, return_sequences=True, return_state=True)
    print(f"   输出形状: {outputs.shape}")
    print(f"   最终h形状: {final_h.shape}")
    print(f"   最终c形状: {final_c.shape}")
    assert final_h.shape == (batch_size, hidden_size), "隐藏状态形状不匹配!"
    assert final_c.shape == (batch_size, hidden_size), "单元状态形状不匹配!"
    print(f"   ✓ 所有形状正确")

    # 测试5: 验证初始化属性
    print(f"\n6. 验证参数初始化...")
    params = lstm_with_proj.get_params()

    # 检查遗忘门偏置初始化为1.0
    assert np.allclose(params['b_f'], 1.0), "遗忘偏置应初始化为1.0!"
    print(f"   ✓ 遗忘门偏置初始化为1.0")

    # 检查其他偏置为零
    assert np.allclose(params['b_i'], 0.0), "输入偏置应初始化为0.0!"
    assert np.allclose(params['b_c'], 0.0), "单元偏置应初始化为0.0!"
    assert np.allclose(params['b_o'], 0.0), "输出偏置应初始化为0.0!"
    print(f"   ✓ 其他偏置初始化为0.0")

    # 检查循环权重是正交的 (U @ U.T ≈ I)
    U_f = params['U_f']
    ortho_check = U_f @ U_f.T
    identity = np.eye(hidden_size)
    is_orthogonal = np.allclose(ortho_check, identity, atol=1e-5)
    print(f"   ✓ 循环权重是{'正交的' if is_orthogonal else '近似正交的'}")
    print(f"     与单位矩阵的最大偏差: {np.max(np.abs(ortho_check - identity)):.6f}")

    # 测试6: 验证状态演化
    print(f"\n7. 测试状态演化...")
    # 创建带有模式的简单序列
    simple_seq = np.ones((1, 5, input_size)) * 0.1
    outputs_1 = lstm_with_proj.forward(simple_seq, return_sequences=True)

    # 不同的输入应产生不同的输出
    simple_seq_2 = np.ones((1, 5, input_size)) * 0.5
    outputs_2 = lstm_with_proj.forward(simple_seq_2, return_sequences=True)

    assert not np.allclose(outputs_1, outputs_2), "不同的输入应产生不同的输出!"
    print(f"   ✓ 状态随不同输入正确演化")

    # 测试7: 单时间步处理
    print(f"\n8. 测试单时间步...")
    cell = LSTMCell(input_size, hidden_size)
    x = np.random.randn(batch_size, input_size)
    h_prev = np.zeros((hidden_size, batch_size))
    c_prev = np.zeros((hidden_size, batch_size))

    h, c = cell.forward(x, h_prev, c_prev)
    assert h.shape == (hidden_size, batch_size), "隐藏状态形状不匹配!"
    assert c.shape == (hidden_size, batch_size), "单元状态形状不匹配!"
    assert not np.isnan(h).any(), "隐藏状态中有NaN!"
    assert not np.isnan(c).any(), "单元状态中有NaN!"
    print(f"   ✓ 单步处理工作正常")

    # 总结
    print("\n" + "="*60)
    print("所有测试通过! ✓")
    print("="*60)
    print("\nLSTM实现总结:")
    print(f"- 输入大小: {input_size}")
    print(f"- 隐藏大小: {hidden_size}")
    print(f"- 输出大小: {output_size}")
    print(f"- 遗忘偏置初始化为1.0 (有助于长期依赖)")
    print(f"- 循环权重使用正交初始化")
    print(f"- 输入权重使用Xavier初始化")
    print(f"- 前向传播中无NaN/Inf")
    print(f"- 所有输出形状已验证")
    print("="*60)

    return lstm_with_proj


if __name__ == "__main__":
    # 运行测试
    np.random.seed(42)  # 用于可重复性
    model = test_lstm()

    print("\n" + "="*60)
    print("LSTM基准已准备好进行比较!")
    print("="*60)
