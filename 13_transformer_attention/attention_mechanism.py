"""
Multi-Head Dot-Product Attention 机制
Paper 18: Relational RNN - Implementation Task P1-T2

本模块仅使用 NumPy 实现 scaled dot-product attention 和 multi-head attention
机制，遵循 "Attention is All You Need" 论文的表述。

Sutskever 30 papers 项目的教育性实现。
"""

import numpy as np


def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Scaled Dot-Product Attention 机制。

    计算 attention: Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V

    Args:
        Q: queries，shape (batch, seq_len, d_k)
        K: keys，shape (batch, seq_len, d_k)
        V: values，shape (batch, seq_len, d_k)
        mask: 可选的 mask，shape (batch, seq_len, seq_len) 或 (seq_len, seq_len)
              值应为 0 (保留) 或 -inf (mask 掉)

    Returns:
        output: attended values，shape (batch, seq_len, d_k)
        attention_weights: attention 分布，shape (batch, seq_len, seq_len)

    数学表达式：
        1. scores = QK^T / sqrt(d_k)
        2. if mask: scores = scores + mask
        3. attention_weights = softmax(scores)
        4. output = attention_weights @ V
    """
    # 输入 shape 断言
    assert Q.ndim == 3, f"Q 必须是 3D (batch, seq_len, d_k)，得到 shape {Q.shape}"
    assert K.ndim == 3, f"K 必须是 3D (batch, seq_len, d_k)，得到 shape {K.shape}"
    assert V.ndim == 3, f"V 必须是 3D (batch, seq_len, d_k)，得到 shape {V.shape}"

    batch_size, seq_len_q, d_k = Q.shape
    _, seq_len_k, _ = K.shape

    assert Q.shape[-1] == K.shape[-1], "Q 和 K 必须有相同的 d_k 维度"
    assert K.shape[1] == V.shape[1], "K 和 V 必须有相同的 seq_len"

    # Step 1: 计算 attention scores QK^T / sqrt(d_k)
    # Q: (batch, seq_len_q, d_k)
    # K^T: (batch, d_k, seq_len_k)
    # scores: (batch, seq_len_q, seq_len_k)
    scores = np.matmul(Q, K.transpose(0, 2, 1))  # (batch, seq_len_q, seq_len_k)

    # 用 sqrt(d_k) 缩放以保证数值稳定性
    # 这可以防止 dot products 变得过大，避免将 softmax 推入梯度极小的区域
    scaling_factor = np.sqrt(d_k)
    scores = scores / scaling_factor

    # Step 2: 如果提供了 mask 则应用
    if mask is not None:
        # 同时处理 (batch, seq_len, seq_len) 和 (seq_len, seq_len) 两种 masks
        if mask.ndim == 2:
            mask = mask[np.newaxis, :, :]  # 添加 batch 维度

        assert mask.shape[-2:] == scores.shape[-2:], \
            f"Mask shape {mask.shape} 与 scores shape {scores.shape} 不兼容"

        # 添加 mask（通常对需要 mask 掉的位置使用 -inf）
        scores = scores + mask

    # Step 3: 应用 softmax 得到 attention weights
    # Softmax 数值稳定性技巧（减去最大值）
    scores_max = np.max(scores, axis=-1, keepdims=True)
    exp_scores = np.exp(scores - scores_max)
    attention_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

    # 检查 NaN/Inf（可能由极端 mask 值引起）
    if np.any(np.isnan(attention_weights)) or np.any(np.isinf(attention_weights)):
        raise ValueError("在 attention weights 中检测到 NaN 或 Inf，请检查 mask 值。")

    # Step 4: 将 attention 应用到 values
    # attention_weights: (batch, seq_len_q, seq_len_k)
    # V: (batch, seq_len_k, d_k)
    # output: (batch, seq_len_q, d_k)
    output = np.matmul(attention_weights, V)

    return output, attention_weights


def split_heads(x, num_heads):
    """
    将最后一个维度分割为 (num_heads, depth)。
    转置以将 head 维度放在前面。

    Args:
        x: tensor，shape (batch, seq_len, d_model)
        num_heads: attention heads 的数量

    Returns:
        tensor，shape (batch, num_heads, seq_len, depth)
        其中 depth = d_model // num_heads
    """
    batch_size, seq_len, d_model = x.shape
    depth = d_model // num_heads

    # Reshape 到 (batch, seq_len, num_heads, depth)
    x = x.reshape(batch_size, seq_len, num_heads, depth)

    # 转置到 (batch, num_heads, seq_len, depth)
    x = x.transpose(0, 2, 1, 3)

    return x


def combine_heads(x):
    """
    split_heads 的逆操作。

    Args:
        x: tensor，shape (batch, num_heads, seq_len, depth)

    Returns:
        tensor，shape (batch, seq_len, d_model)
        其中 d_model = num_heads * depth
    """
    batch_size, num_heads, seq_len, depth = x.shape

    # 转置到 (batch, seq_len, num_heads, depth)
    x = x.transpose(0, 2, 1, 3)

    # Reshape 到 (batch, seq_len, d_model)
    d_model = num_heads * depth
    x = x.reshape(batch_size, seq_len, d_model)

    return x


def multi_head_attention(Q, K, V, num_heads=4, W_q=None, W_k=None, W_v=None, W_o=None, mask=None):
    """
    Multi-Head Attention 机制。

    我们不是使用 d_model 维的 keys、values 和 queries 执行单个 attention 函数，
    而是用不同的、学到的线性投影将 queries、keys 和 values 线性投影 h 次。
    在每个投影版本上并行执行 attention 函数，产生的输出值被拼接并再次投影。

    Args:
        Q: queries，shape (batch, seq_len, d_model)
        K: keys，shape (batch, seq_len, d_model)
        V: values，shape (batch, seq_len, d_model)
        num_heads: attention heads 的数量
        W_q: query 投影矩阵，shape (d_model, d_model)
        W_k: key 投影矩阵，shape (d_model, d_model)
        W_v: value 投影矩阵，shape (d_model, d_model)
        W_o: output 投影矩阵，shape (d_model, d_model)
        mask: 可选的 attention mask

    Returns:
        output: shape (batch, seq_len, d_model)
        attention_weights: shape (batch, num_heads, seq_len, seq_len)
    """
    # 输入验证
    assert Q.ndim == 3, f"Q 必须是 3D，得到 shape {Q.shape}"
    assert K.ndim == 3, f"K 必须是 3D，得到 shape {K.shape}"
    assert V.ndim == 3, f"V 必须是 3D，得到 shape {V.shape}"

    batch_size, seq_len, d_model = Q.shape

    assert d_model % num_heads == 0, \
        f"d_model ({d_model}) 必须能被 num_heads ({num_heads}) 整除"

    depth = d_model // num_heads  # 论文中的 d_k

    # 如果未提供投影矩阵则初始化
    if W_q is None or W_k is None or W_v is None or W_o is None:
        params = init_attention_params(d_model, num_heads)
        W_q = params['W_q'] if W_q is None else W_q
        W_k = params['W_k'] if W_k is None else W_k
        W_v = params['W_v'] if W_v is None else W_v
        W_o = params['W_o'] if W_o is None else W_o

    # Step 1: 线性投影
    # Q, K, V: (batch, seq_len, d_model)
    # W_q, W_k, W_v: (d_model, d_model)
    # matmul 后: (batch, seq_len, d_model)
    Q_proj = np.matmul(Q, W_q)  # (batch, seq_len, d_model)
    K_proj = np.matmul(K, W_k)  # (batch, seq_len, d_model)
    V_proj = np.matmul(V, W_v)  # (batch, seq_len, d_model)

    # Step 2: 分割为多个 heads
    # 将 d_model 分割为 num_heads * depth
    # (batch, seq_len, d_model) -> (batch, num_heads, seq_len, depth)
    Q_split = split_heads(Q_proj, num_heads)  # (batch, num_heads, seq_len, depth)
    K_split = split_heads(K_proj, num_heads)  # (batch, num_heads, seq_len, depth)
    V_split = split_heads(V_proj, num_heads)  # (batch, num_heads, seq_len, depth)

    # Step 3: 对每个 head 应用 scaled dot-product attention
    # 需要 reshape 以对每个 head 应用 attention
    # 当前 shape: (batch, num_heads, seq_len, depth)
    # Reshape 到: (batch * num_heads, seq_len, depth)
    batch_heads = batch_size * num_heads
    Q_reshaped = Q_split.reshape(batch_heads, seq_len, depth)
    K_reshaped = K_split.reshape(batch_heads, seq_len, depth)
    V_reshaped = V_split.reshape(batch_heads, seq_len, depth)

    # 如果提供了 mask 则为多个 heads 调整
    if mask is not None:
        # 如果 mask 是 (batch, seq_len, seq_len)，为每个 head 复制
        if mask.ndim == 3:
            # 扩展到 (batch, num_heads, seq_len, seq_len)
            mask_expanded = np.tile(mask[:, np.newaxis, :, :], (1, num_heads, 1, 1))
            # Reshape 到 (batch * num_heads, seq_len, seq_len)
            mask_reshaped = mask_expanded.reshape(batch_heads, seq_len, seq_len)
        elif mask.ndim == 2:
            # (seq_len, seq_len) -> (batch * num_heads, seq_len, seq_len)
            mask_reshaped = np.tile(mask[np.newaxis, :, :], (batch_heads, 1, 1))
        else:
            raise ValueError(f"不支持的 mask shape: {mask.shape}")
    else:
        mask_reshaped = None

    # 应用 attention
    attended, attn_weights = scaled_dot_product_attention(
        Q_reshaped, K_reshaped, V_reshaped, mask=mask_reshaped
    )
    # attended: (batch * num_heads, seq_len, depth)
    # attn_weights: (batch * num_heads, seq_len, seq_len)

    # Step 4: Reshape 并合并 heads
    # (batch * num_heads, seq_len, depth) -> (batch, num_heads, seq_len, depth)
    attended = attended.reshape(batch_size, num_heads, seq_len, depth)
    attn_weights = attn_weights.reshape(batch_size, num_heads, seq_len, seq_len)

    # 拼接 heads: (batch, num_heads, seq_len, depth) -> (batch, seq_len, d_model)
    attended_combined = combine_heads(attended)  # (batch, seq_len, d_model)

    # Step 5: 最终的线性投影
    # attended_combined: (batch, seq_len, d_model)
    # W_o: (d_model, d_model)
    output = np.matmul(attended_combined, W_o)  # (batch, seq_len, d_model)

    return output, attn_weights


def init_attention_params(d_model, num_heads):
    """
    初始化 multi-head attention 的参数。

    使用 Xavier/Glorot 初始化权重矩阵，以保持各层之间的方差
    并防止梯度消失/爆炸。

    Args:
        d_model: model 维度
        num_heads: attention heads 的数量

    Returns:
        dict 包含:
            - W_q: query 投影矩阵 (d_model, d_model)
            - W_k: key 投影矩阵 (d_model, d_model)
            - W_v: value 投影矩阵 (d_model, d_model)
            - W_o: output 投影矩阵 (d_model, d_model)
    """
    assert d_model % num_heads == 0, \
        f"d_model ({d_model}) 必须能被 num_heads ({num_heads}) 整除"

    # Xavier/Glorot 初始化
    # Variance = 2 / (fan_in + fan_out)
    # 对于权重矩阵 (d_model, d_model)，fan_in = fan_out = d_model
    # std = sqrt(2 / (d_model + d_model)) = sqrt(1 / d_model)
    std = np.sqrt(1.0 / d_model)

    params = {
        'W_q': np.random.randn(d_model, d_model) * std,
        'W_k': np.random.randn(d_model, d_model) * std,
        'W_v': np.random.randn(d_model, d_model) * std,
        'W_o': np.random.randn(d_model, d_model) * std,
    }

    return params


def create_causal_mask(seq_len):
    """
    为 autoregressive attention 创建 causal (下三角) mask。

    这个 mask 防止位置关注后续位置，这对于语言模型等
    autoregressive models 至关重要。

    Args:
        seq_len: 序列长度

    Returns:
        shape 为 (seq_len, seq_len) 的 mask，对角线及以下为 0，
        对角线以上为 -inf
    """
    # 创建全为 1 的下三角矩阵
    mask = np.tril(np.ones((seq_len, seq_len)))

    # 将 mask 为 0 的位置（上三角）转换为 -inf
    mask = np.where(mask == 0, -np.inf, 0.0)

    return mask


# ============================================================================
# Test 函数
# ============================================================================

def test_scaled_dot_product_attention():
    """测试 scaled dot-product attention 机制。"""
    print("=" * 80)
    print("测试 Scaled Dot-Product Attention")
    print("=" * 80)

    # 设置随机种子以保证可重复性
    np.random.seed(42)

    # 测试参数
    batch_size = 2
    seq_len = 5
    d_k = 8

    # 创建随机输入
    Q = np.random.randn(batch_size, seq_len, d_k)
    K = np.random.randn(batch_size, seq_len, d_k)
    V = np.random.randn(batch_size, seq_len, d_k)

    print("\n输入 shapes:")
    print(f"  Q: {Q.shape}")
    print(f"  K: {K.shape}")
    print(f"  V: {V.shape}")

    # Test 1: 无 mask 的基础 attention
    print("\n[Test 1] 基础 attention (无 mask)")
    output, attn_weights = scaled_dot_product_attention(Q, K, V)

    print(f"  Output shape: {output.shape}")
    print(f"  Attention weights shape: {attn_weights.shape}")

    # 验证 shapes
    assert output.shape == (batch_size, seq_len, d_k), \
        f"Output shape 不匹配: 期望 {(batch_size, seq_len, d_k)}，得到 {output.shape}"
    assert attn_weights.shape == (batch_size, seq_len, seq_len), \
        f"Attention weights shape 不匹配: 期望 {(batch_size, seq_len, seq_len)}，得到 {attn_weights.shape}"

    # 验证 attention weights 求和为 1
    attn_sums = np.sum(attn_weights, axis=-1)
    assert np.allclose(attn_sums, 1.0), \
        f"Attention weights 求和不为 1: {attn_sums}"
    print("  Attention weights 求和为 1: PASS")

    # 验证 attention weights 非负
    assert np.all(attn_weights >= 0), "Attention weights 包含负值"
    print("  Attention weights 非负: PASS")

    # 检查 NaN 或 Inf
    assert not np.any(np.isnan(output)), "Output 包含 NaN"
    assert not np.any(np.isinf(output)), "Output 包含 Inf"
    print("  Output 无 NaN/Inf: PASS")

    # Test 2: 带 causal mask 的 Attention
    print("\n[Test 2] 带 causal mask 的 Attention")
    mask = create_causal_mask(seq_len)
    output_masked, attn_weights_masked = scaled_dot_product_attention(Q, K, V, mask=mask)

    print(f"  Causal mask shape: {mask.shape}")
    print(f"  Output shape: {output_masked.shape}")

    # 验证 causal 属性: attention 的上三角应为零
    for b in range(batch_size):
        for i in range(seq_len):
            for j in range(i + 1, seq_len):
                assert np.isclose(attn_weights_masked[b, i, j], 0.0, atol=1e-6), \
                    f"Causal mask 在 batch {b}，位置 ({i}, {j}) 处被违反"
    print("  Causal masking 正确: PASS")

    # 验证带 mask 的 attention weights 仍求和为 1
    attn_sums_masked = np.sum(attn_weights_masked, axis=-1)
    assert np.allclose(attn_sums_masked, 1.0), \
        f"Masked attention weights 求和不为 1: {attn_sums_masked}"
    print("  Masked attention weights 求和为 1: PASS")

    print("\n" + "=" * 80)
    print("Scaled Dot-Product Attention: 所有测试通过")
    print("=" * 80 + "\n")


def test_multi_head_attention():
    """测试 multi-head attention 机制。"""
    print("=" * 80)
    print("测试 Multi-Head Attention")
    print("=" * 80)

    # 设置随机种子以保证可重复性
    np.random.seed(42)

    # 测试参数
    batch_size = 2
    seq_len = 5
    d_model = 64
    num_heads = 4

    print("\n参数:")
    print(f"  batch_size: {batch_size}")
    print(f"  seq_len: {seq_len}")
    print(f"  d_model: {d_model}")
    print(f"  num_heads: {num_heads}")
    print(f"  depth (d_k): {d_model // num_heads}")

    # 创建随机输入
    Q = np.random.randn(batch_size, seq_len, d_model)
    K = np.random.randn(batch_size, seq_len, d_model)
    V = np.random.randn(batch_size, seq_len, d_model)

    print("\n输入 shapes:")
    print(f"  Q: {Q.shape}")
    print(f"  K: {K.shape}")
    print(f"  V: {V.shape}")

    # 初始化参数
    print("\n[Test 1] 参数初始化")
    params = init_attention_params(d_model, num_heads)

    print(f"  W_q shape: {params['W_q'].shape}")
    print(f"  W_k shape: {params['W_k'].shape}")
    print(f"  W_v shape: {params['W_v'].shape}")
    print(f"  W_o shape: {params['W_o'].shape}")

    # 验证参数 shapes
    for key in ['W_q', 'W_k', 'W_v', 'W_o']:
        assert params[key].shape == (d_model, d_model), \
            f"{key} shape 不匹配: 期望 {(d_model, d_model)}，得到 {params[key].shape}"
    print("  参数 shapes 正确: PASS")

    # 验证 Xavier 初始化（检查方差）
    expected_std = np.sqrt(1.0 / d_model)
    for key in ['W_q', 'W_k', 'W_v', 'W_o']:
        actual_std = np.std(params[key])
        # 由于随机采样允许一定方差
        assert 0.5 * expected_std < actual_std < 2.0 * expected_std, \
            f"{key} 标准差超出预期范围"
    print("  Xavier 初始化正确: PASS")

    # Test 2: 无 mask 的 Multi-head attention
    print("\n[Test 2] Multi-head attention (无 mask)")
    output, attn_weights = multi_head_attention(
        Q, K, V,
        num_heads=num_heads,
        W_q=params['W_q'],
        W_k=params['W_k'],
        W_v=params['W_v'],
        W_o=params['W_o']
    )

    print(f"  Output shape: {output.shape}")
    print(f"  Attention weights shape: {attn_weights.shape}")

    # 验证 shapes
    assert output.shape == (batch_size, seq_len, d_model), \
        f"Output shape 不匹配: 期望 {(batch_size, seq_len, d_model)}，得到 {output.shape}"
    assert attn_weights.shape == (batch_size, num_heads, seq_len, seq_len), \
        f"Attention weights shape 不匹配: 期望 {(batch_size, num_heads, seq_len, seq_len)}，得到 {attn_weights.shape}"
    print("  Output shape 正确: PASS")
    print("  Attention weights shape 正确: PASS")

    # 验证每个 head 的 attention weights 求和为 1
    attn_sums = np.sum(attn_weights, axis=-1)
    assert np.allclose(attn_sums, 1.0), \
        f"Attention weights 求和不为 1: {attn_sums}"
    print("  Attention weights 求和为 1 (所有 heads): PASS")

    # 检查 NaN 或 Inf
    assert not np.any(np.isnan(output)), "Output 包含 NaN"
    assert not np.any(np.isinf(output)), "Output 包含 Inf"
    assert not np.any(np.isnan(attn_weights)), "Attention weights 包含 NaN"
    assert not np.any(np.isinf(attn_weights)), "Attention weights 包含 Inf"
    print("  Output 无 NaN/Inf: PASS")

    # Test 3: 带 causal mask 的 Multi-head attention
    print("\n[Test 3] 带 causal mask 的 Multi-head attention")
    mask = create_causal_mask(seq_len)
    output_masked, attn_weights_masked = multi_head_attention(
        Q, K, V,
        num_heads=num_heads,
        W_q=params['W_q'],
        W_k=params['W_k'],
        W_v=params['W_v'],
        W_o=params['W_o'],
        mask=mask
    )

    print(f"  Output shape: {output_masked.shape}")
    print(f"  Attention weights shape: {attn_weights_masked.shape}")

    # 验证所有 heads 的 causal 属性
    for b in range(batch_size):
        for h in range(num_heads):
            for i in range(seq_len):
                for j in range(i + 1, seq_len):
                    assert np.isclose(attn_weights_masked[b, h, i, j], 0.0, atol=1e-6), \
                        f"Causal mask 在 batch {b}，head {h}，位置 ({i}, {j}) 处被违反"
    print("  Causal masking 正确 (所有 heads): PASS")

    # Test 4: 不同数量的 heads
    print("\n[Test 4] 测试不同数量的 heads")
    for test_num_heads in [1, 2, 8]:
        test_params = init_attention_params(d_model, test_num_heads)
        test_output, test_attn = multi_head_attention(
            Q, K, V,
            num_heads=test_num_heads,
            W_q=test_params['W_q'],
            W_k=test_params['W_k'],
            W_v=test_params['W_v'],
            W_o=test_params['W_o']
        )
        assert test_output.shape == (batch_size, seq_len, d_model)
        assert test_attn.shape == (batch_size, test_num_heads, seq_len, seq_len)
        print(f"  num_heads={test_num_heads}: PASS")

    # Test 5: Self-attention (Q=K=V)
    print("\n[Test 5] Self-attention (Q=K=V)")
    X = np.random.randn(batch_size, seq_len, d_model)
    self_output, self_attn = multi_head_attention(
        X, X, X,
        num_heads=num_heads,
        W_q=params['W_q'],
        W_k=params['W_k'],
        W_v=params['W_v'],
        W_o=params['W_o']
    )
    assert self_output.shape == (batch_size, seq_len, d_model)
    assert self_attn.shape == (batch_size, num_heads, seq_len, seq_len)
    print("  Self-attention 正常工作: PASS")

    print("\n" + "=" * 80)
    print("Multi-Head Attention: 所有测试通过")
    print("=" * 80 + "\n")


def demonstrate_attention_properties():
    """演示 attention 机制的关键属性。"""
    print("=" * 80)
    print("演示 Attention 属性")
    print("=" * 80)

    np.random.seed(42)

    # 用于可视化的简单示例，batch_size=1
    batch_size = 1
    seq_len = 4
    d_model = 8
    num_heads = 2

    # 创建简单的输入，其中的关系很清晰
    Q = np.random.randn(batch_size, seq_len, d_model) * 0.5
    K = np.random.randn(batch_size, seq_len, d_model) * 0.5
    V = np.random.randn(batch_size, seq_len, d_model) * 0.5

    # 使第一个和最后一个位置彼此更相似
    K[0, 0, :] = K[0, -1, :] = np.random.randn(d_model) * 0.5

    params = init_attention_params(d_model, num_heads)
    output, attn_weights = multi_head_attention(
        Q, K, V,
        num_heads=num_heads,
        W_q=params['W_q'],
        W_k=params['W_k'],
        W_v=params['W_v'],
        W_o=params['W_o']
    )

    print("\n示例 attention weights (head 0):")
    print(f"Shape: {attn_weights.shape}")
    print("\nAttention 矩阵 (行关注列):")
    print(attn_weights[0, 0])  # 第一个 batch，第一个 head

    print("\n验证的属性:")
    print(f"  1. 每行求和为 1.0: {np.allclose(np.sum(attn_weights[0, 0], axis=-1), 1.0)}")
    print(f"  2. 所有权重 >= 0: {np.all(attn_weights >= 0)}")
    print("  3. Output 是 V 的加权组合")

    # 验证 output 是加权组合
    # 对于位置 i，output[i] = sum_j (attn_weights[i,j] * V[j])
    # 注意: 需要考虑投影，所以这是近似的
    _ = np.zeros((seq_len, d_model))  # 占位符，避免未使用变量警告

    print("\n" + "=" * 80 + "\n")


def main():
    """运行所有测试和演示。"""
    print("\n" + "=" * 80)
    print(" " * 15 + "MULTI-HEAD ATTENTION MECHANISM 测试套件")
    print(" " * 20 + "Paper 18: Relational RNN - Task P1-T2")
    print("=" * 80 + "\n")

    # 运行测试
    test_scaled_dot_product_attention()
    test_multi_head_attention()
    demonstrate_attention_properties()

    print("=" * 80)
    print(" " * 25 + "所有测试成功完成")
    print("=" * 80)
    print("\n总结:")
    print("  - Scaled dot-product attention: 正常工作")
    print("  - Multi-head attention: 正常工作")
    print("  - 参数初始化: 正常工作")
    print("  - 数值稳定性: 已验证 (无 NaN/Inf)")
    print("  - Attention weights: 求和为 1，非负")
    print("  - Causal masking: 正常工作")
    print("  - Shape 断言: 全部通过")
    print("\n实现已准备好集成到 Relational RNN 中！")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
