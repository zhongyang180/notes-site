import numpy as np

def scaled_dot_product_attention(Q, K, V, mask=None, eps=1e-9):
    """
    单头 scaled dot-product attention（仅 NumPy）

    参数
    ----
    Q: (B, T_q, D) 查询
    K: (B, T_k, D) 键
    V: (B, T_k, D_v) 值
    mask: 与得分矩阵相同的形状 (B, T_q, T_k) 的0/1掩码；
          1=允许关注，0=屏蔽。若为 None 则不屏蔽。
    eps: 用于数值稳定的极小值

    返回
    ----
    out: (B, T_q, D_v) 注意力输出
    attn: (B, T_q, T_k) 注意力权重（softmax 后）
    """
    B, T_q, D = Q.shape
    _, T_k, Dk = K.shape
    assert D == Dk, "Q 和 K 的最后一维必须相同"

    # 1) 原始打分：QK^T / sqrt(D)
    # scores: (B, T_q, T_k)
    scale = 1.0 / np.sqrt(D)
    scores = np.matmul(Q, np.swapaxes(K, -1, -2)) * scale  # K^T

    # 2) 应用 mask（将被屏蔽位置设成 -inf）
    if mask is not None:
        # 期望 mask 为 0/1，形状可广播或等同于 scores
        # 先广播到 scores 形状
        mask_b = np.broadcast_to(mask, scores.shape).astype(scores.dtype)
        scores = scores + (1.0 - mask_b) * (-1e9)

    # 3) 稳定 softmax：先减去行内最大值
    # 防止 exp 溢出
    scores_max = np.max(scores, axis=-1, keepdims=True)
    exp_scores = np.exp(scores - scores_max)
    # 若整行都被 mask 掉，exp_scores 可能全 0；加上 eps 防止除 0
    attn = exp_scores / (np.sum(exp_scores, axis=-1, keepdims=True) + eps)

    # 4) 加权求和
    # out: (B, T_q, D_v)
    out = np.matmul(attn, V)
    return out, attn

# ------ 常用掩码辅助函数 ------

def make_padding_mask(valid_lengths, T_k):
    """
    基于每个样本的有效长度构造 padding mask（1=有效，0=padding）
    返回形状：(B, 1, T_k)，可广播到 (B, T_q, T_k)
    """
    B = len(valid_lengths)
    idx = np.arange(T_k)[None, :]  # (1, T_k)
    lens = np.asarray(valid_lengths)[:, None]  # (B, 1)
    mask = (idx < lens).astype(np.float32)  # (B, T_k)
    return mask[:, None, :]  # (B, 1, T_k)

def make_causal_mask(T_q, T_k=None):
    """
    下三角因果掩码（decoder 自回归用），1=允许，0=禁止关注未来
    返回形状：(1, T_q, T_k)
    """
    if T_k is None:
        T_k = T_q
    i = np.arange(T_q)[:, None]
    j = np.arange(T_k)[None, :]
    mask = (j <= i).astype(np.float32)  # (T_q, T_k)
    return mask[None, :, :]  # (1, T_q, T_k)

# ------ 简单示例与自测 ------

if __name__ == "__main__":
    np.random.seed(0)

    B, T_q, T_k, D, D_v = 2, 4, 5, 8, 8
    Q = np.random.randn(B, T_q, D).astype(np.float32)
    K = np.random.randn(B, T_k, D).astype(np.float32)
    V = np.random.randn(B, T_k, D_v).astype(np.float32)

    # 例1：无掩码
    out, attn = scaled_dot_product_attention(Q, K, V, mask=None)
    print("out(no mask):", out.shape, " attn:", attn.shape)

    # 例2：padding mask（假设第二个 batch 只有前3个token有效）
    pad_mask = make_padding_mask(valid_lengths=[5, 3], T_k=T_k)  # (B,1,T_k)
    # 广播到 (B, T_q, T_k) 由函数内部完成，这里可直接传入
    out_pad, attn_pad = scaled_dot_product_attention(Q, K, V, mask=pad_mask)
    print("out(pad mask):", out_pad.shape)

    # 例3：causal mask（再叠加 padding）
    causal = make_causal_mask(T_q, T_k)        # (1,T_q,T_k)
    both = causal * pad_mask                   # 广播得到 (B,T_q,T_k)
    out_both, attn_both = scaled_dot_product_attention(Q, K, V, mask=both)
    print("out(causal+pad):", out_both.shape)
