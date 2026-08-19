"""
用numpy实现一个最简易的注意力层
"""
import numpy as np

def softmax_stable(x, axis=-1):
    """数值稳定的 softmax，不依赖其他库。"""
    x_max = np.max(x, axis=axis, keepdims=True)
    x = x - x_max
    np.exp(x, out=x)
    x_sum = np.sum(x, axis=axis, keepdims=True)
    x /= x_sum
    return x

def scaled_dot_product_attention(Q, K, V, mask=None, dropout_p=0.0, rng=None):
    """
    单头 Scaled Dot-Product Attention（核心）
    参数：
        Q: (B, Lq, Dk)
        K: (B, Lk, Dk)
        V: (B, Lk, Dv)
        mask: 可选，形状可广播到 (B, Lq, Lk) 的 0/1 或 布尔矩阵。
              1/False 表示可见，0/True 表示屏蔽。
        dropout_p: 注意力概率上的 dropout 比例（训练时可用），默认 0.0
        rng: np.random.Generator，用于 dropout（可选）
    返回：
        out: (B, Lq, Dv)
        attn: 注意力权重 (B, Lq, Lk)
    """
    B, Lq, Dk = Q.shape
    _, Lk, _ = K.shape

    # (B, Lq, Lk)
    scores = np.matmul(Q, np.swapaxes(K, -1, -2)) / np.sqrt(Dk)

    if mask is not None:
        # 将 mask 标准化为布尔：True 表示需要屏蔽
        # 支持传入 0/1 或 bool，并允许广播到 scores
        mask_bool = ~mask.astype(bool) if mask.dtype != bool else mask
        # 在被屏蔽的位置填充一个很大的负数，避免 -inf 与全被遮挡的溢出
        scores = np.where(mask_bool, -1e9, scores)

    attn = softmax_stable(scores, axis=-1)  # 对最后一个维度 softmax

    # 注意力上的（可选）dropout：训练时才用；推理时应保持 0
    if dropout_p and dropout_p > 0.0:
        if rng is None:
            rng = np.random.default_rng()
        keep = rng.random(attn.shape) >= dropout_p
        attn = attn * keep / (1.0 - dropout_p)

    out = np.matmul(attn, V)  # (B, Lq, Dv)
    return out, attn

class SingleHeadAttention:
    """
    最小可用的单头注意力层：包含线性投影（Wq/Wk/Wv）与输出投影（Wo）。
    用法：
        mha = SingleHeadAttention(d_model=64, d_k=64, d_v=64, rng=np.random.default_rng(0))
        out, attn = mha(X, mask=causal_mask)
    """
    def __init__(self, d_model, d_k, d_v, rng=None):
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.rng = rng or np.random.default_rng()

        # 简单的 Xavier/Glorot 初始化
        def glorot(shape):
            fan_in, fan_out = shape[0], shape[1]
            limit = np.sqrt(6.0 / (fan_in + fan_out))
            return self.rng.uniform(-limit, limit, size=shape).astype(np.float32)

        self.Wq = glorot((d_model, d_k))
        self.bq = np.zeros((d_k,), dtype=np.float32)
        self.Wk = glorot((d_model, d_k))
        self.bk = np.zeros((d_k,), dtype=np.float32)
        self.Wv = glorot((d_model, d_v))
        self.bv = np.zeros((d_v,), dtype=np.float32)
        self.Wo = glorot((d_v, d_model))
        self.bo = np.zeros((d_model,), dtype=np.float32)

    def __call__(self, X, mask=None, dropout_p=0.0, rng=None):
        """
        X: (B, L, d_model)
        mask: 可选，形状可广播到 (B, L, L) 的 0/1 或 bool 矩阵
        返回：
            out: (B, L, d_model)
            attn: (B, L, L)
        """
        B, L, D = X.shape
        # 线性投影到 Q/K/V
        Q = X @ self.Wq + self.bq    # (B, L, d_k)
        K = X @ self.Wk + self.bk    # (B, L, d_k)
        V = X @ self.Wv + self.bv    # (B, L, d_v)

        out, attn = scaled_dot_product_attention(Q, K, V, mask=mask, dropout_p=dropout_p, rng=rng)
        # 输出投影
        out = out @ self.Wo + self.bo  # (B, L, d_model)
        return out, attn

# ---------- 示例用法 ----------
if __name__ == "__main__":
    rng = np.random.default_rng(42)
    B, L, d_model = 2, 5, 32
    d_k = d_v = 32

    X = rng.normal(size=(B, L, d_model)).astype(np.float32)

    # 构造一个因果掩码（自回归）：上三角（不含对角线）为 True/0 表示屏蔽未来
    causal = np.triu(np.ones((1, L, L), dtype=bool), k=1)  # 形状可广播到 (B, L, L)

    attn_layer = SingleHeadAttention(d_model, d_k, d_v, rng=rng)
    out, attn = attn_layer(X, mask=~causal)  # 这里传入的是“可见”掩码 => 用 ~ 转换

    print("out shape:", out.shape)   # (2, 5, 32)
    print("attn shape:", attn.shape) # (2, 5, 5)
    # 验证每一行注意力权重之和为 1
    print("row sums (should be 1):", np.allclose(attn.sum(axis=-1), 1.0, atol=1e-6))
