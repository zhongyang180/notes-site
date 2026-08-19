"""multihead attention"""
import numpy as np

def softmax_stable(x, axis=-1):
    """数值稳定的 softmax"""
    x_max = np.max(x, axis=axis, keepdims=True)
    x = x - x_max
    np.exp(x, out=x)
    x_sum = np.sum(x, axis=axis, keepdims=True)
    x /= x_sum
    return x

def scaled_dot_product_attention(Q, K, V, mask=None, dropout_p=0.0, rng=None):
    """
    单头注意力算子，可广播至多头场景。
    Q, K, V: (..., Lq, Dk/Dv)
    mask: 可广播到 (..., Lq, Lk)
    """
    d_k = Q.shape[-1]
    scores = np.matmul(Q, np.swapaxes(K, -1, -2)) / np.sqrt(d_k)

    if mask is not None:
        mask_bool = ~mask.astype(bool) if mask.dtype != bool else mask
        scores = np.where(mask_bool, -1e9, scores)

    attn = softmax_stable(scores, axis=-1)

    if dropout_p and dropout_p > 0.0:
        if rng is None:
            rng = np.random.default_rng()
        keep = rng.random(attn.shape) >= dropout_p
        attn = attn * keep / (1.0 - dropout_p)

    out = np.matmul(attn, V)
    return out, attn

class MultiHeadAttention:
    """
    多头注意力层（NumPy实现）
    结构：输入线性投影 → 拆分多头 → Scaled Dot-Product Attention → 拼接 → 输出投影
    """
    def __init__(self, d_model, num_heads, d_k=None, d_v=None, rng=None):
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_k or d_model // num_heads
        self.d_v = d_v or d_model // num_heads
        self.rng = rng or np.random.default_rng()

        def glorot(shape):
            fan_in, fan_out = shape[0], shape[1]
            limit = np.sqrt(6.0 / (fan_in + fan_out))
            return self.rng.uniform(-limit, limit, size=shape).astype(np.float32)

        # 输入线性层参数
        self.Wq = glorot((d_model, num_heads * self.d_k))
        self.bq = np.zeros((num_heads * self.d_k,), dtype=np.float32)
        self.Wk = glorot((d_model, num_heads * self.d_k))
        self.bk = np.zeros((num_heads * self.d_k,), dtype=np.float32)
        self.Wv = glorot((d_model, num_heads * self.d_v))
        self.bv = np.zeros((num_heads * self.d_v,), dtype=np.float32)

        # 输出投影
        self.Wo = glorot((num_heads * self.d_v, d_model))
        self.bo = np.zeros((d_model,), dtype=np.float32)

    def __call__(self, X, mask=None, dropout_p=0.0, rng=None):
        """
        X: (B, L, d_model)
        mask: 可广播到 (B, num_heads, L, L)
        返回:
            out: (B, L, d_model)
            attn: (B, num_heads, L, L)
        """
        B, L, _ = X.shape

        # --- Step 1: 线性投影 ---
        Q = X @ self.Wq + self.bq  # (B, L, num_heads*d_k)
        K = X @ self.Wk + self.bk
        V = X @ self.Wv + self.bv

        # --- Step 2: 拆分多头 ---
        Q = Q.reshape(B, L, self.num_heads, self.d_k).transpose(0, 2, 1, 3)  # (B, H, L, d_k)
        K = K.reshape(B, L, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        V = V.reshape(B, L, self.num_heads, self.d_v).transpose(0, 2, 1, 3)

        # --- Step 3: 注意力计算 ---
        if mask is not None:
            # mask 形状 (B, L, L) → (B, 1, L, L) 以便广播到多头
            if mask.ndim == 3:
                mask = mask[:, None, :, :]
        out, attn = scaled_dot_product_attention(Q, K, V, mask=mask, dropout_p=dropout_p, rng=rng)
        # out: (B, H, L, d_v), attn: (B, H, L, L)

        # --- Step 4: 拼接多头 ---
        out = out.transpose(0, 2, 1, 3).reshape(B, L, self.num_heads * self.d_v)

        # --- Step 5: 输出线性层 ---
        out = out @ self.Wo + self.bo  # (B, L, d_model)
        return out, attn


# ---------- 示例用法 ----------
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    B, L, d_model, num_heads = 2, 6, 64, 4

    X = rng.normal(size=(B, L, d_model)).astype(np.float32)
    causal_mask = np.triu(np.ones((1, L, L), dtype=bool), k=1)  # 因果mask

    mha = MultiHeadAttention(d_model, num_heads, rng=rng)
    out, attn = mha(X, mask=~causal_mask, dropout_p=0.1, rng=rng)

    print("out shape:", out.shape)     # (2, 6, 64)
    print("attn shape:", attn.shape)   # (2, 4, 6, 6)
    print("每头注意力行和≈1:", np.allclose(attn.sum(axis=-1), 1.0, atol=1e-6))
