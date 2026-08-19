"""Transformer Encoder Block"""
import numpy as np


# ---------- 基础函数 ----------
def softmax_stable(x, axis=-1):
    """数值稳定的 softmax"""
    x_max = np.max(x, axis=axis, keepdims=True)
    x = x - x_max
    np.exp(x, out=x)
    x_sum = np.sum(x, axis=axis, keepdims=True)
    x /= x_sum
    return x


def layer_norm(x, eps=1e-6):
    """不带参数的 LayerNorm"""
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps)


def scaled_dot_product_attention(Q, K, V, mask=None, dropout_p=0.0, rng=None):
    """单头 Scaled Dot-Product Attention，可广播至多头"""
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


# ---------- 多头注意力层 ----------
class MultiHeadAttention:
    def __init__(self, d_model, num_heads, d_k=None, d_v=None, rng=None):
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_k or d_model // num_heads
        self.d_v = d_v or d_model // num_heads
        self.rng = rng or np.random.default_rng()

        def glorot(shape):
            fan_in, fan_out = shape[0], shape[1]
            limit = np.sqrt(6.0 / (fan_in + fan_out))
            return self.rng.uniform(-limit, limit, size=shape).astype(np.float32)

        self.Wq = glorot((d_model, num_heads * self.d_k))
        self.bq = np.zeros((num_heads * self.d_k,), dtype=np.float32)
        self.Wk = glorot((d_model, num_heads * self.d_k))
        self.bk = np.zeros((num_heads * self.d_k,), dtype=np.float32)
        self.Wv = glorot((d_model, num_heads * self.d_v))
        self.bv = np.zeros((num_heads * self.d_v,), dtype=np.float32)
        self.Wo = glorot((num_heads * self.d_v, d_model))
        self.bo = np.zeros((d_model,), dtype=np.float32)

    def __call__(self, X, mask=None, dropout_p=0.0, rng=None):
        B, L, _ = X.shape
        Q = X @ self.Wq + self.bq
        K = X @ self.Wk + self.bk
        V = X @ self.Wv + self.bv

        Q = Q.reshape(B, L, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        K = K.reshape(B, L, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        V = V.reshape(B, L, self.num_heads, self.d_v).transpose(0, 2, 1, 3)

        if mask is not None and mask.ndim == 3:
            mask = mask[:, None, :, :]  # (B,1,L,L)

        out, attn = scaled_dot_product_attention(Q, K, V, mask, dropout_p, rng)
        out = out.transpose(0, 2, 1, 3).reshape(B, L, self.num_heads * self.d_v)
        out = out @ self.Wo + self.bo
        return out, attn


# ---------- 前馈层 ----------
class PositionwiseFeedForward:
    """FFN: Linear(d_model → d_ff) → ReLU → Linear(d_ff → d_model)"""
    def __init__(self, d_model, d_ff, rng=None):
        self.d_model = d_model
        self.d_ff = d_ff
        self.rng = rng or np.random.default_rng()

        def glorot(shape):
            fan_in, fan_out = shape[0], shape[1]
            limit = np.sqrt(6.0 / (fan_in + fan_out))
            return self.rng.uniform(-limit, limit, size=shape).astype(np.float32)

        self.W1 = glorot((d_model, d_ff))
        self.b1 = np.zeros((d_ff,), dtype=np.float32)
        self.W2 = glorot((d_ff, d_model))
        self.b2 = np.zeros((d_model,), dtype=np.float32)

    def __call__(self, X):
        hidden = np.maximum(0, X @ self.W1 + self.b1)  # ReLU
        return hidden @ self.W2 + self.b2


# ---------- Transformer Encoder Block ----------
class TransformerEncoderBlock:
    def __init__(self, d_model=64, num_heads=4, d_ff=256, dropout_p=0.1, rng=None):
        self.rng = rng or np.random.default_rng()
        self.attn = MultiHeadAttention(d_model, num_heads, rng=self.rng)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, rng=self.rng)
        self.dropout_p = dropout_p

    def __call__(self, X, mask=None, training=True):
        rng = self.rng if training else None

        # --- 子层 1：多头注意力 + 残差 + Norm ---
        attn_out, attn_weights = self.attn(X, mask=mask, dropout_p=self.dropout_p if training else 0.0, rng=rng)
        X = X + attn_out  # 残差
        X = layer_norm(X)

        # --- 子层 2：前馈网络 + 残差 + Norm ---
        ffn_out = self.ffn(X)
        X = X + ffn_out  # 残差
        X = layer_norm(X)

        return X, attn_weights


# ---------- 示例 ----------
if __name__ == "__main__":
    rng = np.random.default_rng(42)
    B, L, d_model = 2, 6, 64
    num_heads = 4
    d_ff = 256

    X = rng.normal(size=(B, L, d_model)).astype(np.float32)
    causal_mask = np.triu(np.ones((1, L, L), dtype=bool), k=1)

    encoder = TransformerEncoderBlock(d_model, num_heads, d_ff, dropout_p=0.1, rng=rng)
    out, attn = encoder(X, mask=~causal_mask, training=True)

    print("out shape:", out.shape)     # (2, 6, 64)
    print("attn shape:", attn.shape)   # (2, 4, 6, 6)
    print("注意力行和≈1:", np.allclose(attn.sum(axis=-1), 1.0, atol=1e-6))
