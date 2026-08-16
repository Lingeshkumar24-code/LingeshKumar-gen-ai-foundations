"""PDF Task 1: vectorized scaled dot-product multi-head causal attention."""
import numpy as np

def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)

def causal_attention(q, k, v):
    # q,k,v: [batch, heads, seq, head_dim]
    d_k = q.shape[-1]
    scores = q @ np.swapaxes(k, -1, -2) / np.sqrt(d_k)
    n = scores.shape[-1]
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    scores = np.where(mask[None, None, :, :], -1e9, scores)
    weights = softmax(scores, axis=-1)
    return weights @ v, weights

if __name__ == '__main__':
    rng = np.random.default_rng(7)
    q = rng.normal(size=(2, 4, 6, 16)); k = rng.normal(size=q.shape); v = rng.normal(size=q.shape)
    out, weights = causal_attention(q, k, v)
    print('Q/K/V:', q.shape, 'Output:', out.shape)
    print('Future attention weights:', weights[0,0][np.triu_indices(6,1)])
