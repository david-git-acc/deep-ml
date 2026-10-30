import numpy as np
from typing import Tuple

def stable_softmax(x : np.ndarray, axis : int = 0) -> np.ndarray:

    c = x.max(axis = axis, keepdims = True)
    exp_powers = np.exp(x - c)
    bigsum = exp_powers.sum(axis = axis, keepdims = True)

    return exp_powers / bigsum


def compute_qkv(X: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Query, Key, and Value matrices.
    
    Args:
        X: Input matrix of shape (seq_len, d_model)
        W_q, W_k, W_v: Weight matrices of shape (d_model, d_model)
    
    Returns:
        Q, K, V matrices each of shape (seq_len, d_model)
    """
    
    Q = X @ W_q
    K = X @ W_k
    V = X @ W_v

    return (Q, K, V)

def self_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> np.ndarray:
    """
    Compute scaled dot-product self-attention.
    
    Args:
        Q: Query matrix of shape (seq_len, d_k)
        K: Key matrix of shape (seq_len, d_k)
        V: Value matrix of shape (seq_len, d_k)
    
    Returns:
        Attention output of shape (seq_len, d_k)
    """
    
    d_k = Q.shape[-1]
    pairs = Q @ K.swapaxes(-1, -2) / (d_k ** 0.5)
    softmaxed = stable_softmax(pairs, axis = -1)
    attention = softmaxed @ V

    return attention


def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray, n_heads: int) -> np.ndarray:
    """
    Compute multi-head attention.
    
    Args:
        Q, K, V: Matrices of shape (seq_len, d_model)
        n_heads: Number of attention heads
    
    Returns:
        Attention output of shape (seq_len, d_model)
    """
    # Your code here

    S, d_model = Q.shape
    d_k = d_model // n_heads

    Q = Q.reshape(S, n_heads, d_k).transpose(1, 0, 2)
    K = K.reshape(S, n_heads, d_k).transpose(1, 0, 2)
    V = V.reshape(S, n_heads, d_k).transpose(1, 0, 2)

    attention = self_attention(Q, K, V)

    untransposed = attention.transpose(1, 0, 2)
    mhattention = untransposed.reshape(S, d_model)

    return mhattention



    