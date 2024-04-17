import torch
import torch.nn.functional as F

def scaled_dot_product_attention(q, k, v, dropout_p=0.0):
    """
    Computes scaled dot product attention.

    Args:
        q (torch.Tensor): Query tensor of shape (batch_size, num_heads, seq_len_q, head_dim).
        k (torch.Tensor): Key tensor of shape (batch_size, num_heads, seq_len_k, head_dim).
        v (torch.Tensor): Value tensor of shape (batch_size, num_heads, seq_len_v, head_dim).
        dropout_p (float, optional): Dropout probability. Default is 0.0.

    Returns:
        torch.Tensor: Output tensor of shape (batch_size, num_heads, seq_len_q, head_dim).
    """
    # Compute attention scores
    attn_scores = torch.matmul(q, k.transpose(-2, -1))
    attn_scores = attn_scores / torch.sqrt(torch.tensor(k.size(-1), dtype=torch.float32))

    # Apply softmax to attention scores
    attn_probs = F.softmax(attn_scores, dim=-1)

    # Apply dropout to attention probabilities
    attn_probs = F.dropout(attn_probs, p=dropout_p)

    # Compute attention output
    attn_output = torch.matmul(attn_probs, v)

    return attn_output