import torch
import torch.nn as nn
import torch.nn.functional as F


def combine_encodings_concat(z, z_a):
    """
    Concatenation: Concatenate the encoded video clips and actions along the feature dimension.
    """
    z_combined = torch.cat([z, z_a], dim=-1)
    return z_combined


def combine_encodings_add(z, z_a):
    """
    Addition: Add the encoded video clips and actions element-wise.
    """
    z_combined = z + z_a
    return z_combined


class AttentionFusion(nn.Module):
    """
    Attention-based fusion: Use an attention mechanism to weight the importance of video clips and actions based on their relevance.
    """

    def __init__(self, hidden_dim):
        super(AttentionFusion, self).__init__()
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=8)

    def forward(self, z, z_a):
        z_combined, _ = self.attention(z, z_a, z_a)
        return z_combined
