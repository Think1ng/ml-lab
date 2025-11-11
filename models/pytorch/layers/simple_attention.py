import torch
import torch.nn as nn

class SimpleAttention(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.query = nn.Linear(input_dim, output_dim)
        self.key = nn.Linear(input_dim, output_dim)
        self.value = nn.Linear(input_dim, output_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        attn_scores = self.softmax(Q @ K.transpose(-2, -1) / (K.shape[-1] ** 0.5))
        return attn_scores @ V
