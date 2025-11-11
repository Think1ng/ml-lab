import torch
import torch.nn as nn

class ConcatOperator(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim
    def forward(self, *inputs):
        return torch.cat(inputs, dim=self.dim)