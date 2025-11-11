import torch.nn as nn

class AddOperator(nn.Module):
    def forward(self, *inputs):
        return sum(inputs)
