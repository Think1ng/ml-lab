import torch.nn as nn

from layers.add_operator import AddOperator
from layers.concat_operator import ConcatOperator
from layers.simple_attention import SimpleAttention

LAYER_REGISTRY = {
    # MLP
    "Linear": nn.Linear,
    "ReLU": nn.ReLU,
    "LeakyReLU": nn.LeakyReLU,
    "ELU": nn.ELU,
    "Sigmoid": nn.Sigmoid,
    "Tanh": nn.Tanh,
    "Dropout": nn.Dropout,
    "BatchNorm1d": nn.BatchNorm1d,
    "LayerNorm": nn.LayerNorm,
    
    # CNN
    "Conv1d": nn.Conv1d,
    "Conv2d": nn.Conv2d,
    "Conv3d": nn.Conv3d,
    "MaxPool1d": nn.MaxPool1d,
    "MaxPool2d": nn.MaxPool2d,
    "AvgPool1d": nn.AvgPool1d,
    "AvgPool2d": nn.AvgPool2d,
    "AdaptiveAvgPool1d": nn.AdaptiveAvgPool1d,
    "AdaptiveAvgPool2d": nn.AdaptiveAvgPool2d,
    "BatchNorm2d": nn.BatchNorm2d,
    "Dropout2d": nn.Dropout2d,
    
    # RNN / sequential
    "RNN": nn.RNN,
    "LSTM": nn.LSTM,
    "GRU": nn.GRU,
    "BidirectionalLSTM": lambda **kwargs: nn.LSTM(**kwargs, bidirectional=True),
    
    # Attention / Transformer
    "SimpleAttention": SimpleAttention,
    "MultiHeadAttention": nn.MultiheadAttention,
    "TransformerEncoderLayer": nn.TransformerEncoderLayer,
    "TransformerDecoderLayer": nn.TransformerDecoderLayer,
    # "PositionalEncoding": lambda d_model, max_len=5000: PositionalEncoding(d_model, max_len),

    # Graph / residual / branch
    "Add": AddOperator,
    "Concat": ConcatOperator,
    # "Mul": lambda: MulOperator(),
    "Identity": nn.Identity,
    "SkipConnection": AddOperator,
    
    # Utility / output
    "Flatten": nn.Flatten,
    # "Reshape": lambda *args: ReshapeOperator(*args),
    "GlobalAvgPool": nn.AdaptiveAvgPool1d,
    "Softmax": nn.Softmax,
    "LogSoftmax": nn.LogSoftmax,
}