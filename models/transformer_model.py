import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x.transpose(0, 1)
        x = x + self.pe[:x.size(0)]
        return x.transpose(0, 1)

class TransformerPredictor(nn.Module):
    def __init__(self, input_dim: int, d_model: int, nhead: int, n_layers: int, output_dim: int):
        super().__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        
        # 输入投影
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer编码器
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=2048,
            dropout=0.1,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers,
            num_layers=n_layers
        )
        
        # 输出层
        self.output_layer = nn.Linear(d_model, output_dim)
        
    def forward(self, src):
        # 输入投影
        src = self.input_projection(src)
        
        # 位置编码
        src = self.pos_encoder(src)
        
        # Transformer编码
        output = self.transformer_encoder(src)
        
        # 取最后一个时间步的输出进行预测
        output = self.output_layer(output[:, -1, :])
        
        return output 