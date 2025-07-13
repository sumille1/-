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

class CNNFeatureExtractor(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        
        self.cnn = nn.Sequential(
            # 第一个卷积块
            nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            # 第二个卷积块
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            # 第三个卷积块
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
    def forward(self, x):
        # 输入形状: [batch_size, seq_len, input_dim]
        # 转换为卷积输入形状: [batch_size, input_dim, seq_len]
        x = x.transpose(1, 2)
        
        # CNN特征提取
        x = self.cnn(x)
        
        # 转换回序列形状: [batch_size, seq_len, hidden_dim]
        x = x.transpose(1, 2)
        
        return x

class CNNTransformerPredictor(nn.Module):
    def __init__(self, input_dim: int, d_model: int, nhead: int, n_layers: int, output_dim: int):
        super().__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        
        # CNN特征提取器
        self.cnn = CNNFeatureExtractor(input_dim, d_model)
        
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
        
        # 特征融合
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # 输出层
        self.output_layer = nn.Linear(d_model, output_dim)
        
    def forward(self, x):
        # CNN特征提取
        cnn_features = self.cnn(x)
        
        # Transformer特征提取
        trans_input = cnn_features
        trans_input = self.pos_encoder(trans_input)
        trans_features = self.transformer_encoder(trans_input)
        
        # 特征融合
        cnn_out = cnn_features[:, -1, :]  # 最后一个时间步的CNN特征
        trans_out = trans_features[:, -1, :]  # 最后一个时间步的Transformer特征
        combined_features = torch.cat([cnn_out, trans_out], dim=1)
        fused_features = self.fusion(combined_features)
        
        # 输出预测
        output = self.output_layer(fused_features)
        
        return output 