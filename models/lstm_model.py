import torch
import torch.nn as nn

class LSTMPredictor(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, n_layers: int, output_dim: int):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        
        # LSTM层
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=0.2 if n_layers > 1 else 0
        )
        
        # 输出层
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        # LSTM前向传播
        lstm_out, _ = self.lstm(x)
        
        # 取最后一个时间步的输出进行预测
        out = self.fc(lstm_out[:, -1, :])
        
        return out 