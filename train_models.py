import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from models.lstm_model import LSTMPredictor
from models.transformer_model import TransformerPredictor
from models.cnn_transformer_model import CNNTransformerPredictor
from config import Config
import os
import json
from tqdm import tqdm

def load_and_preprocess_data(file_path, config):
    # 读取数据
    df = pd.read_csv(file_path, sep=';', decimal=',')
    
    # 转换数值列
    numeric_columns = ['Global_active_power', 'Global_reactive_power', 'Voltage',
                      'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']
    
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # 转换日期列
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
    
    # 按日期聚合
    daily_df = df.groupby('Date').agg({
        'Global_active_power': 'mean',
        'Global_reactive_power': 'mean',
        'Voltage': 'mean',
        'Global_intensity': 'mean',
        'Sub_metering_1': 'mean',
        'Sub_metering_2': 'mean',
        'Sub_metering_3': 'mean'
    }).reset_index()
    
    # 添加时间特征
    daily_df['day_of_week'] = daily_df['Date'].dt.dayofweek
    daily_df['month'] = daily_df['Date'].dt.month
    daily_df['day_of_month'] = daily_df['Date'].dt.day
    daily_df['day_of_year'] = daily_df['Date'].dt.dayofyear
    daily_df['week_of_year'] = daily_df['Date'].dt.isocalendar().week
    
    # 准备特征和目标
    features = ['Global_active_power', 'Global_reactive_power', 'Voltage', 'Global_intensity',
                'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3', 'day_of_week',
                'month', 'day_of_month', 'day_of_year', 'week_of_year']
    
    # 标准化数据
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(daily_df[features])
    
    # 创建时间窗口数据
    X, y = [], []
    for i in range(len(data_scaled) - config.input_window - config.output_window + 1):
        X.append(data_scaled[i:i + config.input_window])
        y.append(data_scaled[i + config.input_window:i + config.input_window + config.output_window, 0])
    
    X = np.array(X)
    y = np.array(y)
    
    # 分割训练集和测试集
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    return X_train, y_train, X_test, y_test, scaler

def train_model(model, model_name, train_loader, val_loader, config, save_dir):
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    for epoch in range(config.num_epochs):
        # 训练阶段
        model.train()
        total_train_loss = 0
        train_batches = 0
        
        for X_batch, y_batch in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{config.num_epochs}'):
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            train_batches += 1
        
        avg_train_loss = total_train_loss / train_batches
        train_losses.append(avg_train_loss)
        
        # 验证阶段
        model.eval()
        total_val_loss = 0
        val_batches = 0
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                y_pred = model(X_batch)
                loss = criterion(y_pred, y_batch)
                total_val_loss += loss.item()
                val_batches += 1
        
        avg_val_loss = total_val_loss / val_batches
        val_losses.append(avg_val_loss)
        
        print(f'Epoch {epoch + 1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}')
        
        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pth'))
    
    # 保存训练结果
    results = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_val_loss': best_val_loss
    }
    
    with open(os.path.join(save_dir, 'training_results.json'), 'w') as f:
        json.dump(results, f, indent=4)

def main():
    config = Config()
    
    # 加载数据
    print("Loading and preprocessing data...")
    X_train, y_train, X_test, y_test, scaler = load_and_preprocess_data(
        '/data/sjchen/machine_learning/data/household_power_consumption.txt',
        config
    )
    
    # 转换为PyTorch张量
    X_train = torch.FloatTensor(X_train)
    y_train = torch.FloatTensor(y_train)
    X_test = torch.FloatTensor(X_test)
    y_test = torch.FloatTensor(y_test)
    
    # 创建数据加载器
    train_size = int(0.8 * len(X_train))
    train_dataset = torch.utils.data.TensorDataset(X_train[:train_size], y_train[:train_size])
    val_dataset = torch.utils.data.TensorDataset(X_train[train_size:], y_train[train_size:])
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config.batch_size)
    
    # 训练LSTM模型
    print("\nTraining LSTM model...")
    lstm_model = LSTMPredictor(**config.lstm_config)
    train_model(lstm_model, 'LSTM', train_loader, val_loader, config, config.model_save_paths['lstm'])
    
    # 训练Transformer模型
    print("\nTraining Transformer model...")
    transformer_model = TransformerPredictor(**config.transformer_config)
    train_model(transformer_model, 'Transformer', train_loader, val_loader, config, config.model_save_paths['transformer'])
    
    # 训练CNN-Transformer模型
    print("\nTraining CNN-Transformer model...")
    cnn_transformer_model = CNNTransformerPredictor(**config.cnn_transformer_config)
    train_model(cnn_transformer_model, 'CNN-Transformer', train_loader, val_loader, config, config.model_save_paths['cnn_transformer'])
    
    print("\nTraining completed! Models saved in their respective directories.")

if __name__ == '__main__':
    main() 