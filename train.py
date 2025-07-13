import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
import logging
from models.lstm_model import LSTMPredictor
from models.transformer_model import TransformerPredictor
from models.cnn_transformer_model import CNNTransformerPredictor
from utils.data_processor import DataProcessor

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def train_model(model, train_loader, val_loader, criterion, optimizer, 
                device, epochs, model_save_path, patience=10):
    """训练模型并保存最佳权重"""
    model.to(device)
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        total_train_loss = 0
        train_steps = 0
        
        for batch_x, batch_y in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs}'):
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            train_steps += 1
        
        avg_train_loss = total_train_loss / train_steps
        train_losses.append(avg_train_loss)
        
        # 验证阶段
        model.eval()
        total_val_loss = 0
        val_steps = 0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                
                total_val_loss += loss.item()
                val_steps += 1
        
        avg_val_loss = total_val_loss / val_steps
        val_losses.append(avg_val_loss)
        
        logger.info(f'Epoch {epoch + 1}: '
                   f'train_loss = {avg_train_loss:.4f}, '
                   f'val_loss = {avg_val_loss:.4f}')
        
        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), model_save_path)
            patience_counter = 0
            logger.info(f'Saved best model with validation loss: {best_val_loss:.4f}')
        else:
            patience_counter += 1
            
        # 早停
        if patience_counter >= patience:
            logger.info(f'Early stopping triggered after {epoch + 1} epochs')
            break
    
    return {
        'train_loss': train_losses,
        'val_loss': val_losses,
        'best_val_loss': best_val_loss
    }

def main():
    # 创建必要的目录
    Path('models/weights').mkdir(parents=True, exist_ok=True)
    Path('results').mkdir(exist_ok=True)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')
    
    # 加载数据
    data_processor = DataProcessor()
    train_loader, val_loader, test_loader = data_processor.get_data_loaders()
    
    # 模型参数
    input_dim = 12  # 特征维度
    hidden_dim = 128
    output_dim = 90  # 短期预测天数
    
    # 训练参数
    epochs = 100
    learning_rate = 0.001
    
    # 定义模型配置
    models = {
        'LSTM': LSTMPredictor(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            n_layers=2,
            output_dim=output_dim
        ),
        'Transformer': TransformerPredictor(
            input_dim=input_dim,
            d_model=hidden_dim,
            nhead=8,
            n_layers=3,
            output_dim=output_dim
        ),
        'CNN-Transformer': CNNTransformerPredictor(
            input_dim=input_dim,
            d_model=hidden_dim,
            nhead=8,
            n_layers=3,
            output_dim=output_dim
        )
    }
    
    # 训练历史
    training_history = {}
    
    # 训练每个模型
    for model_name, model in models.items():
        logger.info(f'\nTraining {model_name} model...')
        
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
        # 训练模型
        history = train_model(
            model=model,
            train_loader=train_loader['short'],
            val_loader=val_loader['short'],
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epochs=epochs,
            model_save_path=f'models/weights/{model_name.lower()}_model.pth'
        )
        
        training_history[model_name] = {
            'train_loss': history['train_loss'],
            'val_loss': history['val_loss'],
            'best_val_loss': history['best_val_loss']
        }
    
    # 保存训练历史
    with open('results/training_history.json', 'w') as f:
        json.dump(training_history, f, indent=2)
    
    logger.info('\nTraining completed! Models saved in models/weights directory.')

if __name__ == '__main__':
    main() 