import torch
import numpy as np
import json
from pathlib import Path
import logging
from models.lstm_model import LSTMPredictor
from models.transformer_model import TransformerPredictor
from models.cnn_transformer_model import CNNTransformerPredictor
from utils.data_processor import DataProcessor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def evaluate_model(model, test_loader, criterion, device):
    """评估模型性能"""
    model.eval()
    predictions = []
    true_values = []
    
    logger.info("Starting model evaluation...")
    with torch.no_grad():
        for batch_x, batch_y in tqdm(test_loader, desc="Evaluating"):
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            outputs = model(batch_x)
            
            predictions.extend(outputs.cpu().numpy())
            true_values.extend(batch_y.cpu().numpy())
    
    predictions = np.array(predictions)
    true_values = np.array(true_values)
    
    mse = mean_squared_error(true_values, predictions)
    mae = mean_absolute_error(true_values, predictions)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((true_values - predictions) / true_values)) * 100
    
    logger.info(f"Evaluation metrics:")
    logger.info(f"MSE: {mse:.4f}")
    logger.info(f"MAE: {mae:.4f}")
    logger.info(f"RMSE: {rmse:.4f}")
    logger.info(f"MAPE: {mape:.4f}%")
    
    return {
        'predictions': predictions.tolist(),
        'true_values': true_values.tolist(),
        'metrics': {
            'mse': float(mse),
            'mae': float(mae),
            'rmse': float(rmse),
            'mape': float(mape)
        }
    }

def plot_predictions(results, save_path, prediction_type='short'):
    """绘制预测结果对比图"""
    logger.info(f"Plotting {prediction_type}-term predictions...")
    
    plt.style.use('seaborn')  # 使用更现代的样式
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # 确保数据是一维的
    predictions = np.array(results['predictions']).flatten()
    true_values = np.array(results['true_values']).flatten()
    
    # 使用移动平均来平滑数据
    window = 7  # 7天移动平均
    predictions_smooth = pd.Series(predictions).rolling(window=window, center=True).mean()
    true_values_smooth = pd.Series(true_values).rolling(window=window, center=True).mean()
    
    # 设置时间步长
    time_steps = len(true_values)
    days = 90 if prediction_type == 'short' else 365
    x_ticks = np.linspace(0, time_steps-1, 7)
    x_labels = [f'{int(x/time_steps * days)}' for x in x_ticks]
    
    # 绘制真实值和预测值
    ax.plot(true_values_smooth, 
            label='True Values', 
            color='#2ecc71',  # 绿色
            linewidth=2,
            alpha=0.9)
    
    ax.plot(predictions_smooth, 
            label='Predictions', 
            color='#e74c3c',  # 红色
            linewidth=2,
            alpha=0.9)
    
    # 设置标题和标签
    title = f"{'Short-term' if prediction_type == 'short' else 'Long-term'} Power Consumption Prediction ({days} days)"
    ax.set_title(title, fontsize=14, pad=20)
    ax.set_xlabel('Time (Days)', fontsize=12)
    ax.set_ylabel('Normalized Power Consumption', fontsize=12)
    
    # 设置网格
    ax.grid(True, linestyle='--', alpha=0.3)
    
    # 设置图例
    ax.legend(fontsize=12, frameon=True, loc='upper right')
    
    # 设置刻度
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels)
    ax.tick_params(axis='both', which='major', labelsize=10)
    
    # 移除顶部和右侧边框
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Plot saved to {save_path}")

def train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs, model_save_path, patience=10):
    """训练模型并保存最佳权重"""
    logger.info(f"Training {model.__class__.__name__}...")
    model.to(device)
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
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
        
        avg_val_loss = total_val_loss / val_steps if val_steps > 0 else float('inf')
        
        logger.info(f'Epoch {epoch + 1}: train_loss = {avg_train_loss:.4f}, val_loss = {avg_val_loss:.4f}')
        
        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict().copy()
            logger.info(f'Found new best model with validation loss: {best_val_loss:.4f}')
            patience_counter = 0
        else:
            patience_counter += 1
            
        # 早停
        if patience_counter >= patience:
            logger.info(f'Early stopping triggered after {epoch + 1} epochs')
            break
    
    # 保存最佳模型
    if best_model_state is not None:
        torch.save(best_model_state, model_save_path)
        logger.info(f'Saved best model to {model_save_path}')
        # 加载最佳模型权重
        model.load_state_dict(best_model_state)
    
    return best_val_loss

def main():
    try:
        # 创建必要的目录
        current_dir = Path('/data/sjchen/power_consumption')
        results_dir = current_dir / 'results'
        figures_dir = current_dir / 'figures'
        weights_dir = current_dir / 'models' / 'weights'
        
        for dir_path in [results_dir, figures_dir, weights_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Directory created/verified: {dir_path}")
        
        # 设置设备
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")
        
        # 加载数据
        data_processor = DataProcessor()
        train_loaders, val_loaders, test_loaders = data_processor.get_data_loaders()
        
        # 模型参数
        input_dim = 7  # 特征维度
        hidden_dim = 128
        
        # 评估结果
        results = {'short': {}, 'long': {}}
        criterion = torch.nn.MSELoss()
        
        # 评估每个模型的短期和长期预测
        for prediction_type in ['short', 'long']:
            output_dim = 90 if prediction_type == 'short' else 365
            logger.info(f"\nProcessing {prediction_type}-term predictions ({output_dim} days)...")
            
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
            
            for name, model in models.items():
                logger.info(f"\nProcessing {name} model...")
                model.to(device)
                
                # 训练模型
                optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
                weight_path = weights_dir / f'{name.lower()}_{prediction_type}_model.pth'
                
                best_val_loss = train_model(
                    model=model,
                    train_loader=train_loaders[prediction_type],
                    val_loader=val_loaders[prediction_type],
                    criterion=criterion,
                    optimizer=optimizer,
                    device=device,
                    epochs=50,
                    model_save_path=weight_path
                )
                
                # 评估模型
                model_results = evaluate_model(
                    model=model,
                    test_loader=test_loaders[prediction_type],
                    criterion=criterion,
                    device=device
                )
                results[prediction_type][name] = model_results
                
                # 绘制预测结果
                figure_path = figures_dir / f'{name.lower()}_{prediction_type}_predictions.png'
                plot_predictions(
                    results=model_results,
                    save_path=str(figure_path),
                    prediction_type=prediction_type
                )
        
        # 保存评估结果
        results_path = results_dir / 'evaluation_metrics.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"\nResults saved to {results_path}")
        
        logger.info("\nEvaluation completed successfully!")
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)
        raise

if __name__ == '__main__':
    main() 