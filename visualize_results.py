import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path
import torch
from models.lstm_model import LSTMPredictor
from models.transformer_model import TransformerPredictor
from models.cnn_transformer_model import CNNTransformerPredictor
from utils.data_processor import DataProcessor

def load_predictions():
    """加载模型预测结果"""
    with open('results/predictions.json', 'r') as f:
        predictions = json.load(f)
    return predictions

def plot_predictions(predictions, true_values, model_name, prediction_type):
    """绘制预测值与真实值的对比图"""
    plt.figure(figsize=(12, 6))
    plt.plot(true_values, label='Ground Truth', color='blue')
    plt.plot(predictions, label='Prediction', color='red', linestyle='--')
    plt.title(f'{model_name} - {prediction_type} Prediction vs Ground Truth')
    plt.xlabel('Time (days)')
    plt.ylabel('Global Active Power')
    plt.legend()
    plt.grid(True)
    return plt.gcf()

def plot_error_distribution(predictions, true_values, model_names):
    """绘制预测误差分布图"""
    plt.figure(figsize=(12, 6))
    for i, model in enumerate(model_names):
        errors = np.array(predictions[model]) - np.array(true_values)
        sns.kdeplot(errors, label=model)
    plt.title('Prediction Error Distribution')
    plt.xlabel('Error')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)
    return plt.gcf()

def plot_training_curves(history_file='results/training_history.json'):
    """绘制训练过程损失曲线"""
    with open(history_file, 'r') as f:
        history = json.load(f)
    
    plt.figure(figsize=(12, 6))
    for model in history:
        plt.plot(history[model]['train_loss'], label=f'{model} - Train')
        plt.plot(history[model]['val_loss'], label=f'{model} - Validation')
    plt.title('Training Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    return plt.gcf()

def main():
    # 创建输出目录
    Path('figures').mkdir(exist_ok=True)
    
    # 加载预测结果
    predictions = load_predictions()
    
    # 生成短期预测对比图
    fig_short = plot_predictions(
        predictions['short_term']['predictions'],
        predictions['short_term']['true_values'],
        'All Models',
        'Short-term'
    )
    fig_short.savefig('figures/short_term_prediction.png')
    
    # 生成长期预测对比图
    fig_long = plot_predictions(
        predictions['long_term']['predictions'],
        predictions['long_term']['true_values'],
        'All Models',
        'Long-term'
    )
    fig_long.savefig('figures/long_term_prediction.png')
    
    # 生成误差分布图
    fig_error = plot_error_distribution(
        predictions['short_term']['predictions'],
        predictions['short_term']['true_values'],
        ['LSTM', 'Transformer', 'CNN-Transformer']
    )
    fig_error.savefig('figures/error_distribution.png')
    
    # 生成训练过程曲线
    fig_train = plot_training_curves()
    fig_train.savefig('figures/training_curves.png')
    
    plt.close('all')

if __name__ == '__main__':
    main() 