# 家庭电力消耗预测系统

这个项目实现了一个基于深度学习的家庭电力消耗预测系统，使用多种模型对家庭电力消耗进行短期（90天）和长期（365天）预测。

## 项目结构

```
power_consumption/
├── data/               # 数据目录
│   ├── train.csv      # 训练数据
│   └── test.csv       # 测试数据
├── models/            # 模型实现
│   ├── lstm_model.py
│   ├── transformer_model.py
│   └── cnn_transformer_model.py
├── utils/             # 工具函数
│   └── data_processor.py
├── train.py          # 训练脚本
└── requirements.txt   # 项目依赖
```

## 数据说明

项目使用的数据来自UCI Machine Learning Repository的"Individual household electric power consumption"数据集，包含以下特征：

- global_active_power：全局有功功率（kW）
- global_reactive_power：全局无功功率（kW）
- voltage：电压（V）
- global_intensity：电流强度（A）
- sub_metering_1：厨房区域能耗（Wh）
- sub_metering_2：洗衣房区域能耗（Wh）
- sub_metering_3：气候控制系统能耗（Wh）
- RR：月累计降水高度
- NBJRR1/5/10：当月日降水≥1/5/10mm的天数
- NBJBROU：当月雾出现的天数

## 模型说明

项目实现了三种深度学习模型：

1. LSTM模型：使用长短期记忆网络进行时间序列预测
2. Transformer模型：使用Transformer架构进行序列建模
3. CNN-Transformer混合模型：结合CNN的局部特征提取能力和Transformer的长期依赖建模能力

## 环境配置

1. 创建虚拟环境（推荐）：
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

## 使用说明

1. 准备数据：
   - 将训练数据放在 `data/train.csv`
   - 将测试数据放在 `data/test.csv`

2. 训练模型：
```bash
python train.py
```

训练过程会自动：
- 对数据进行预处理
- 训练三种不同的模型
- 保存训练结果和最佳模型
- 输出评估指标（MSE和MAE）

## 模型评估

评估使用两个指标：
- MSE（均方误差）：评估预测值与真实值的平方差
- MAE（平均绝对误差）：评估预测值与真实值的绝对差

每个模型都会进行5轮实验，取平均值并计算标准差以评估模型的稳定性。

## 改进模型说明

CNN-Transformer混合模型的主要改进点：

1. CNN特征提取：
   - 使用多层卷积网络提取局部时间特征
   - 采用残差连接提升梯度流动
   - 使用批归一化加速训练

2. 多尺度注意力机制：
   - 在不同时间尺度上计算注意力
   - 通过池化和上采样处理多尺度特征
   - 融合不同尺度的特征表示

3. 混合架构优势：
   - 结合CNN的局部特征提取能力
   - 利用Transformer的长期依赖建模能力
   - 通过多尺度机制捕获不同时间跨度的模式 