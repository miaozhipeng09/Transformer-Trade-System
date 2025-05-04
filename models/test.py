import torch
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from model import TransformerModel  # 导入模型定义
import torch.nn as nn
from sklearn.metrics import accuracy_score

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 数据预处理部分
df = pd.read_csv('order_book_data_with_features_normalized.csv')
df = df.apply(pd.to_numeric, errors='coerce')
df = df.fillna(0)
future_avg_price = df['best_ask'].rolling(window=50).mean().shift(-50)
df['target'] = (future_avg_price > df['best_ask']).astype(int)  # 1表示未来均价高，0表示未来均价低
df = df.dropna()

X = df.drop(columns=['target','bids','asks' ]).values
y = df['target'].values

split_idx = int(len(X) * 0.8)  # 划分索引，前 80% 训练，后 20% 测试
X_test = X[split_idx:]
y_test = y[split_idx:]

# 转换为 PyTorch 张量
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1).to(device)

# 创建 DataLoader
test_data = TensorDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(test_data, batch_size=1024, shuffle=False, num_workers=0)

# 初始化模型
input_dim = X.shape[1]  # 输入特征的维度
model_dim = 256  # 模型维度
num_heads = 16  # 注意力头数
num_layers = 4  # Transformer 层数
output_dim = 1  # 输出维度（回归任务时是1）

# 初始化模型并迁移到设备
model = TransformerModel(input_dim, model_dim, num_heads, num_layers, output_dim).to(device)

# 加载训练好的模型
model.load_state_dict(torch.load('transformer_model.pth'))
model.eval()  # 切换到评估模式

# 预测并计算准确度
def test_model(model, test_loader):
    all_preds = []
    all_labels = []

    with torch.no_grad():  # 关闭梯度计算以节省内存
        for inputs, targets in test_loader:
            # 数据迁移到设备
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            preds = torch.sigmoid(outputs)  # 将输出映射到[0, 1]之间
            preds = (preds > 0.5).float()  # 二分类阈值为0.5

            all_preds.append(preds.cpu().numpy())  # 收集预测结果
            all_labels.append(targets.cpu().numpy())  # 收集实际标签

    # 将所有批次的结果合并
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    accuracy = accuracy_score(all_labels, all_preds)  # 计算准确度

    # 保存预测结果到 CSV 文件
    results_df = pd.DataFrame({
        'True_Label': all_labels.flatten(),  # 真实标签
        'Predicted_Label': all_preds.flatten()  # 预测标签
    })
    results_df.to_csv('prediction_results.csv', index=False)
    print("Predictions saved to 'prediction_results.csv'")

    return accuracy

# 在测试集上测试模型
accuracy = test_model(model, test_loader)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
