import torch
from torch.utils.data import DataLoader, TensorDataset
from model import TransformerModel
import pandas as pd
import torch.nn as nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 数据预处理部分
df = pd.read_csv('order_book_data_with_features_normalized.csv')
df = df.apply(pd.to_numeric, errors='coerce')
df = df.fillna(0)

future_avg_price = df['best_ask'].rolling(window=50).mean().shift(-50)
df['target'] = (future_avg_price > df['best_ask']).astype(int)  # 1表示未来均价高，0表示未来均价低
df = df.dropna()  # 丢弃无法计算的行

X = df.drop(columns=['target', 'bids', 'asks']).values
y = df['target'].values

split_idx = int(len(X) * 0.8)  # 80% 用于训练
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# 张量处理
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(device)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1).to(device)

# 创建 DataLoader
train_data = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_data, batch_size=1024, shuffle=True, num_workers=0)

# 初始化模型
input_dim = X_train.shape[1]
model_dim = 256
num_heads = 16
num_layers = 4
output_dim = 1

model = TransformerModel(input_dim, model_dim, num_heads, num_layers, output_dim).to(device)

# 定义损失函数和优化器
criterion = nn.BCEWithLogitsLoss()  # 二分类交叉熵损失
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 训练模型
def train_model(model, train_loader, criterion, optimizer, epochs=10):
    for epoch in range(epochs):
        model.train()
        running_loss = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}")

# 开始训练
train_model(model, train_loader, criterion, optimizer, epochs=10)

# 保存模型
torch.save(model.state_dict(), 'transformer_model1.pth')
print("Model saved to 'transformer_model.pth'")
