import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score

df = pd.read_csv('order_book_data_with_features_normalized.csv')

df = df.apply(pd.to_numeric, errors='coerce')

df = df.fillna(0)

df['target'] = (df['best_bid'].shift(-10) > df['best_ask'].shift(0)).astype(int)  # 1表示涨，0表示跌

df = df.dropna()

X = df.drop(columns=['target',])
y = df['target']
#张量处理
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

# 创建 DataLoader
train_data = TensorDataset(X_train_tensor, y_train_tensor)
test_data = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32)

#模型
class TransformerModel(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_layers, output_dim):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(input_dim, model_dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads),
            num_layers=num_layers
        )
        self.fc = nn.Linear(model_dim, output_dim)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x.unsqueeze(1))  # 增加一个维度，使其符合Transformer的要求
        return self.fc(x.squeeze(1))  # 输出单个标量

    def train_model(self, train_loader, criterion, optimizer, epochs=20):
        # 训练模型
        for epoch in range(epochs):
            self.train()  # 设定模型为训练模式
            running_loss = 0.0
            for inputs, targets in train_loader:
                optimizer.zero_grad()  # 清空梯度
                outputs = self(inputs)  # 前向传播

                loss = criterion(outputs, targets)  # 计算损失
                loss.backward()  # 反向传播
                optimizer.step()  # 更新参数

                running_loss += loss.item()

            avg_loss = running_loss / len(train_loader)
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}")

    def evaluate(self, test_loader):
        # 在测试集上评估模型
        self.eval()  # 设定模型为评估模式
        predictions = []
        targets = []

        with torch.no_grad():
            for inputs, target in test_loader:
                outputs = self(inputs)

                predictions.append(outputs)
                targets.append(target)

        # 将预测结果和实际值转换为一维数组
        predictions = torch.cat(predictions).numpy()
        targets = torch.cat(targets).numpy()

        # 使用 sigmoid 激活函数输出概率，并将其转化为标签（0 或 1）
        predictions = torch.sigmoid(torch.tensor(predictions)).numpy()
        predictions = (predictions > 0.5).astype(int)
        accuracy = accuracy_score(targets, predictions)
        print(f"Test Accuracy: {accuracy:.4f}")
        return accuracy

#初始化模型
input_dim = X_train.shape[1]  # 输入特征的维度
model_dim = 256  # 模型维度
num_heads = 8  # 注意力头数
num_layers = 4  # Transformer 层数
output_dim = 1  # 输出维度（回归任务时是1）

model = TransformerModel(input_dim, model_dim, num_heads, num_layers, output_dim)

# Step 9: 定义损失函数和优化器
criterion = nn.BCEWithLogitsLoss()  # 二分类交叉熵损失
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# Step 10: 训练模型
model.train_model(train_loader, criterion, optimizer, epochs=20)

# Step 11: 在测试集上评估模型
model.evaluate(test_loader)

# Step 12: 保存模型
torch.save(model.state_dict(), 'transformer_model.pth')
print("Model saved to 'transformer_model.pth'")

