import torch
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from model import TransformerModel  # 导入模型定义
import okx.Trade as Trade

class OrderBookModel:
    def __init__(self, data_path, model_path, device=None):
        # 选择设备
        self.device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # 存储数据路径
        self.data_path = data_path

        # 读取数据
        self.df = pd.read_csv(self.data_path)
        self.df = self.df.apply(pd.to_numeric, errors='coerce')
        self.df = self.df.fillna(0)

        # 计算未来 50 个时间间隔的平均价格
        future_avg_price = self.df['best_ask'].rolling(window=50).mean().shift(-50)

        # 目标值定义：未来平均价格是否高于当前价格
        self.df['target'] = (future_avg_price > self.df['best_ask']).astype(int)

        # 移除无法计算目标值的行
        self.df = self.df.dropna()

        # 数据准备
        self.X = self.df.drop(columns=['target']).values
        self.y = self.df['target'].values

        # 转换为 PyTorch 张量
        self.X_tensor = torch.tensor(self.X, dtype=torch.float32).to(self.device)
        self.y_tensor = torch.tensor(self.y, dtype=torch.float32).view(-1, 1).to(self.device)

        # 创建 DataLoader
        self.test_data = TensorDataset(self.X_tensor, self.y_tensor)
        self.test_loader = DataLoader(self.test_data, batch_size=1024, shuffle=False, num_workers=0)

        # 模型初始化参数
        self.input_dim = self.X.shape[1]  # 输入特征的维度
        self.model_dim = 256  # 模型维度
        self.num_heads = 16  # 注意力头数
        self.num_layers = 4  # Transformer 层数
        self.output_dim = 1  # 输出维度（回归任务时是1）

        # 初始化 Transformer 模型并迁移到设备
        self.model = TransformerModel(self.input_dim, self.model_dim, self.num_heads, self.num_layers, self.output_dim).to(self.device)

        # 加载训练好的模型
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()  # 切换到评估模式

    def test_model(self):
        # 每次调用时都重新加载最新的数据
        self.df = pd.read_csv(self.data_path)  # 使用 self.data_path 而不是 data_path
        self.df = self.df.apply(pd.to_numeric, errors='coerce')
        self.df = self.df.fillna(0)
        self.df['target'] = (self.df['best_bid'].shift(-50) > self.df['best_ask'].shift(0)).astype(int)
        self.df = self.df.dropna()

        # 数据准备
        self.X = self.df.drop(columns=['target']).values
        self.y = self.df['target'].values

        # 转换为 PyTorch 张量
        self.X_tensor = torch.tensor(self.X, dtype=torch.float32).to(self.device)
        self.y_tensor = torch.tensor(self.y, dtype=torch.float32).view(-1, 1).to(self.device)

        # 创建 DataLoader
        self.test_data = TensorDataset(self.X_tensor, self.y_tensor)
        self.test_loader = DataLoader(self.test_data, batch_size=1024, shuffle=False, num_workers=0)

        # 开始预测
        all_preds = []
        all_labels = []

        with torch.no_grad():  # 关闭梯度计算以节省内存
            for inputs, targets in self.test_loader:
                # 数据迁移到设备
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                preds = torch.sigmoid(outputs)  # 将输出映射到[0, 1]之间
                preds = (preds > 0.5).float()  # 二分类阈值为0.5

                all_preds.append(preds.cpu().numpy())  # 收集预测结果
                all_labels.append(targets.cpu().numpy())  # 收集实际标签

        # 将所有批次的结果合并
        all_preds = np.concatenate(all_preds, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)

        # 保存预测结果到 CSV 文件
        results_df = pd.DataFrame({
            'Predicted_Label': all_preds.flatten()  # 预测标签
        })
        results_df.to_csv('prediction_results.csv', index=False)

        # 计算预测结果中1的比例
        ones_count = np.sum(all_preds)
        total_count = len(all_preds)
        ones_percentage = ones_count / total_count * 100
        print(f"Predicted win rates : {ones_percentage:.2f}%")

        # 根据预测结果执行订单
        if ones_percentage >= 70:
            print("Executing Buy Order...")
            self.execute_order("buy", "20000")  # 执行买单
        elif ones_percentage <= 60:
            print("Executing Sell Order...")
            self.execute_order("sell", "0.2")  # 执行卖单
        else:
            print("No action taken.")

    def execute_order(self, side, size):
        # 初始化 OKX 交易API
        api_key = "70df264e-0681-4ff8-9c6b-91476ed65912"
        secret_key = "67774F2EEDBC970B9D5905CEB8C45F41"
        passphrase = "Miaozhipeng1!"
        flag = "1"  # live trading: 0, demo trading: 1

        tradeAPI = Trade.TradeAPI(api_key, secret_key, passphrase, False, flag)

        # 执行订单
        result = tradeAPI.place_order(
            instId="BTC-USDT",
            tdMode="cash",
            side=side,  # "buy" 或 "sell"
            ordType="market",
            sz=size,
        )
        print(result)

# 使用类进行模型测试
if __name__ == "__main__":
    # 初始化 OrderBookModel 类
    order_book_model = OrderBookModel(data_path='order_book_normalized_data.csv', model_path='transformer_model.pth')

    # 在整个数据集上测试模型并执行订单
    order_book_model.test_model()
