import asyncio
import json
import websockets
import csv
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from collections import deque
from real_test import OrderBookModel  # 导入模型类

SYMBOL = "BTC-USDT"
WS_URL = "wss://ws.okx.com:8443/ws/v5/public"
OUTPUT_FILE = "order_book_normalized_data.csv"
MODEL_PATH = "transformer_model.pth"  # 模型路径


# 初始化 CSV 文件
def init_csv(output_file):
    with open(output_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "timestamp",
                "symbol",
                "best_bid",
                "best_ask",
                "spread",
                "total_bid_volume",
                "best_bid_volume",
                "total_ask_volume",
                "best_ask_volume",
            ]
        )


# 保存归一化后的数据到 CSV（覆盖文件）
def save_to_csv(data, output_file):
    with open(output_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        # 写入标题行
        writer.writerow(
            [
                "timestamp",
                "symbol",
                "best_bid",
                "best_ask",
                "spread",
                "total_bid_volume",
                "best_bid_volume",
                "total_ask_volume",
                "best_ask_volume",
            ]
        )
        # 写入归一化数据
        writer.writerows(data)


# 特征提取
def extract_features(timestamp, symbol, bids, asks):
    if not bids or not asks:
        return []

    bid_prices = [float(item[0]) for item in bids]
    bid_volumes = [float(item[1]) for item in bids]
    ask_prices = [float(item[0]) for item in asks]
    ask_volumes = [float(item[1]) for item in asks]

    features = [
        float(timestamp),  # 时间戳转为浮点数
        symbol,  # 交易对
        bid_prices[0],  # 最优买入价格
        ask_prices[0],  # 最优卖出价格
        ask_prices[0] - bid_prices[0],  # 买卖差价
        sum(bid_volumes),  # 总买入量
        bid_volumes[0],  # 最优买入挂单量
        sum(ask_volumes),  # 总卖出量
        ask_volumes[0],  # 最优卖出挂单量
    ]
    return features


# WebSocket消息处理函数
async def on_message(message, model):
    global data_window
    data = json.loads(message)

    if "data" in data:
        order_book = data["data"][0]
        timestamp = float(order_book["ts"])  # 确保时间戳是数值型
        bids = order_book["bids"]
        asks = order_book["asks"]

        # 提取特征
        features = extract_features(timestamp, SYMBOL, bids, asks)
        if not features:
            return

        # 添加到滑动窗口
        data_window.append(features)

        # 当滑动窗口满 n 条时，进行归一化处理
        if len(data_window) == window_size:
            # 转换窗口数据为 NumPy 数组
            window_array = np.array(data_window)  # 确保每列都是数值型
            symbol_column = window_array[:, 1]  # 提取 symbol 列
            numeric_columns = window_array[:, [0, 2, 3, 4, 5, 6, 7, 8]].astype(float)  # 仅数值列

            # 使用 StandardScaler 进行归一化
            scaler = StandardScaler()
            normalized_numeric_columns = scaler.fit_transform(numeric_columns)

            # 将归一化后的数据重新组合
            normalized_data = [
                [normalized_numeric_columns[i, 0]] + [symbol_column[i]] + normalized_numeric_columns[i, 1:].tolist()
                for i in range(window_size)
            ]

            # 保存归一化结果到 CSV（覆盖文件）
            save_to_csv(normalized_data, OUTPUT_FILE)


            # 调用模型进行预测
            model.test_model()

            # 清空滑动窗口，准备接收新的数据
            data_window.clear()


# WebSocket连接函数
async def subscribe(model):
    print("开始实时抓取和归一化...")
    init_csv(OUTPUT_FILE)

    async with websockets.connect(WS_URL) as ws:
        subscribe_message = {
            "op": "subscribe",
            "args": [{"channel": "books", "instId": SYMBOL, "depth": "5"}],
        }
        await ws.send(json.dumps(subscribe_message))

        while True:
            message = await ws.recv()
            await on_message(message, model)


# 主函数
def main():
    # 初始化模型
    order_book_model = OrderBookModel(data_path=OUTPUT_FILE, model_path=MODEL_PATH,)

    # 启动 WebSocket 数据抓取和处理
    asyncio.run(subscribe(order_book_model))


# 初始化滑动窗口
window_size = 50
data_window = deque(maxlen=window_size)

if __name__ == "__main__":
    main()
