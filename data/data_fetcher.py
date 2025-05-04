import asyncio
import json
import websockets
import csv
import os
import time
import sys

SYMBOL = "BTC-USDT"
WS_URL = "wss://ws.okx.com:8443/ws/v5/public"
OUTPUT_FILE = "order_book_data3.csv"
TIMESTAMP_FILE = "last_timestamp.txt"


# 获取最后存储的时间戳
def get_last_timestamp():
    if os.path.isfile(TIMESTAMP_FILE):
        with open(TIMESTAMP_FILE, 'r') as f:
            return f.read().strip()
    return None


# 保存时间戳
def save_timestamp(timestamp):
    with open(TIMESTAMP_FILE, 'w') as f:
        f.write(str(timestamp))


# 保存数据到CSV文件
def save_to_csv(data, output_file):
    file_exists = os.path.isfile(output_file)
    with open(output_file, mode="a", newline="") as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["timestamp", "symbol", "bids", "asks"])
        writer.writerow(data)



def spinning_cursor():
    while True:
        for cursor in '|/-\\':
            yield cursor


# WebSocket消息处理函数
async def on_message(message):
    data = json.loads(message)

    if 'data' in data:
        order_book = data['data'][0]
        timestamp = order_book['ts']
        bids = order_book['bids']
        asks = order_book['asks']

        # 保存到CSV文件
        save_to_csv([timestamp, SYMBOL, bids, asks], OUTPUT_FILE)

        # 保存最新的时间戳
        save_timestamp(timestamp)


# WebSocket连接函数
async def subscribe():
    last_timestamp = get_last_timestamp()

    # 打印一次抓取开始信息
    print("抓取正在进行...")

    # 旋转进度条生成器
    spinner = spinning_cursor()

    # 连接WebSocket并订阅
    async with websockets.connect(WS_URL) as ws:
        subscribe_message = {
            "op": "subscribe",
            "args": [{"channel": "books", "instId": SYMBOL, "depth": "5"}]
        }
        await ws.send(json.dumps(subscribe_message))

        # 如果有最后的时间戳，可以从该时间戳开始抓取
        if last_timestamp:
            print(f"Resuming from timestamp: {last_timestamp}")

        # 接收消息并调用on_message函数处理
        while True:
            message = await ws.recv()
            await on_message(message)

            # 更新进度条
            sys.stdout.write(f"\r抓取中... {next(spinner)}")
            sys.stdout.flush()


# 主函数
def main():
    asyncio.run(subscribe())


if __name__ == "__main__":
    main()
