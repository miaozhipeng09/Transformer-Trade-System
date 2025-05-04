import asyncio
import json
import websockets
import csv
import sys
import os

SYMBOL = "BTC-USDT"
WS_URL = "wss://ws.okx.com:8443/ws/v5/public"
OUTPUT_FILE = "order_book_data.csv"


# 保存数据到CSV文件
def save_to_csv(data, output_file):
    file_exists = os.path.isfile(output_file)
    with open(output_file, mode="a", newline="") as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["timestamp", "symbol", "bids", "asks"])  # 写入表头
        writer.writerow(data)  # 写入数据


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


# WebSocket连接函数
async def subscribe():
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
