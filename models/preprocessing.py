import pandas as pd
import ast

file_name = 'order_book_data.csv'

# 读取 CSV 文件
df = pd.read_csv(file_name)


def process_order_data(order_data):
    try:
        order_data = ast.literal_eval(order_data)  # 转换为列表形式

        # 检查是否为空数组，如果是，则返回空列表
        if not order_data:
            return [], []

        prices = [float(item[0]) for item in order_data]  # 提取价格
        volumes = [float(item[1]) for item in order_data]  # 提取挂单量
        return prices, volumes
    except Exception as e:
        print(f"Error processing order data: {e}")
        return [], []


# 删除空数组行并处理缺失值
df['bids'] = df['bids'].apply(lambda x: '[]' if x == '[]' else x)
df['asks'] = df['asks'].apply(lambda x: '[]' if x == '[]' else x)

df['bids'] = df['bids'].astype(str)
df['asks'] = df['asks'].astype(str)


# 函数：提取特征
def extract_features(row):
    bid_prices, bid_volumes = process_order_data(row['bids'])
    ask_prices, ask_volumes = process_order_data(row['asks'])

    # 特征计算
    features = {}

    # 提取最优买入价格、最优卖出价格、买卖差价
    if len(bid_prices) > 0 and len(ask_prices) > 0:
        features['best_bid'] = bid_prices[0]  # 最优买入价格
        features['best_ask'] = ask_prices[0]  # 最优卖出价格
        features['spread'] = ask_prices[0] - bid_prices[0]  # 买卖差价

    # 提取买入和卖出的总挂单量
    if len(bid_volumes) > 0:
        features['total_bid_volume'] = sum(bid_volumes)  # 总买入量
        features['best_bid_volume'] = bid_volumes[0]  # 最优买入挂单量

    if len(ask_volumes) > 0:
        features['total_ask_volume'] = sum(ask_volumes)  # 总卖出量
        features['best_ask_volume'] = ask_volumes[0]  # 最优卖出挂单量

    return features


# 使用 apply 函数提取每行的特征
df_features = df.apply(extract_features, axis=1)

# 转换为 DataFrame
features_df = pd.DataFrame(df_features.tolist())

# 合并原始数据和特征数据
final_df = pd.concat([df, features_df], axis=1)

# 删除任何含有空值的行
final_df = final_df.dropna()

# 保存最终的 CSV
final_df.to_csv('order_book_data_with_features_cleaned.csv', index=False)

print(final_df.head())  # 打印前几行，检查结果
