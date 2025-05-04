import pandas as pd
from sklearn.preprocessing import StandardScaler

# 读取已经清洗并保存的 CSV 文件
file_name = 'order_book_data_with_features_cleaned.csv'
df = pd.read_csv(file_name)


features_to_normalize = ['timestamp','best_bid', 'best_ask', 'spread', 'total_bid_volume', 'total_ask_volume',
                         'best_bid_volume','best_ask_volume']


missing_features = [col for col in features_to_normalize if col not in df.columns]
if missing_features:
    print(f"Warning: The following features are missing and won't be normalized: {missing_features}")

# 使用 StandardScaler 进行归一化
scaler = StandardScaler()

# 对需要归一化的列进行标准化
df[features_to_normalize] = scaler.fit_transform(df[features_to_normalize])

# 保存归一化后的数据到新的 CSV 文件
final_file_name = 'order_book_data_with_features_normalized.csv'
df.to_csv(final_file_name, index=False)

# 打印前几行查看结果
print(df.head())
