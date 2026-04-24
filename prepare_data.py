"""
原创作者：PandaAI-Tech （pandaai)
二创作者:fuzijun
全网搜甫子君联系商用授权，最终授权必须由pandaai授权给予商用；
授权流程：提交作者商用授权-->提交主办方商用授权-->作者授权成功-->主办方pandaai授权成功-->给予商用授权。
侵权必究！！！
"""
import pandas as pd
import numpy as np
import os
import gc

data_path = './data/binance_futures'
output_path = './data/binance_futures_merged.parquet'

print("正在读取所有CSV文件...")

all_data = []
files = [f for f in os.listdir(data_path) if f.endswith('.csv')]
print(f"共 {len(files)} 个文件")

for i, f in enumerate(files):
    if i % 100 == 0:
        print(f"进度: {i}/{len(files)}")
    try:
        df = pd.read_csv(os.path.join(data_path, f))
        all_data.append(df)
    except Exception as e:
        print(f"读取 {f} 失败: {e}")

print("正在合并数据...")
df_all = pd.concat(all_data, ignore_index=True)
del all_data
gc.collect()

df_all['trade_date'] = pd.to_datetime(df_all['trade_date'])
df_all['turnover'] = df_all['volume'] * df_all['close']

print(f"总数据量: {len(df_all)}")
print(f"日期范围: {df_all['trade_date'].min()} ~ {df_all['trade_date'].max()}")

def process_top_n(df_all, top_n, threshold, output_dir='./data'):
    """
    按流动性筛选并取每天成交额前N的币对,保存为parquet
    :param df_all: 全量数据
    :param top_n: 每天取前N个币对
    :param threshold: 最低成交额门槛
    :param output_dir: 输出目录
    """
    df_filtered = df_all[df_all['turnover'] > threshold].copy()
    print(f"交易额>{threshold/1e6:.0f}万的数据量: {len(df_filtered)}")

    df_filtered = df_filtered.sort_values(['trade_date', 'turnover'], ascending=[True, False])

    top_list = []
    for date, group in df_filtered.groupby('trade_date'):
        top = group.head(top_n)
        top_list.append(top)

    df_top = pd.concat(top_list, ignore_index=True)

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'binance_futures_top{top_n}.parquet')
    df_top.to_parquet(output_path, index=False)

    print(f"TOP{top_n} 已保存到: {output_path}")
    print(f"- 总行数: {len(df_top):,}")
    print(f"- 币种数: {df_top['symbol'].nunique()}")
    print(f"- 日期数: {df_top['trade_date'].nunique()}")
    print(f"- 平均每天币种数: {len(df_top) / df_top['trade_date'].nunique():.1f}")
    print()
    return df_top


threshold = 10_000_000

for top_n in [5, 10, 20, 50, 100, 200]:
    print(f"{'='*40}")
    print(f"正在处理 TOP{top_n}...")
    process_top_n(df_all, top_n, threshold)
