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
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import statsmodels.api as sm
from factor_func import *
from factor_func import factor
from scipy.stats import norm
from pandas.plotting import table
import winsound

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
pd.set_option('display.max_rows', 200)
pd.set_option('display.max_columns', 20)


start_date = '2017-01-01'
end_date = '2026-4-23'

# 读取加密货币k线数据(币安期货U本位合约)
# 流动性筛选: TOP10 / TOP20 / TOP50 / TOP100 / TOP200 (需先运行prepare_data.py)
top_n = 200
df_kdata = read_crypto_kdata_top(start_date, end_date, top_n)
df_kdata = df_kdata.groupby('symbol', group_keys=False).apply(cal_return)


# 定义因子列表 - 基于币安期货数据可以计算的因子
# 例如: 使用close作为因子(收盘价因子)
df_kdata['close'] = df_kdata['close']
factor_list = ['close']
factor_list


# 因子间相关性,如果factor_list不止一个因子的话
if len(factor_list) > 1:
    factor_corr = df_kdata[factor_list].corr()
    display(factor_corr)


df = df_kdata.copy()
df = clean_df(df, factor_list)
df_kdata_clean = 0


df = df.groupby('trade_date', group_keys=False).apply(ext_out_3std, factor_list)  # 3倍标准差去极值
df = df.groupby('trade_date', group_keys=False).apply(z_score, factor_list)  # z-score标准化


if len(factor_list) == 1:
    fig_num = 2
else:
    fig_num = len(factor_list)
fig, axes = plt.subplots(1, fig_num, figsize=(22, 10), dpi=200)
for ax in axes:
    ax.set_box_aspect(1/1.1)
count = 0
for f in factor_list:
    mu = np.mean(df[f])
    sigma = np.std(df[f])
    axes[count].set_title('factor distribution')
    n, bins, patches = axes[count].hist(x=df[f], bins=60, density=True, color=(138/255,188/255,220/255), edgecolor='black')
    y = norm.pdf(bins, mu, sigma)
    axes[count].plot(bins, y, color=(61/255,109/255,141/255))
    axes[count].axvline(mu, color='black', linestyle='--')
    axes[count].set_ylabel('Density')
    axes[count].set_xlabel(f'{f}')
    count += 1

plt.tight_layout()
plt.show()


df = cal_pct_lag(df)

df_cuted, df_benchmark = grouping_factor(df, factor_list)

# 使用BTC作为基准指数(替代A股沪深300)
df_btc = read_crypto_kdata(start_date, end_date, symbol_list=['BTCUSDT'])
df_btc = df_btc.sort_values('trade_date').reset_index(drop=True)
df_btc['1D_m'] = (df_btc['open'].shift(-2) / df_btc['open'].shift(-1)) - 1
df_btc['3D_m'] = (df_btc['open'].shift(-4) / df_btc['open'].shift(-1)) - 1
df_btc['5D_m'] = (df_btc['open'].shift(-6) / df_btc['open'].shift(-1)) - 1
df_btc['10D_m'] = (df_btc['open'].shift(-11) / df_btc['open'].shift(-1)) - 1
df_btc['20D_m'] = (df_btc['open'].shift(-21) / df_btc['open'].shift(-1)) - 1
df_btc = df_btc.set_index('trade_date')[['1D_m', '3D_m', '5D_m', '10D_m', '20D_m']]
df_benchmark = df_btc


factor_obj_list = []
"""
:param period: 回测周期
:param predict_direction: 预测方向(0为因子值越小越好,IC为负/1为因子值越大越好,IC为正)
:param commission: 手续费+滑点(默认为千2)
:param mode: 0为简单回测,1为全面回测(默认为1)
"""
for f in factor_list:
    factor_obj = factor(f)
    factor_obj_list.append(factor_obj)
    factor_obj.set_backtest_parameters(period=5, predict_direction=1, commission=0.0003,mode=1)
    factor_obj.start_backtest(df_cuted, df_benchmark)


for f in factor_obj_list:
    print(f)

winsound.MessageBeep()

