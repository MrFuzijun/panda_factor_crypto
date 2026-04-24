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
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.stats import norm
from scipy.stats import ttest_ind
from IPython.display import display

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'binance_futures')
DATA_TOP_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
RESULT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'factor_lib')

def cal_return(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算未来1/3/5/10/20天的回报率(加密货币无需复权)
    :param df: 单个币对的pd.DataFrame,需包含open列,按trade_date排序
    """
    df = df.sort_values(by='trade_date').reset_index(drop=True)
    df['1day_return'] = df['open'].shift(-2) / df['open'].shift(-1) - 1
    df['3day_return'] = df['open'].shift(-4) / df['open'].shift(-1) - 1
    df['5day_return'] = df['open'].shift(-6) / df['open'].shift(-1) - 1
    df['10day_return'] = df['open'].shift(-11) / df['open'].shift(-1) - 1
    df['20day_return'] = df['open'].shift(-21) / df['open'].shift(-1) - 1
    return df

def cal_pct_lag(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算币对滞后0-20天的收益率
    :param df: 待计算的pd.DataFrame
    """
    for i in range(0, 21):
        df[f'returns_lag{i}'] = df.groupby('symbol')['1day_return'].transform(lambda x: x.shift(-i))
    return df

def str_round(number: float, decimal_places: int, percentage: bool = False) -> str:
    """
    自定义更准确的四舍五入方法
    :param number: 待处理的数
    :param decimal_places: 需要保留的小数位数
    :param percentage: 是否百分比显示
    """
    if number == number:
        multiplier = 10 ** decimal_places
        rounded_number = int(number * multiplier + 0.5) / multiplier
        if percentage:
            rounded_number *= 100
            format_string = "{:." + str(decimal_places - 2) + "f}%"
        else:
            format_string = "{:." + str(decimal_places) + "f}"
        result_str = format_string.format(rounded_number)
        return result_str
    else:
        return np.nan

def read_crypto_kdata(start_date: str, end_date: str, symbol_list: list = None) -> pd.DataFrame:
    """
    读取加密货币k线数据(币安期货U本位合约日频数据)
    :param start_date: 数据开始日期
    :param end_date: 数据结束日期(包含)
    :param symbol_list: 需要读取的币对列表,为None时读取所有
    :return: 包含 trade_date,symbol,open,high,low,close,volume,quote_volume,trade_num,taker_buy_volume,taker_buy_quote_volume 的DataFrame
    """
    df_list = []

    if symbol_list is None:
        symbol_list = get_all_crypto_symbol()

    for symbol in symbol_list:
        file_path = os.path.join(DATA_PATH, f'{symbol}.csv')
        if not os.path.exists(file_path):
            continue
        df_temp = pd.read_csv(file_path)
        df_temp = df_temp[['trade_date', 'symbol', 'open', 'high', 'low', 'close',
                           'volume', 'quote_volume', 'trade_num',
                           'taker_buy_volume', 'taker_buy_quote_volume']]
        df_temp['trade_date'] = pd.to_datetime(df_temp['trade_date'], format='%Y-%m-%d')
        df_temp = df_temp[(df_temp['trade_date'] >= start_date) & (df_temp['trade_date'] <= end_date)]
        df_list.append(df_temp)

    if not df_list:
        return pd.DataFrame()

    df_kdata = pd.concat(df_list, ignore_index=True)
    df_kdata = df_kdata.sort_values(['trade_date', 'symbol']).reset_index(drop=True)
    return df_kdata

def get_all_crypto_symbol() -> list:
    """
    获取所有可用的加密货币交易对列表
    """
    symbol_list = []
    if not os.path.exists(DATA_PATH):
        return symbol_list
    for filename in os.listdir(DATA_PATH):
        if filename.endswith('.csv'):
            symbol_list.append(filename[:-4])
    return symbol_list

def read_crypto_kdata_top(start_date: str, end_date: str, top_n: int = 50) -> pd.DataFrame:
    """
    读取经过流动性筛选的加密货币k线数据(从prepare_data生成的parquet文件)
    :param start_date: 数据开始日期
    :param end_date: 数据结束日期(包含)
    :param top_n: 流动性筛选的TOP N (10/20/50/100/200)
    :return: DataFrame
    """
    parquet_path = os.path.join(DATA_TOP_PATH, f'binance_futures_top{top_n}.parquet')
    if os.path.exists(parquet_path):
        df_kdata = pd.read_parquet(parquet_path)
        df_kdata['trade_date'] = pd.to_datetime(df_kdata['trade_date'])
        df_kdata = df_kdata[(df_kdata['trade_date'] >= start_date) & (df_kdata['trade_date'] <= end_date)]
        df_kdata = df_kdata.sort_values(['trade_date', 'symbol']).reset_index(drop=True)
        return df_kdata
    else:
        print(f'文件不存在: {parquet_path}')
        print(f'请先运行 prepare_data.py 生成流动性筛选数据')
        return pd.DataFrame()

def ext_out_mad(group: pd.DataFrame, factor_list: list) -> pd.DataFrame:
    """
    中位数去极值(MAD法)
    :param group: 每天的df因子数据
    :param factor_list: 需要处理的因子名称列表
    """
    for f in factor_list:
        factor = group[f]
        median = factor.median()
        mad = (factor - median).abs().median()
        edge_up = median + 3 * mad
        edge_low = median - 3 * mad
        factor.clip(lower=edge_low, upper=edge_up, inplace=True)
        group[f] = factor
    return group

def ext_out_3std(group: pd.DataFrame, factor_list: list) -> pd.DataFrame:
    """
    3倍标准差去极值
    :param group: 每天的df因子数据
    :param factor_list: 需要处理的因子名称列表
    """
    for f in factor_list:
        factor = group[f]
        edge_up = factor.mean() + 3 * factor.std()
        edge_low = factor.mean() - 3 * factor.std()
        factor.clip(lower=edge_low, upper=edge_up, inplace=True)
        group[f] = factor
    return group

def volume_neutralization(group: pd.DataFrame, factor_list: list) -> pd.DataFrame:
    """
    成交额对数中性化(用quote_volume替代A股的total_mv)
    :param group: 每天的df因子数据
    :param factor_list: 需要处理的因子名称列表
    """
    valid_mask = group['quote_volume'] > 0
    if valid_mask.sum() < 3:
        return group
    for f in factor_list:
        valid = valid_mask & group[f].notna()
        if valid.sum() < 3:
            continue
        X = group.loc[valid, 'quote_volume'].apply(np.log)
        y = group.loc[valid, f]
        X = sm.add_constant(X)
        try:
            model = sm.OLS(y, X).fit()
            group.loc[valid, f] = model.resid
        except np.linalg.LinAlgError:
            pass
    return group

def z_score(group: pd.DataFrame, factor_list: list) -> pd.DataFrame:
    """
    z_score标准化
    :param group: 每天的df因子数据
    :param factor_list: 需要处理的因子名称列表
    """
    for f in factor_list:
        factor = group[f]
        if factor.std() != 0:
            group[f] = (factor - factor.mean()) / factor.std()
        else:
            group[f] = np.nan
    return group

def clean_df(df: pd.DataFrame, factor_list: list) -> pd.DataFrame:
    """
    清洗数据(包含剔除回报率缺失、因子值缺失)
    :param df: 待处理的df因子数据
    :param factor_list: 需要处理的因子名称列表
    """
    df = df[df[factor_list].notna().all(axis=1)]
    df = df[df['20day_return'].notna()]
    return df

def grouping_factor(df: pd.DataFrame, factor_list: list) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    为因子数据进行截面分组
    为包含因子值数据的df分组,记录组号,同时记录每天的市场涨幅均值
    :param df: 待处理的每天的df因子数据
    :param factor_list: 需要处理的因子名称列表
    :return: 返回tuple元组(包含组号的df数据新列名为'{factor_name}_group',记录每天的市场涨幅均值的df)
    """
    benchmark_pct = {}
    grouped_dfs = []
    for date, group in df.groupby('trade_date'):
        benchmark_pct_child = {}
        benchmark_pct_child['1D_m'] = group['1day_return'].mean()
        benchmark_pct_child['3D_m'] = group['3day_return'].mean()
        benchmark_pct_child['5D_m'] = group['5day_return'].mean()
        benchmark_pct_child['10D_m'] = group['10day_return'].mean()
        benchmark_pct_child['20D_m'] = group['20day_return'].mean()
        benchmark_pct[date] = benchmark_pct_child

        if group.empty:
            continue

        new_group = group.copy()
        for f in factor_list:
            new_group[f'{f}_group'] = np.nan

        for f in factor_list:
            if group[f].dropna().nunique() < 10:
                print(f"因子{f},{date},分组小于10,已跳过")
                continue
            try:
                new_group[f'{f}_group'] = pd.qcut(group[f].dropna(), 10, labels=range(1, 11), duplicates='drop')
            except ValueError:
                print(f"因子{f},{date},分箱失败,已跳过")
                continue

        grouped_dfs.append(new_group)

    df_cuted = pd.concat(grouped_dfs) if grouped_dfs else pd.DataFrame()
    df_benchmark_pct = pd.DataFrame(benchmark_pct).T
    return df_cuted, df_benchmark_pct

class factor():
    def __init__(self, name: str) -> None:
        self.name: str = name

        self.period: int = 5
        self.predict_direction = 0
        self.commission = 0.002
        self.mode = 1

        self.df_pnl = pd.DataFrame()
        self.df_stock = pd.DataFrame()
        self.df_turnover = pd.DataFrame()
        self.df_ic = pd.DataFrame()

        self.df_group = pd.DataFrame()

        self.df_info = pd.DataFrame(index=[f'分组{i}' for i in range(1, 11)] + ['多空组合', '多空组合2'],
                                    columns=['年化收益率', '超额年化', '最大回撤', '超额最大回撤', '年化波动', '超额年化波动',
                                             '换手率', '月度胜率', '超额月度胜率', '跟踪误差', '夏普比率', '信息比率'])
        self.df_info2 = pd.DataFrame(index=[self.name],
                                     columns=['IC_mean', 'Rank_IC', 'IC_std', 'IC_IR', 'IR',
                                              'P(IC<-0.02)', 'P(IC>0.02)', 't统计量', 'p-value', '单调性']).T

    def set_backtest_parameters(self, period: int, predict_direction: int = 0,
                                commission: float = 0.002, mode: int = 1) -> None:
        """
        设置因子回测的相关参数
        :param period: 回测周期
        :param predict_direction: 预测方向(0为因子值越小越好,IC为负/1为因子值越大越好,IC为正)
        :param commission: 手续费+滑点(默认为千2)
        :param mode: 0为简单回测,1为全面回测(默认为1)
        """
        self.period = period
        self.predict_direction = predict_direction
        self.commission = commission
        self.mode = mode

    def cal_turnover_rate(self) -> None:
        """
        计算各个分组换手率,结果保存到self.df_turnover中
        """
        if self.df_stock.empty:
            return

        column_list = []
        for n in range(1, 11):
            column_list.append(f'group{n}_day_turnover')
        self.df_turnover = pd.DataFrame(index=self.df_stock.index, columns=column_list)

        for i in range(0, self.df_stock.shape[0]):
            if i < self.period:
                continue

            for n in range(0, 10):
                prev_stock = self.df_stock.iloc[i - self.period, n]
                today_stock = self.df_stock.iloc[i, n]

                if not prev_stock or not today_stock:
                    self.df_turnover.iloc[i, n] = np.nan
                else:
                    prev_stock_set = set(prev_stock)
                    today_stock_set = set(today_stock)

                    changed_stock_num = len(today_stock_set - prev_stock_set)
                    turnover_rate = changed_stock_num / len(prev_stock_set)
                    self.df_turnover.iloc[i, n] = turnover_rate

    def cal_df_stock(self, df: pd.DataFrame) -> None:
        stock_dict = {}
        for date, group in df.groupby('trade_date'):
            stock_child_dict = {}
            for num, temp in group.groupby(f'{self.name}_group'):
                stock_child_dict[f'group{num}_code'] = temp['symbol'].tolist()

            stock_dict[date] = stock_child_dict

        df_stock = pd.DataFrame(stock_dict).T
        self.df_stock = df_stock

    def show_df_info(self, types: int = 0) -> None:
        """
        显示多空分组、IC各类统计信息
        :param types: 0为多空分组年化、夏普等统计指标矩阵,1为因子IC各项评估信息
        """
        if self.df_info.empty:
            print('统计指标数据缺失!')
            return

        if types == 0:
            display(self.df_info)
            return self.df_info
        else:
            display(self.df_info2)
            return self.df_info2

    def cal_df_info1(self) -> None:
        for i in range(1, 13):
            if i < 11:
                group_return = self.df_pnl[f'group{i}_pnl']
                group_pro = self.df_pnl[f'group{i}_pro']
                self.df_info.iloc[i - 1]['换手率'] = str_round(self.df_turnover.iloc[:, i - 1].mean(), 4, True)
            elif i == 11:
                group_return = self.df_pnl['group_ls']
                group_pro = self.df_pnl['group_ls_pro']
                self.df_info.iloc[i - 1]['换手率'] = str_round(
                    (self.df_turnover.iloc[:, 0].mean() + self.df_turnover.iloc[:, 9].mean()) / 2, 4, True)
            elif i == 12:
                group_return = self.df_pnl['group_ls_2']
                group_pro = self.df_pnl['group_ls_2_pro']
                self.df_info.iloc[i - 1]['换手率'] = str_round(
                    (self.df_turnover.iloc[:, 1].mean() + self.df_turnover.iloc[:, 8].mean()) / 2, 4, True)

            annualized_return = np.mean(group_return) * (365 / self.period)
            self.df_info.iloc[i - 1]['年化收益率'] = str_round(annualized_return, 4, True)

            annualized_volatility = np.std(group_return) * np.sqrt(365 / self.period)
            self.df_info.iloc[i - 1]['年化波动'] = str_round(annualized_volatility, 4, True)

            annualized_pro_volatility = np.std(group_pro) * np.sqrt(365 / self.period)
            self.df_info.iloc[i - 1]['超额年化波动'] = str_round(annualized_pro_volatility, 4, True)

            cumulative_returns = np.cumsum(np.array(group_return))
            max_drawdown = np.max(np.maximum.accumulate(cumulative_returns) - cumulative_returns)
            self.df_info.iloc[i - 1]['最大回撤'] = str_round(max_drawdown, 4, True)

            cumulative_pro_returns = np.cumsum(np.array(group_pro))
            max_drawdown = np.max(np.maximum.accumulate(cumulative_pro_returns) - cumulative_pro_returns)
            self.df_info.iloc[i - 1]['超额最大回撤'] = str_round(max_drawdown, 4, True)

            sharpe_ratio = (annualized_return - 0.025) / annualized_volatility
            self.df_info.iloc[i - 1]['夏普比率'] = str_round(sharpe_ratio, 4)

            excess_annualized_return = group_pro.mean() * (365 / self.period)
            tracking_error = np.sqrt(np.sum(group_pro ** 2) / (len(group_pro) - 1))
            self.df_info.iloc[i - 1]['跟踪误差'] = str_round(tracking_error, 4)

            self.df_info.iloc[i - 1]['超额年化'] = str_round(excess_annualized_return, 4, True)

            monthly_return = group_return.groupby(pd.Grouper(freq='M')).sum()
            win_rate = monthly_return[monthly_return > 0].count() / len(monthly_return)
            self.df_info.iloc[i - 1]['月度胜率'] = str_round(win_rate, 4, True)

            monthly_pro_return = group_pro.groupby(pd.Grouper(freq='M')).sum()
            win_pro_rate = monthly_pro_return[monthly_pro_return > 0].count() / len(monthly_pro_return)
            self.df_info.iloc[i - 1]['超额月度胜率'] = str_round(win_pro_rate, 4, True)

            IR = excess_annualized_return / annualized_pro_volatility
            self.df_info.iloc[i - 1]['信息比率'] = str_round(IR, 4)

            factor_path = os.path.join(RESULT_PATH, self.name)
            os.makedirs(factor_path, exist_ok=True)
            self.df_info.to_csv(os.path.join(factor_path, '多空分组统计指标.csv'))

    def cal_df_info2(self) -> None:
        ic_value = self.df_ic['ic']
        ir_value = self.df_ic['ir']
        self.df_info2.loc['IC_mean', self.name] = str_round(ic_value.mean(), 4)
        self.df_info2.loc['Rank_IC', self.name] = str_round(self.df_ic['rank_ic'].mean(), 4)
        self.df_info2.loc['IC_std', self.name] = str_round(ic_value.std(), 4)
        self.df_info2.loc['IC_IR', self.name] = str_round(ic_value.mean() / ic_value.std(), 4)
        self.df_info2.loc['IR', self.name] = str_round(ir_value.mean(), 4)
        self.df_info2.loc['P(IC>0.02)', self.name] = str_round(np.sum((ic_value) > 0.02) / len(ic_value), 4, True)
        self.df_info2.loc['P(IC<-0.02)', self.name] = str_round(np.sum((ic_value) < -0.02) / len(ic_value), 4, True)

        t_stat, p_value = ttest_ind(ic_value, np.zeros(len(ic_value)))
        self.df_info2.loc['t统计量', self.name] = str_round(t_stat, 4)
        self.df_info2.loc['p-value', self.name] = str_round(p_value, 4)

        ann_return_rankings = self.df_info.iloc[0:10]['年化收益率'].str.replace('%', '').astype(float).reset_index(
            drop=True)
        self.df_info2.loc['单调性', self.name] = str_round(
            abs(ann_return_rankings.corr(pd.Series(np.arange(1, 11)))), 2)
        factor_path = os.path.join(RESULT_PATH, self.name)
        os.makedirs(factor_path, exist_ok=True)
        self.df_info2.to_csv(os.path.join(factor_path, 'IC统计指标.csv'))

    def start_backtest(self, df: pd.DataFrame, df_benchmark_pct: pd.DataFrame) -> None:
        """
        参数设置好后就可以开始回测了
        :param df: 整理好的因子和k线数据dataframe
        """
        if df.empty or self.name not in df.columns:
            print('回测数据缺失!')
            return

        self.cal_df_stock(df)
        self.cal_turnover_rate()

        pnl_dict = {}
        ic_dict = {}
        group_pct = {}

        day_count = 0
        for date, group in df.groupby('trade_date'):
            if group.empty:
                continue

            if group[self.name].dropna().nunique() < 10:
                print(f"因子{self.name},{date},分组小于10,已跳过")
                continue

            group_child_pct = {}
            for n in range(1, 11):
                for d in [1, 3, 5, 10, 20]:
                    group_child_pct[f'group{n}_return{d}'] = (
                            group[group[f'{self.name}_group'] == n][f'{d}day_return'].mean()) / d
                    group_child_pct[f'group{n}_std{d}'] = (
                            group[group[f'{self.name}_group'] == n][f'{d}day_return'].std()) / d

            group_pct[date] = group_child_pct

            if day_count % self.period != 0:
                day_count += 1
                continue
            day_count += 1

            pnl_child_dict = {}
            ic_child_dict = {}

            ic_child_dict['ic'] = group[self.name].corr(group[f'{self.period}day_return'])
            ic_child_dict['rank_ic'] = group[self.name].rank().corr(
                group[f'{self.period}day_return'].rank())
            ic_child_dict['ir'] = ic_child_dict['ic'] * np.power(group.shape[0], 0.5)
            ic_child_dict['ic_lag0'] = ic_child_dict['rank_ic']
            for i in range(1, 21):
                ic_child_dict[f'ic_lag{i}'] = group[self.name].rank().corr(
                    group[f'returns_lag{i}'].rank())

            ic_dict[date] = ic_child_dict

            for n in range(1, 11):
                return_mean = group[group[f'{self.name}_group'] == n][f'{self.period}day_return'].mean()
                pnl_child_dict[f'group{n}_pnl'] = return_mean

            return_benchmark = df_benchmark_pct.loc[date, f'{self.period}D_m']
            pnl_child_dict['return_benchmark'] = return_benchmark

            pnl_dict[date] = pnl_child_dict

        df_pnl = pd.DataFrame(pnl_dict).T
        for n in range(1, 13):
            if n < 11:
                df_pnl[f'group{n}_pro'] = df_pnl[f'group{n}_pnl'] - df_pnl['return_benchmark']

            elif n == 11:
                if self.predict_direction == 0:
                    df_pnl['group_ls'] = df_pnl[f'group1_pnl'] - df_pnl[f'group10_pnl']
                    df_pnl['group_ls_pro'] = df_pnl[f'group1_pro'] - df_pnl[f'group10_pro']
                else:
                    df_pnl['group_ls'] = df_pnl[f'group10_pnl'] - df_pnl[f'group1_pnl']
                    df_pnl['group_ls_pro'] = df_pnl[f'group10_pro'] - df_pnl[f'group1_pro']
            elif n == 12:
                if self.predict_direction == 0:
                    df_pnl['group_ls_2'] = df_pnl[f'group2_pnl'] - df_pnl[f'group9_pnl']
                    df_pnl['group_ls_2_pro'] = df_pnl[f'group2_pro'] - df_pnl[f'group9_pro']
                else:
                    df_pnl['group_ls_2'] = df_pnl[f'group9_pnl'] - df_pnl[f'group2_pnl']
                    df_pnl['group_ls_2_pro'] = df_pnl[f'group9_pro'] - df_pnl[f'group2_pro']

        self.df_pnl = df_pnl

        df_ic = pd.DataFrame(ic_dict).T
        self.df_ic = df_ic

        df_group = pd.DataFrame(group_pct).T
        self.df_group = df_group

        self.cal_df_info1()
        self.cal_df_info2()

    def draw_pct(self) -> None:
        """
        画多空分组绝对收益图和超额收益图
        """
        if self.df_pnl.empty:
            print('收益率数据缺失!')
            return

        colors = [plt.cm.coolwarm(i) for i in np.linspace(0, 1, 10)]
        fig, axes = plt.subplots(1, 2, figsize=(22, 10), dpi=200)

        for ax in axes:
            ax.set_box_aspect(1 / 1.1)

        for i in range(1, 11):
            axes[0].plot(self.df_pnl.index, self.df_pnl[f'group{i}_pnl'].cumsum(),
                         label=f'group {i}', color=colors[i - 1], linewidth=1.5)

        if self.predict_direction == 0:
            axes[0].plot(self.df_pnl.index, self.df_pnl['group_ls'].cumsum(),
                         label=f'group 1-10', color='black', linestyle='--', linewidth=1.5)
            axes[0].plot(self.df_pnl.index, self.df_pnl['group_ls_2'].cumsum(),
                         label=f'group 2-9', color='purple', linestyle='--', linewidth=1.5)
        else:
            axes[0].plot(self.df_pnl.index, self.df_pnl['group_ls'].cumsum(),
                         label=f'group 10-1', color='black', linestyle='--', linewidth=1.5)
            axes[0].plot(self.df_pnl.index, self.df_pnl['group_ls_2'].cumsum(),
                         label=f'group 9-2', color='purple', linestyle='--', linewidth=1.5)

        axes[0].set_title(f'{self.name} 10 groups return')

        axes[0].legend(loc='upper left', prop={'size': 10}, ncol=2)
        axes[0].grid(True)

        for i in range(1, 11):
            axes[1].plot(self.df_pnl.index, self.df_pnl[f'group{i}_pro'].cumsum(),
                         label=f'group {i}', color=colors[i - 1], linewidth=1.5)

        if self.predict_direction == 0:
            axes[1].plot(self.df_pnl.index, self.df_pnl['group_ls_pro'].cumsum(),
                         label=f'group 1-10', color='black', linestyle='--', linewidth=1.5)
            axes[1].plot(self.df_pnl.index, self.df_pnl['group_ls_2_pro'].cumsum(),
                         label=f'group 2-9', color='purple', linestyle='--', linewidth=1.5)
        else:
            axes[1].plot(self.df_pnl.index, self.df_pnl['group_ls_pro'].cumsum(),
                         label=f'group 10-1', color='black', linestyle='--', linewidth=1.5)
            axes[1].plot(self.df_pnl.index, self.df_pnl['group_ls_2_pro'].cumsum(),
                         label=f'group 9-2', color='purple', linestyle='--', linewidth=1.5)

        axes[1].set_title(f'{self.name} 10 groups excess return')
        axes[1].legend(loc='upper left', prop={'size': 10}, ncol=2)
        axes[1].grid(True)

        axes[0].yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))
        axes[1].yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))
        plt.tight_layout()
        factor_path = os.path.join(RESULT_PATH, self.name)
        os.makedirs(factor_path, exist_ok=True)
        plt.savefig(os.path.join(factor_path, '分组收益图.png'), dpi=200)
        plt.show()

    def draw_ic(self, types: int = 0) -> None:
        """
        画IC时序图和IC密度图
        :param types: IC时序图类型(0为normal_IC,1为rank_IC)
        """
        if self.df_ic.empty:
            print('IC序列数据缺失!')
            return

        fig, axes = plt.subplots(1, 2, figsize=(22, 10), dpi=200)

        if types:
            ic_value = self.df_ic['rank_ic']
            mu = np.mean(ic_value)
            sigma = np.std(ic_value)
            axes[0].set_title(f'{self.name} Rank_IC={str_round(mu, 3)} IC_IR={str_round((mu / sigma), 4)}')
        else:
            ic_value = self.df_ic['ic']
            mu = np.mean(ic_value)
            sigma = np.std(ic_value)
            axes[0].set_title(f'{self.name} IC={str_round(mu, 3)} IC_IR={str_round((mu / sigma), 4)}')

        for date, value in ic_value.items():
            if value > 0:
                axes[0].bar(date, value, color='red')
            else:
                axes[0].bar(date, value, color=(31 / 255, 119 / 255, 180 / 255))
        axes[0].set_ylabel('IC')
        axes[0].set_xlabel('Date')
        axes[0].set_ylim([ic_value.min(), ic_value.max()])

        ax2 = axes[0].twinx()
        ax2.plot(self.df_ic.index, self.df_ic['ic'].cumsum(), color='black')
        ax2.set_ylabel('cum_IC')

        axes[1].set_title(
            f'{self.name} IC distribution skew={str_round(ic_value.skew(), 3)} kurt={str_round(ic_value.kurt(), 3)}')
        n, bins, patches = axes[1].hist(x=ic_value, bins=60, density=True,
                                         color=(138 / 255, 188 / 255, 220 / 255), edgecolor='black')
        y = norm.pdf(bins, mu, sigma)
        axes[1].plot(bins, y, '--', color='red')
        axes[1].set_ylabel('Probability')
        axes[1].set_xlabel('IC')

        plt.tight_layout()
        factor_path = os.path.join(RESULT_PATH, self.name)
        os.makedirs(factor_path, exist_ok=True)
        plt.savefig(os.path.join(factor_path, 'IC图.png'), dpi=200)
        plt.show()

    def draw_ic_dacay(self) -> None:
        """
        画IC衰减图和IC自相关图
        """
        if self.df_ic.empty:
            print('IC序列数据缺失!')
            return

        fig, axes = plt.subplots(1, 2, figsize=(22, 10), dpi=200)

        ic_decay_cols = [f'ic_lag{i}' for i in range(0, 21)]
        ic_decay = self.df_ic[ic_decay_cols]

        for i, col in enumerate(ic_decay_cols):
            ic_mean = ic_decay[col].mean()
            if i == 0:
                axes[0].bar(i, ic_mean, color='red', edgecolor='black')
                axes[0].text(i, ic_mean, str_round(ic_mean, 3), ha='center', va='bottom', fontsize=10)
            else:
                axes[0].bar(i, ic_mean, color=(31 / 255, 119 / 255, 180 / 255), edgecolor='black')
                if ic_mean >= 0:
                    axes[0].text(i, ic_mean, str_round(ic_mean, 3), ha='center', va='bottom', fontsize=10)
                else:
                    axes[0].text(i, ic_mean, str_round(ic_mean, 3), ha='center', va='top', fontsize=10)

        axes[0].set_title('IC衰减图')
        axes[0].set_ylabel('Rank_IC')
        axes[0].set_xlabel('滞后期数')
        axes[0].set_xticks([i for i in range(0, 21)])

        sm.graphics.tsa.plot_acf(self.df_ic['ic'], lags=40, alpha=0.05, ax=axes[1])
        axes[1].set_xlabel('滞后期数')
        axes[1].set_title("ACF with 95% Confidence Intervals")

        plt.tight_layout()
        factor_path = os.path.join(RESULT_PATH, self.name)
        os.makedirs(factor_path, exist_ok=True)
        plt.savefig(os.path.join(factor_path, 'IC衰减图.png'), dpi=200)
        plt.show()

    def draw_group_pct(self) -> None:
        """
        画不同分组不同预测周期下的平均收益率、标准差情况
        """
        if self.df_group.empty:
            print('分组df数据缺失!')
            return

        fig, axes = plt.subplots(2, 1, figsize=(22, 10), dpi=200)

        barWidth = 0.1

        axes[0].set_ylabel('日平均收益率')
        axes[0].set_xlabel('持有周期(1/3/5/10/20)')
        axes[0].set_title('分组-持有周期日均收益图')

        for i in range(1, 11):
            mena1 = self.df_group[f'group{i}_return1'].mean()
            axes[0].bar(i - barWidth * 2, mena1, color=(31 / 255, 119 / 255, 180 / 255),
                        width=barWidth, edgecolor='black')
            if mena1 >= 0:
                axes[0].text(i - barWidth * 2, mena1, str_round(mena1, 4, True),
                             ha='center', va='bottom', fontsize=10)
            else:
                axes[0].text(i - barWidth * 2, mena1, str_round(mena1, 4, True),
                             ha='center', va='top', fontsize=10)

            mena3 = self.df_group[f'group{i}_return3'].mean()
            axes[0].bar(i - barWidth, mena3, color=(31 / 255, 119 / 255, 180 / 255),
                        width=barWidth, edgecolor='black')

            mena5 = self.df_group[f'group{i}_return5'].mean()
            axes[0].bar(i, mena5, color=(31 / 255, 119 / 255, 180 / 255),
                        width=barWidth, edgecolor='black')
            if mena1 >= 0:
                axes[0].text(i, mena5, str_round(mena5, 4, True), ha='center', va='bottom', fontsize=10)
            else:
                axes[0].text(i, mena5, str_round(mena5, 4, True), ha='center', va='top', fontsize=10)

            mena10 = self.df_group[f'group{i}_return10'].mean()
            axes[0].bar(i + barWidth, mena10, color=(31 / 255, 119 / 255, 180 / 255),
                        width=barWidth, edgecolor='black')

            mena20 = self.df_group[f'group{i}_return20'].mean()
            axes[0].bar(i + barWidth * 2, mena20, color=(31 / 255, 119 / 255, 180 / 255),
                        width=barWidth, edgecolor='black')
            if mena1 >= 0:
                axes[0].text(i + barWidth * 2, mena20, str_round(mena20, 4, True),
                             ha='center', va='bottom', fontsize=10)
            else:
                axes[0].text(i + barWidth * 2, mena20, str_round(mena20, 4, True),
                             ha='center', va='top', fontsize=10)

        axes[0].grid(True)
        axes[0].yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=2))
        axes[0].yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=2))
        axes[0].set_xticks([i for i in range(1, 11)])
        axes[0].set_xticklabels([f'Group {i + 1}' for i in range(10)])

        axes[1].set_ylabel('日平均标准差')
        axes[1].set_xlabel('持有周期(1/3/5/10/20)')
        axes[1].set_title('分组-持有周期收益标准差')

        for i in range(1, 11):
            std1 = self.df_group[f'group{i}_std1'].mean()
            axes[1].bar(i - barWidth * 2, std1, color=(31 / 255, 119 / 255, 180 / 255),
                        width=barWidth, edgecolor='black')
            if std1 >= 0:
                axes[1].text(i - barWidth * 2, std1, str_round(std1, 2),
                             ha='center', va='bottom', fontsize=10)
            else:
                axes[1].text(i - barWidth * 2, std1, str_round(std1, 2),
                             ha='center', va='top', fontsize=10)

            std3 = self.df_group[f'group{i}_std3'].mean()
            axes[1].bar(i - barWidth, std3, color=(31 / 255, 119 / 255, 180 / 255),
                        width=barWidth, edgecolor='black')

            std5 = self.df_group[f'group{i}_std5'].mean()
            axes[1].bar(i, std5, color=(31 / 255, 119 / 255, 180 / 255),
                        width=barWidth, edgecolor='black')
            if std5 >= 0:
                axes[1].text(i, std5, str_round(std5, 2), ha='center', va='bottom', fontsize=10)
            else:
                axes[1].text(i, std5, str_round(std5, 2), ha='center', va='top', fontsize=10)

            std10 = self.df_group[f'group{i}_std10'].mean()
            axes[1].bar(i + barWidth, std10, color=(31 / 255, 119 / 255, 180 / 255),
                        width=barWidth, edgecolor='black')

            std20 = self.df_group[f'group{i}_std20'].mean()
            axes[1].bar(i + barWidth * 2, std20, color=(31 / 255, 119 / 255, 180 / 255),
                        width=barWidth, edgecolor='black')
            if std1 >= 0:
                axes[1].text(i + barWidth * 2, std20, str_round(std20, 2),
                             ha='center', va='bottom', fontsize=10)
            else:
                axes[1].text(i + barWidth * 2, std20, str_round(std20, 2),
                             ha='center', va='top', fontsize=10)

        axes[1].grid(True)
        axes[1].set_xticks([i for i in range(1, 11)])
        axes[1].set_xticklabels([f'Group {i + 1}' for i in range(10)])

        plt.tight_layout()
        factor_path = os.path.join(RESULT_PATH, self.name)
        os.makedirs(factor_path, exist_ok=True)
        plt.savefig(os.path.join(factor_path, '分组持有周期图.png'), dpi=200)
        plt.show()

    def __str__(self) -> str:
        self.draw_pct()
        self.draw_ic()
        self.draw_ic_dacay()
        self.draw_group_pct()
        self.show_df_info(0)
        self.show_df_info(1)
        return ''


if __name__ == '__main__':
    pass
