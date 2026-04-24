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
import requests
import time
from datetime import datetime

PROXY = {"http": "http://127.0.0.1:7890", "https": "http://127.0.0.1:7890"}
MAX_RETRIES = 3
RETRY_DELAY = 2

def fetch_with_retry(url, params=None, headers=None, timeout=30):
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(url, params=params, headers=headers, proxies=PROXY, timeout=timeout)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"第 {attempt + 1} 次请求失败: {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
            else:
                raise e

def download_binance_daily_futures(symbols:list, start_date:str, end_date:str, save_path:str) -> pd.DataFrame:
    os.makedirs(save_path, exist_ok=True)
    
    start_ts = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp() * 1000)
    end_ts = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp() * 1000)
    
    all_data = []
    
    for symbol in symbols:
        print(f"正在下载 {symbol}...")
        file_path = os.path.join(save_path, f"{symbol}.csv")
        
        symbol_data = []
        interval = "1d"
        limit = 500
        
        current_start = start_ts
        while current_start < end_ts:
            url = f"https://fapi.binance.com/fapi/v1/klines"
            params = {
                "symbol": symbol,
                "interval": interval,
                "startTime": current_start,
                "endTime": end_ts,
                "limit": limit
            }
            
            try:
                data = fetch_with_retry(url, params)
                if not data:
                    break
                    
                for k in data:
                    kline = {
                        'symbol': symbol,
                        'trade_date': datetime.fromtimestamp(k[0] / 1000).strftime('%Y-%m-%d'),
                        'open': float(k[1]),
                        'high': float(k[2]),
                        'low': float(k[3]),
                        'close': float(k[4]),
                        'volume': float(k[5]),
                        'quote_volume': float(k[7]),
                        'trade_num': float(k[8]),
                        'taker_buy_volume': float(k[9]),
                        'taker_buy_quote_volume': float(k[10])
                    }
                    symbol_data.append(kline)
                
                current_start = data[-1][0] + 1
                time.sleep(0.5)
                
            except Exception as e:
                print(f"下载 {symbol} 失败: {e}")
                break
        
        if symbol_data:
            df_symbol = pd.DataFrame(symbol_data)
            df_symbol.to_csv(file_path, index=False)
            all_data.extend(symbol_data)
            print(f"{symbol} 下载完成，共 {len(symbol_data)} 条数据")
        else:
            print(f"{symbol} 无数据")
    
    if all_data:
        return pd.DataFrame(all_data)
    return pd.DataFrame()

def get_all_futures_symbols() -> list:
    url = "https://fapi.binance.com/fapi/v1/exchangeInfo"
    data = fetch_with_retry(url)
    symbols = [s['symbol'] for s in data['symbols'] if s['quoteAsset'] == 'USDT']
    return symbols

def incremental_update(save_path: str, end_date: str = None) -> pd.DataFrame:
    """
    增量更新: 只下载每个币对CSV中缺失的日期数据并追加
    :param save_path: 数据存放目录
    :param end_date: 更新到哪天, 默认今天
    """
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    end_ts = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp() * 1000) + 86400000

    symbols = get_all_futures_symbols()
    print(f"共 {len(symbols)} 个合约, 增量更新到 {end_date}")

    all_new_data = []

    for i, symbol in enumerate(symbols):
        file_path = os.path.join(save_path, f"{symbol}.csv")

        if os.path.exists(file_path):
            df_exist = pd.read_csv(file_path)
            last_date = df_exist['trade_date'].max()
            last_dt = datetime.strptime(last_date, '%Y-%m-%d')
            next_dt = last_dt.timestamp() * 1000 + 86400000
            if next_dt >= end_ts:
                continue
        else:
            next_dt = int(datetime.strptime('2017-01-01', '%Y-%m-%d').timestamp() * 1000)
            df_exist = pd.DataFrame()

        symbol_data = []
        current_start = int(next_dt)

        while current_start < end_ts:
            url = "https://fapi.binance.com/fapi/v1/klines"
            params = {
                "symbol": symbol,
                "interval": "1d",
                "startTime": current_start,
                "endTime": end_ts,
                "limit": 500
            }
            try:
                data = fetch_with_retry(url, params)
                if not data:
                    break
                for k in data:
                    kline = {
                        'symbol': symbol,
                        'trade_date': datetime.fromtimestamp(k[0] / 1000).strftime('%Y-%m-%d'),
                        'open': float(k[1]),
                        'high': float(k[2]),
                        'low': float(k[3]),
                        'close': float(k[4]),
                        'volume': float(k[5]),
                        'quote_volume': float(k[7]),
                        'trade_num': float(k[8]),
                        'taker_buy_volume': float(k[9]),
                        'taker_buy_quote_volume': float(k[10])
                    }
                    symbol_data.append(kline)
                current_start = data[-1][0] + 1
                time.sleep(0.3)
            except Exception as e:
                print(f"  {symbol} 更新失败: {e}")
                break

        if symbol_data:
            df_new = pd.DataFrame(symbol_data)
            if not df_exist.empty:
                df_all = pd.concat([df_exist, df_new], ignore_index=True)
                df_all = df_all.drop_duplicates(subset=['trade_date'], keep='last')
                df_all = df_all.sort_values('trade_date').reset_index(drop=True)
            else:
                df_all = df_new
            df_all.to_csv(file_path, index=False)
            all_new_data.extend(symbol_data)
            print(f"  [{i+1}/{len(symbols)}] {symbol} +{len(symbol_data)}条")
        else:
            if (i + 1) % 100 == 0:
                print(f"  [{i+1}/{len(symbols)}] 已检查...")

    print(f"\n增量更新完成, 共新增 {len(all_new_data)} 条数据")
    if all_new_data:
        return pd.DataFrame(all_new_data)
    return pd.DataFrame()

if __name__ == "__main__":
    save_path = './data/binance_futures'

    print("=" * 50)
    print("1. 全量下载")
    print("2. 增量更新(只下载缺失的日期)")
    print("=" * 50)
    choice = input("请选择模式 (1/2, 默认2): ").strip() or "2"

    if choice == "1":
        print("正在获取所有U本位合约列表...")
        symbols = get_all_futures_symbols()
        print(f"共获取到 {len(symbols)} 个U本位合约")
        start_date = '2017-01-01'
        end_date = datetime.now().strftime('%Y-%m-%d')
        df = download_binance_daily_futures(symbols, start_date, end_date, save_path)
        print(f"共下载 {len(df)} 条数据")
    else:
        df = incremental_update(save_path)
