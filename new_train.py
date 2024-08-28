import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from tqdm import tqdm
import logging
import re
from pathlib import Path
from scipy.stats import linregress
from concurrent.futures import ProcessPoolExecutor
from typing import List, Tuple

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

def calculate_factors(price_series, volume_series, high_series, low_series, window_short=5, window_medium=10, window_long=30):
    factors = pd.DataFrame(index=price_series.index)
    # 高频偏度
    returns = price_series.pct_change()
    factors['high_freq_skewness'] = returns.rolling(window=window_medium).skew()
    # 下行波动占比
    downside_returns = returns.where(returns < 0, 0)
    factors['downside_volatility_ratio'] = downside_returns.rolling(window=window_medium).std() / returns.rolling(window=window_medium).std()
    # 尾盘成交占比
    factors['end_of_day_volume_ratio'] = volume_series.rolling(window=window_medium).apply(lambda x: x.iloc[-30:].sum() / x.sum())
    # 高频量价相关性
    factors['high_freq_volume_price_corr'] = returns.rolling(window=window_medium).corr(volume_series.pct_change())
    # 改进反转
    factors['improved_reversal'] = (price_series / price_series.shift(30) - 1) - (price_series.shift(1) / price_series.shift(31) - 1)
    # 平均单笔流出金额占比
    factors['avg_outflow_ratio'] = (volume_series * price_series).where(returns < 0, 0).rolling(window=window_medium).mean() / (volume_series * price_series).rolling(window=window_medium).mean()
    
    # 大单推动涨
    large_trades = (volume_series * price_series) > (volume_series * price_series).rolling(window=window_medium).quantile(0.7)
    factors['large_trade_impact'] = returns.where(large_trades, 0).rolling(window=window_medium).sum()
    
    # 价格动量
    factors['momentum_short'] = price_series.pct_change(window_short)
    factors['momentum_medium'] = price_series.pct_change(window_medium)
    factors['momentum_long'] = price_series.pct_change(window_long)
    
    # 移动平均
    factors['ma_short'] = price_series.rolling(window=window_short).mean()
    factors['ma_medium'] = price_series.rolling(window=window_medium).mean()
    factors['ma_long'] = price_series.rolling(window=window_long).mean()
    
    # 移动平均交叉
    factors['ma_cross_short_medium'] = factors['ma_short'] / factors['ma_medium'] - 1
    factors['ma_cross_short_long'] = factors['ma_short'] / factors['ma_long'] - 1
    factors['ma_cross_medium_long'] = factors['ma_medium'] / factors['ma_long'] - 1
    
    # 波动率
    factors['volatility_short'] = price_series.rolling(window=window_short).std()
    factors['volatility_medium'] = price_series.rolling(window=window_medium).std()
    factors['volatility_long'] = price_series.rolling(window=window_long).std()
    
    # 相对强弱指标 (RSI)
    delta = price_series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window_medium).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window_medium).mean()
    rs = gain / loss
    factors['rsi'] = 100 - (100 / (1 + rs))
    
    # 布林带
    rolling_mean = price_series.rolling(window=window_medium).mean()
    rolling_std = price_series.rolling(window=window_medium).std()
    factors['bollinger_upper'] = rolling_mean + (rolling_std * 2)
    factors['bollinger_lower'] = rolling_mean - (rolling_std * 2)
    factors['bollinger_percent'] = (price_series - factors['bollinger_lower']) / (factors['bollinger_upper'] - factors['bollinger_lower'])
    
    # 价格加速度
    factors['price_acceleration'] = factors['momentum_short'].diff()
    
    if volume_series is not None:
        # 成交量相关因子
        factors['volume_momentum'] = volume_series.pct_change(window_short)
        factors['volume_ma_ratio'] = volume_series / volume_series.rolling(window=window_medium).mean()
        
        # 价量相关性
        factors['price_volume_corr'] = price_series.rolling(window=window_medium).corr(volume_series)
        
        # 资金流量指标 (MFI)
        if high_series is not None and low_series is not None:
            typical_price = (price_series + high_series + low_series) / 3
            raw_money_flow = typical_price * volume_series
            positive_flow = raw_money_flow.where(typical_price > typical_price.shift(1), 0).rolling(window=window_medium).sum()
            negative_flow = raw_money_flow.where(typical_price < typical_price.shift(1), 0).rolling(window=window_medium).sum()
            mfi_ratio = positive_flow / negative_flow
            factors['mfi'] = 100 - (100 / (1 + mfi_ratio))
    
    if high_series is not None and low_series is not None:
        # 真实波幅 (ATR)
        high_low = high_series - low_series
        high_close = np.abs(high_series - price_series.shift())
        low_close = np.abs(low_series - price_series.shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        factors['atr'] = true_range.rolling(window=window_medium).mean()
        
        # 价格通道
        factors['channel_upper'] = high_series.rolling(window=window_medium).max()
        factors['channel_lower'] = low_series.rolling(window=window_medium).min()
        factors['channel_position'] = (price_series - factors['channel_lower']) / (factors['channel_upper'] - factors['channel_lower'])
    
    # 趋势强度指标
    def calculate_trend_strength(series, window):
        slopes = [linregress(range(window), series.iloc[i:i+window])[0] for i in range(len(series) - window + 1)]
        return pd.Series(slopes + [np.nan] * (window - 1), index=series.index)
    
    factors['trend_strength'] = calculate_trend_strength(price_series, window_medium)
    
    # 删除包含 NaN 的行并填充剩余的 NaN
    factors = factors.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    return factors

