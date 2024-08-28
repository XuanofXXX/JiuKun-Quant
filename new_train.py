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

def calculate_period_metrics(factor: pd.Series, stk_ret: pd.Series, period: int) -> dict:
    # 获得分组数据
    group = pd.qcut(factor, q=10, labels=False, duplicates='drop')
    
    # 计算IC
    ic = factor.corr(stk_ret)
    
    # 计算分组收益
    group_ret = stk_ret.groupby(group).mean()
    cum_ret = group_ret.cumsum()
    
    # 计算统计指标
    long_ret = group_ret.iloc[-1].mean() * 3000 / period
    long_std = group_ret.iloc[-1].std() * np.sqrt(3000)
    long_ir = long_ret / long_std if long_std != 0 else 0
    
    # 计算最大回撤，处理可能的单一值情况
    if len(cum_ret) > 1:
        rolling_max = cum_ret.cummax()
        drawdowns = (rolling_max - cum_ret) / rolling_max
        long_maxdrawdown = drawdowns.max() * 100
    else:
        long_maxdrawdown = 0
    
    hedge_ret = (group_ret.iloc[-1] - group_ret.iloc[0]).mean() * 3000 / period if len(group_ret) > 1 else 0
    hedge_std = (group_ret.iloc[-1] - group_ret.iloc[0]).std() * np.sqrt(3000) if len(group_ret) > 1 else 0
    hedge_ir = hedge_ret / hedge_std if hedge_std != 0 else 0
    
    # 计算对冲策略的最大回撤，处理可能的单一值情况
    if len(cum_ret) > 1:
        hedge_cum_ret = cum_ret.iloc[-1] - cum_ret.iloc[0]
        hedge_rolling_max = hedge_cum_ret.cummax()
        hedge_drawdowns = (hedge_rolling_max - hedge_cum_ret) / hedge_rolling_max
        hedge_maxdrawdown = hedge_drawdowns.max() * 100
    else:
        hedge_maxdrawdown = 0
    
    return {
        'ic': ic, 'long_ir': long_ir, 'hedge_ir': hedge_ir,
        'long_ret': long_ret, 'hedge_ret': hedge_ret,
        'long_maxdrawdown': long_maxdrawdown, 'hedge_maxdrawdown': hedge_maxdrawdown
    }

def select_factors_simple(factors_df: pd.DataFrame, returns: pd.Series, n_factors: int = 10) -> List[str]:
    """
    使用简单的相关性和波动性指标来选择因子。
    
    :param factors_df: 包含所有因子的DataFrame
    :param returns: 对应的收益率Series
    :param n_factors: 要选择的因子数量
    :return: 选中的因子名称列表
    """
    # 计算每个因子与收益率的相关性
    correlations = factors_df.apply(lambda x: x.corr(returns))
    
    # 计算每个因子的波动性
    volatilities = factors_df.std()
    
    # 计算综合得分：相关性的绝对值 * 波动性
    scores = correlations.abs() * volatilities
    
    # 选择得分最高的n个因子
    selected_factors = scores.nlargest(n_factors).index.tolist()
    
    return selected_factors

import multiprocessing
from functools import partial

def process_stock_file(file, sequence_length, forward_periods):
    df = pd.read_csv(file, delimiter='|')
    df['MiddlePrice'] = (df['BidPrice1'] * df['AskVolume1'] + df['AskPrice1'] * df['BidVolume1']) / (df['AskVolume1'] + df['BidVolume1'])
    
    factors = calculate_factors(df['MiddlePrice'], df['TotalTradeVolume'], df['AskPrice1'], df['BidPrice1'])
    returns = df['MiddlePrice'].pct_change(periods=forward_periods).shift(-forward_periods)

    # Remove NaN values
    valid_data = factors.dropna().join(returns.dropna(), how='inner')
    
    if len(valid_data) > sequence_length + forward_periods:
        return valid_data.iloc[:, :-1], valid_data.iloc[:, -1]
    else:
        return None, None

def load_and_preprocess_data(directory, sequence_length=10, forward_periods=10, n_factors=10):
    directory = Path(directory)
    all_files = list(directory.glob('*_UBIQ*.csv'))
    
    # Use multiprocessing to process files in parallel
    with multiprocessing.Pool() as pool:
        results = list(tqdm(
            pool.imap(partial(process_stock_file, sequence_length=sequence_length, forward_periods=forward_periods), all_files),
            total=len(all_files),
            desc="Processing stock files"
        ))
    
    all_factors = [r[0] for r in results if r[0] is not None]
    all_returns = [r[1] for r in results if r[1] is not None]

    all_factors_df = pd.concat(all_factors, axis=0)
    all_returns_series = pd.concat(all_returns, axis=0)

    # 使用新的简化选择策略
    best_factors = select_factors_simple(all_factors_df, all_returns_series, n_factors)
    logging.info(f"Selected best factors: {best_factors}")

    # 准备序列，只使用最佳因子
    all_data = []
    all_labels = []

    for factors, returns in zip(all_factors, all_returns):
        factors = factors[best_factors]
        
        scaler = StandardScaler()
        scaled_factors = scaler.fit_transform(factors)

        for i in range(len(scaled_factors) - sequence_length - forward_periods + 1):
            sequence = scaled_factors[i:i+sequence_length]
            label = returns.iloc[i+sequence_length-1] * 100000
            if not np.isnan(label):
                all_data.append(sequence)
                all_labels.append(label)

    return torch.tensor(np.array(all_data), dtype=torch.float32), torch.tensor(np.array(all_labels), dtype=torch.float32), best_factors


def train_lstm_model(X, y, hidden_dim=32, num_layers=2, test_size=0.2, random_state=42, batch_size=64, num_epochs=1, gpu_id=0):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    print(f"Training data shape: {X_train.shape}")
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu_id}")
        print(f"Using GPU: {torch.cuda.get_device_name(device)}")
    else:
        device = torch.device("cpu")
        print("CUDA is not available. Using CPU.")
    
    input_dim = X.shape[2]  # number of features
    output_dim = 1  # predicting a single value
    
    model = LSTMModel(input_dim, hidden_dim, num_layers, output_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())
    
    epoch_pbar = tqdm(total=num_epochs, desc="Training Progress", position=0)
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        batch_pbar = tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{num_epochs}", position=1, leave=False)
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs.squeeze(), batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            batch_pbar.update(1)
        
        batch_pbar.close()
        epoch_pbar.update(1)
        epoch_pbar.set_postfix({"Loss": total_loss/len(train_loader)})
        if(epoch % 20 == 0):
            torch.save(model.state_dict(), f"trained_lstm_stock_model_{epoch}.pth")
    epoch_pbar.close()
    print(f"Final Loss: {total_loss/len(train_loader):.4f}")
    return model

if __name__ == "__main__":
    directory = './snapshots'
    sequence_length = 10
    forward_periods = 10
    n_factors = 10
    
    X, y, best_factors = load_and_preprocess_data(directory, sequence_length, forward_periods, n_factors)
    
    logging.info(f"Data shape: X: {X.shape}, y: {y.shape}")
    logging.info(f"Selected factors: {best_factors}")
    
    gpu_id = 2  # 使用第三个 GPU (索引从0开始)
    
    model = train_lstm_model(X, y, gpu_id=gpu_id, num_epochs=200)
    
    torch.save(model.state_dict(), "trained_lstm_stock_model.pth")
    
    print("Model training completed and saved.")