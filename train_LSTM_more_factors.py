import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import logging
import re
from pathlib import Path
from scipy.stats import linregress

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def preprocess_dataframe(df, sequence_length=10, forward_periods=10):
    # 按StockID和Tick排序
    df = df.sort_values(['StockID', 'Tick'])
    
    # 计算中间价格
    df['MiddlePrice'] = (df['BidPrice1'] * df['AskVolume1'] + df['AskPrice1'] * df['BidVolume1']) / (df['AskVolume1'] + df['BidVolume1'])
    
    # 初始化结果列表
    all_sequences = []
    all_labels = []
    all_stock_ids = []

    # 对每个股票分别处理
    for stock_id, stock_data in df.groupby('StockID'):
        price_series = stock_data['MiddlePrice']
        volume_series = stock_data['TotalTradeVolume']
        high_series = stock_data['AskPrice1']
        low_series = stock_data['BidPrice1']

        # 计算因子
        factors = calculate_factors(price_series, volume_series, high_series, low_series)

        if len(factors) < sequence_length + forward_periods:
            continue  # 跳过数据不足的股票

        # 标准化因子
        scaler = StandardScaler()
        scaled_factors = scaler.fit_transform(factors)

        # 计算未来收益率
        returns = price_series.pct_change(periods=forward_periods).shift(-forward_periods)

        for i in range(len(scaled_factors) - sequence_length - forward_periods + 1):
            sequence = scaled_factors[i:i+sequence_length]
            label = returns.iloc[i+sequence_length-1]

            if not np.isnan(label):
                all_sequences.append(sequence)
                all_labels.append(label)
                all_stock_ids.append(stock_id)

    if not all_sequences:
        return None, None, None  # 如果没有有效序列，返回None

    # 转换为PyTorch张量
    X = torch.tensor(np.array(all_sequences), dtype=torch.float32)
    y = torch.tensor(np.array(all_labels), dtype=torch.float32)
    stock_ids = np.array(all_stock_ids)

    return X, y, stock_ids

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

def calculate_factors(price_series, volume_series=None, high_series=None, low_series=None, window_short=5, window_medium=10, window_long=30):
    factors = pd.DataFrame(index=price_series.index)
    
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
    
    # 删除包含 NaN 的行
    factors = factors.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    return factors

from tqdm import tqdm

def load_and_preprocess_data(directory, sequence_length=10, forward_periods=10):
    all_data = []
    all_labels = []
    
    directory = Path(directory)

    # 获取所有交易日
    trading_days = sorted(set([int(file.name.split('_')[0]) for file in directory.glob('*_UBIQ*.csv')]))

    for day in tqdm(trading_days, desc="Processing trading days"):
        day_data = []
        day_labels = []
        
        # 获取当前交易日的所有股票文件
        stock_files = list(directory.glob(f'{day}_UBIQ*.csv'))
        
        for stock in stock_files:
            stock_index = re.search(r'UBIQ(\d+)', stock.name).group(1)
            df = pd.read_csv(stock, delimiter='|')
            
            df['MiddlePrice'] = (df['BidPrice1'] * df['AskVolume1'] + df['AskPrice1'] * df['BidVolume1']) / (df['AskVolume1'] + df['BidVolume1'])
            
            price_series = df['MiddlePrice']
            volume_series = df['TotalTradeVolume']
            high_series = df['AskPrice1']
            low_series = df['BidPrice1']

            factors = calculate_factors(price_series, volume_series, high_series, low_series)

            if len(factors) < sequence_length + forward_periods:
                continue  # 跳过数据不足的情况

            scaler = StandardScaler()
            scaled_factors = scaler.fit_transform(factors)

            # 计算每个时间点的收益率
            returns = price_series.pct_change(periods=forward_periods).shift(-forward_periods)

            for i in range(len(scaled_factors) - sequence_length - forward_periods + 1):
                sequence = scaled_factors[i:i+sequence_length]
                label = returns.iloc[i+sequence_length-1]

                if not np.isnan(label):
                    day_data.append(sequence)
                    day_labels.append(label)

        # 如果当天有有效数据，则添加到总数据集中
        if day_data:
            all_data.extend(day_data)
            all_labels.extend(day_labels)

    print(f"Total sequences generated: {len(all_data)}")
    return torch.tensor(all_data, dtype=torch.float32), torch.tensor(all_labels, dtype=torch.float32)

def save_model(model, filename):
    torch.save(model.state_dict(), filename)

def load_model(filename, input_dim, hidden_dim, num_layers, output_dim):
    model = LSTMModel(input_dim, hidden_dim, num_layers, output_dim)
    model.load_state_dict(torch.load(filename))
    return model

from tqdm import tqdm

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
    
    # 创建一个进度条来跟踪整个训练过程
    epoch_pbar = tqdm(total=num_epochs, desc="Training Progress", position=0)
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        # 创建一个进度条来跟踪每个epoch内的批次处理
        batch_pbar = tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{num_epochs}", position=1, leave=False)
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs.squeeze(), batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
            # 更新批次进度条
            batch_pbar.update(1)
        
        # 关闭批次进度条
        batch_pbar.close()
        
        # 更新epoch进度条
        epoch_pbar.update(1)
        epoch_pbar.set_postfix({"Loss": total_loss/len(train_loader)})
    
    # 关闭epoch进度条
    epoch_pbar.close()
    
    print(f"Final Loss: {total_loss/len(train_loader):.4f}")
    return model

if __name__ == "__main__":
    directory = './snapshots'
    sequence_length = 10
    forward_periods = 10
    
    X, y = load_and_preprocess_data(directory, sequence_length, forward_periods)
    
    # 指定要使用的GPU编号（例如，使用第三个GPU）
    gpu_id = 2
    model = train_lstm_model(X, y, gpu_id=gpu_id, num_epochs=5)
    
    save_model(model, "trained_lstm_stock_model.pth")
    
    print("Model training completed and saved.")