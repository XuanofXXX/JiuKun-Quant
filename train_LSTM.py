import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tqdm import tqdm

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

def calculate_factors(df, sequence_length=10):
    # 计算中间价
    mid_price = (df['AskPrice1'] + df['BidPrice1']) / 2
    
    # 计算价格动量
    price_momentum_short = mid_price.pct_change(5)
    price_momentum_mid = mid_price.pct_change(30)
    
    # 计算交易量变化
    volume_change_short = df['TotalTradeVolume'].pct_change(5)
    volume_change_mid = df['TotalTradeVolume'].pct_change(30)
    
    # 计算价差
    spread = (df['AskPrice1'] - df['BidPrice1']) / mid_price
    
    # 计算订单簿不平衡
    depth_imbalance = (df['BidVolume1'] - df['AskVolume1']) / (df['BidVolume1'] + df['AskVolume1'])
    
    # 计算波动率
    volatility = mid_price.rolling(30).std() / mid_price
    
    # 计算RSI
    delta = mid_price.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    # 计算移动平均线交叉
    ma_short = mid_price.rolling(window=5).mean()
    ma_long = mid_price.rolling(window=30).mean()
    ma_cross = ma_short / ma_long - 1
    
    # 将所有因子组合在一起
    factors = pd.concat([
        price_momentum_short,
        price_momentum_mid,
        volume_change_short,
        volume_change_mid,
        spread,
        depth_imbalance,
        volatility,
        rsi,
        ma_cross
    ], axis=1)
    
    factors.columns = ['price_momentum_short', 'price_momentum_mid', 'volume_change_short', 'volume_change_mid', 
                       'spread', 'depth_imbalance', 'volatility', 'rsi', 'ma_cross']
    factors = factors.replace([np.inf, -np.inf], np.nan).fillna(0)

    return factors

def load_and_preprocess_data(directory, sequence_length=10):
    all_data = []
    all_labels = []
    
    for filename in tqdm(os.listdir(directory), desc="Loading files"):
        if filename.endswith(".csv"):
            filepath = os.path.join(directory, filename)
            df = pd.read_csv(filepath, delimiter='|')
            
            factors = calculate_factors(df, sequence_length)
            factors = factors.dropna()
            
            scaler = StandardScaler()
            scaled_factors = scaler.fit_transform(factors)
            
            for i in range(len(scaled_factors) - sequence_length):
                sequence = scaled_factors[i:i+sequence_length]
                label = df['BidPrice1'].iloc[i+sequence_length] - df['BidPrice1'].iloc[i+sequence_length-1]
                all_data.append(sequence)
                all_labels.append(label)
    
    return torch.tensor(all_data, dtype=torch.float32), torch.tensor(all_labels, dtype=torch.float32)

def save_model(model, filename):
    torch.save(model.state_dict(), filename)

def load_model(filename, input_dim, hidden_dim, num_layers, output_dim):
    model = LSTMModel(input_dim, hidden_dim, num_layers, output_dim)
    model.load_state_dict(torch.load(filename))
    return model

def train_lstm_model(X, y, hidden_dim=32, num_layers=2, test_size=0.2, random_state=42, batch_size=64, num_epochs=1, gpu_id=0):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    print(X_train.size)
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
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch_X, batch_y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs.squeeze(), batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")
    
        # model.eval()
        # with torch.no_grad():
        #     X_test = X_test.to(device)
        #     y_pred = model(X_test).cpu()
        #     rmse = torch.sqrt(criterion(y_pred.squeeze(), y_test))
        #     print(f"RMSE: {rmse.item():.4f}")
    
    return model

if __name__ == "__main__":
    directory = './snapshots'
    sequence_length = 10
    
    X, y = load_and_preprocess_data(directory, sequence_length)
    
    # 指定要使用的GPU编号（例如，使用第一个GPU）
    # gpu_id = 2
    # model = train_lstm_model(X, y, gpu_id=gpu_id)
    
    # save_model(model, "trained_lstm_stock_model.pth")
    
    # print("Model training completed and saved.")