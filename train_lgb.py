import os
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
import glob

class MultiStockLOBDataset:
    def __init__(self, data_dir, lookback=10):
        self.data_dir = data_dir
        self.lookback = lookback
        self.stock_data = {}
        self.load_all_data()

    def load_all_data(self):
        file_pattern = os.path.join(self.data_dir, "*_*")
        all_files = glob.glob(file_pattern)
        
        for file_path in all_files:
            day, stock = os.path.basename(file_path).split('_')
            if stock not in self.stock_data:
                self.stock_data[stock] = {}
            self.stock_data[stock][day] = pd.read_csv(file_path)

    def calculate_factors(self, window):
        mid_price = (window['AskPrice1'] + window['BidPrice1']) / 2
        log_returns = np.log(mid_price).diff().dropna()
        
        factors = {
            'price_momentum': log_returns.mean(),
            'price_volatility': log_returns.std(),
            'volume_momentum': np.log(window['TotalTradeVolume']).diff().dropna().mean(),
            'volume_volatility': np.log(window['TotalTradeVolume']).diff().dropna().std(),
            'spread': ((window['AskPrice1'] - window['BidPrice1']) / mid_price).mean(),
            'book_imbalance': ((window['BidVolume1'] - window['AskVolume1']) / (window['BidVolume1'] + window['AskVolume1'])).mean(),
            'depth_imbalance': ((window[['BidVolume1', 'BidVolume2', 'BidVolume3']].sum(axis=1) - 
                                 window[['AskVolume1', 'AskVolume2', 'AskVolume3']].sum(axis=1)) / 
                                (window[['BidVolume1', 'BidVolume2', 'BidVolume3']].sum(axis=1) + 
                                 window[['AskVolume1', 'AskVolume2', 'AskVolume3']].sum(axis=1))).mean()
        }
        return list(factors.values())

    def prepare_data_for_stock(self, stock):
        stock_data = self.stock_data[stock]
        features = []
        labels = []
        
        for day in sorted(stock_data.keys())[:-1]:  # Exclude the last day for labels
            day_data = stock_data[day]
            next_day_data = stock_data[sorted(stock_data.keys())[sorted(stock_data.keys()).index(day) + 1]]
            
            for i in range(len(day_data) - self.lookback):
                window = day_data.iloc[i:i+self.lookback]
                factors = self.calculate_factors(window)
                
                last_mid_price = (day_data['AskPrice1'].iloc[i+self.lookback-1] + day_data['BidPrice1'].iloc[i+self.lookback-1]) / 2
                next_mid_price = (next_day_data['AskPrice1'].iloc[0] + next_day_data['BidPrice1'].iloc[0]) / 2
                log_return = np.log(next_mid_price) - np.log(last_mid_price)
                
                features.append(factors)
                labels.append(log_return)
        
        return np.array(features), np.array(labels)

    def prepare_all_data(self):
        all_features = []
        all_labels = []
        
        with ProcessPoolExecutor() as executor:
            results = list(executor.map(self.prepare_data_for_stock, self.stock_data.keys()))
        
        for features, labels in results:
            all_features.append(features)
            all_labels.append(labels)
        
        return np.concatenate(all_features), np.concatenate(all_labels)

def train_lightgbm_model(data_dir, lookback=10):
    dataset = MultiStockLOBDataset(data_dir, lookback)
    X, y = dataset.prepare_all_data()
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    train_data = lgb.Dataset(X_train, label=y_train)
    test_data = lgb.Dataset(X_test, label=y_test)
    
    params = {
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': 'mse',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1
    }
    
    callbacks = [
        lgb.early_stopping(stopping_rounds=10, verbose=True),
        lgb.log_evaluation(period=10)
    ]
    
    model = lgb.train(
        params, 
        train_data, 
        num_boost_round=1000, 
        valid_sets=[test_data], 
        callbacks=callbacks
    )
    
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")
    
    feature_importance = model.feature_importance()
    feature_names = ['Price Momentum', 'Price Volatility', 'Volume Momentum', 'Volume Volatility', 
                     'Spread', 'Book Imbalance', 'Depth Imbalance']
    
    plt.figure(figsize=(10, 6))
    plt.bar(feature_names, feature_importance)
    plt.title('Feature Importance')
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    return model

if __name__ == "__main__":
    data_dir = "./snapshots_raw"
    model = train_lightgbm_model(data_dir, lookback=10)