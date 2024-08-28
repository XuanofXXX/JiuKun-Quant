import os
import asyncio
import aiohttp
import sys
import json
import time
import logging
import random
import torch
import torch.nn as nn
import pandas as pd
import numpy as np

from scipy import stats
from functools import wraps
from typing import List, Dict
from collections import deque
from scipy.stats import linregress
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from functools import wraps

from utils.logger_config import setup_logger
from utils.convert import ConvertToSimTime_us

def async_timer(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = await func(*args, **kwargs)
        end = time.perf_counter()
        logger.info(f"{func.__name__} took {end - start:.4f} seconds")
        return result
    return wrapper

logger = setup_logger('quant_log')

def instrument2id(instrument: str):
    return int(instrument[-3:])

class AsyncInterfaceClass:
    def __init__(self, domain_name):
        self.domain_name = domain_name
        self.session = None

    async def create_session(self):
        if self.session is None:
            self.session = aiohttp.ClientSession()

    async def close_session(self):
        if self.session:
            await self.session.close()
            self.session = None

    async def send_request(self, endpoint, data):
        if not self.session:
            await self.create_session()
        url = f"{self.domain_name}{endpoint}"
        try:
            async with self.session.post(url, data=json.dumps(data)) as response:
                return await response.json()
        except aiohttp.ClientError as e:
            logger.error(f"Error sending request to {endpoint}: {str(e)}")
            return {"status": "Error", "message": str(e)}

    async def sendLogin(self, username, password):
        return await self.send_request("/Login", {"user": username, "password": password})

    async def sendGetGameInfo(self, token_ub):
        return await self.send_request("/TradeAPI/GetGameInfo", {"token_ub": token_ub})

    async def sendGetInstrumentInfo(self, token_ub):
        return await self.send_request("/TradeAPI/GetInstrumentInfo", {"token_ub": token_ub})

    async def sendGetUserInfo(self, token_ub):
        return await self.send_request("/TradeAPI/GetUserInfo", {"token_ub": token_ub})

    async def sendGetAllLimitOrderBooks(self, token_ub):
        return await self.send_request("/TradeAPI/GetAllLimitOrderBooks", {"token_ub": token_ub})

    async def sendGetActiveOrder(self, token_ub):
        return await self.send_request("/TradeAPI/GetActiveOrder", {"token_ub": token_ub})
    
    async def sendGetTrade(self, token_ub):
        return await self.send_request("/TradeAPI/GetTrade", {"token_ub": token_ub})
    
    async def sendGetAllTrades(self, token_ub):
        return await self.send_request("/TradeAPI/GetAllTrades", {"token_ub": token_ub})
    
    async def sendGetLimitOrderBook(self, token_ub, instrument: str):
        return await self.send_request("/TradeAPI/GetLimitOrderBook", {"token_ub": token_ub, 'instrument': instrument})
    
    async def sendGetAllTrades(self, token_ub):
        return await self.send_request("/TradeAPI/GetAllTrades", {"token_ub": token_ub})
    
    async def sendOrder(self, token_ub, instrument, localtime, direction, price, volume):
        data = {
            "token_ub": token_ub,
            "user_info": "",
            "instrument": instrument,
            "localtime": int(localtime),
            "direction": direction,
            "price": float(price),
            "volume": int(volume),
        }
        logger.debug(f'ordering {data}')
        return await self.send_request("/TradeAPI/Order", data)

    async def sendCancel(self, token_ub, instrument, localtime, index):
        data = {
            "token_ub": token_ub,
            "user_info": "",
            "instrument": instrument,
            "localtime": int(localtime),
            "index": index
        }
        logger.debug(f"canceling :{data}")
        return await self.send_request("/TradeAPI/Cancel", data)

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

class BaseStrategy:
    def __init__(self):
        self._cache_lobs = None
        self._cache_user_info = None
    
    @staticmethod
    def convert_api_to_df_format(api_response):
        # 创建一个字典来存储转换后的数据
        base_data = {
            'Tick': [], 'StockID': [],
            'AskPrice1': [], 'AskPrice2': [], 'AskPrice3': [], 'AskPrice4': [], 'AskPrice5': [],
            'AskPrice6': [], 'AskPrice7': [], 'AskPrice8': [], 'AskPrice9': [], 'AskPrice10': [],
            
            'AskVolume1': [], 'AskVolume2': [], 'AskVolume3': [], 'AskVolume4': [], 'AskVolume5': [],
            'AskVolume6': [], 'AskVolume7': [], 'AskVolume8': [], 'AskVolume9': [], 'AskVolume10': [],
            
            'BidPrice1': [], 'BidPrice2': [], 'BidPrice3': [], 'BidPrice4': [], 'BidPrice5': [],
            'BidPrice6': [], 'BidPrice7': [], 'BidPrice8': [], 'BidPrice9': [], 'BidPrice10': [],
            
            'BidVolume1': [], 'BidVolume2': [], 'BidVolume3': [], 'BidVolume4': [], 'BidVolume5': [],
            'BidVolume6': [], 'BidVolume7': [], 'BidVolume8': [], 'BidVolume9': [], 'BidVolume10': [],
            
            'TotalTradeVolume': [], 'TotalTradeValue': [],
            'limit_up_price': [],
            'limit_down_price': [],'last_price': [],
            'twap': []
        }

        for idx, item in enumerate(api_response):
            # 填充Ask价格和数量
            for i in range(10):
                base_data[f'AskPrice{i+1}'].append(item['askprice'][i])
                base_data[f'AskVolume{i+1}'].append(item['askvolume'][i])

            # 填充Bid价格和数量
            for i in range(10):
                base_data[f'BidPrice{i+1}'].append(item['bidprice'][i])
                base_data[f'BidVolume{i+1}'].append(item['bidvolume'][i])

            # 填充总成交量和总成交额
            base_data['TotalTradeVolume'].append(item['trade_volume'])
            base_data['TotalTradeValue'].append(item['trade_value'])
            base_data['Tick'].append(item['localtime'])
            base_data['StockID'].append(f'UBIQ{idx:03}')
            
            base_data['limit_up_price'].append(item['limit_up_price'])
            base_data['limit_down_price'].append(item['limit_down_price'])
            base_data['twap'].append(item['twap'])
            base_data['last_price'].append(item['last_price'])

        # 创建DataFrame
        df = pd.DataFrame(base_data)
        return df

    def _union_lobs(self, df: pd.DataFrame):
        # if self._cache_lobs is None:
        #     self._cache_lobs = df
        # else:
        self._cache_lobs = pd.concat([self._cache_lobs, df], axis=0, ignore_index=True)

class OrderStrategy:
    def __init__(self, time_window=3000, start_size = 100, alpha=0.1, beta=0.2, gamma=0.3):
        self.time_window = time_window
        self.start_size = start_size
        self.alpha = alpha  # 价格影响因子
        self.beta = beta    # 时间影响因子
        self.gamma = gamma  # 预测影响因子

    def calculate_order_size(self, tradable_size, t, prediction):
        # 基础订单大小就是remain_position的绝对值
        base_size = abs(tradable_size)
        
        # 时间因子：随着时间推移逐渐增加订单大小
        # TODO softmax
        time_factor = np.sqrt(t / self.time_window)
        
        # 预测因子：根据预测的强度调整订单大小
        prediction_factor = 0.5 + self.gamma * abs(prediction)
        
        # 计算最终订单大小
        size = base_size * time_factor * prediction_factor
        
        # 四舍五入到最接近的100的倍数
        size = int(round(size / 100) * 100)
        
        # 确保订单大小不超过需要交易的数量
        size = min(size, base_size)
        
        return max(size, self.start_size)

    def calculate_order_price(self, mid_price, spread, side, prediction):
        # 基础价格：买入时略低于中间价，卖出时略高于中间价
        base_price = mid_price + (0.5 * spread) if side == "buy" else mid_price - (0.5 * spread)
        
        # 根据预测调整价格
        price_adjustment = self.alpha * spread * prediction
        
        logger.debug(f"mid price: {mid_price}, spread: {spread}, prediction: {prediction}, price_adjustment: {price_adjustment}")
        
        # TODO 如果是买的话，就低价速速买进
        if side == "buy":
            price = base_price + price_adjustment
        else:  # sell
            price = base_price - price_adjustment
        
        # 确保价格在合理范围内
        return max(mid_price - spread, min(mid_price + spread, price))

    def get_order_params(self, lob, remain_position, t, side, prediction):
        mid_price = (lob["AskPrice1"] * lob['AskVolume1'] + lob["BidPrice1"] * lob['BidVolume1']) / (lob['AskVolume1'] + lob['BidVolume1'])
        spread = lob["AskPrice1"] - lob["BidPrice1"]
        
        size = self.calculate_order_size(remain_position, t, prediction)
        price = self.calculate_order_price(mid_price, spread, side, prediction)
        
        return side, round(size, 2), price
    
class NaiveOrderStrategy:
    def __init__(self, time_window=3000, start_size = 100,):
        self.time_window = time_window
        self.start_size = start_size

    def calculate_order_size(self, tradable_size, t, prediction):
        # 基础订单大小就是remain_position的绝对值
        base_size = abs(tradable_size)
        
        # 时间因子：随着时间推移逐渐增加订单大小
        # TODO softmax
        
        time_factor = np.sqrt(t / self.time_window)

        # 计算最终订单大小
        size = base_size * time_factor
        
        # 四舍五入到最接近的100的倍数
        size = int(round(size / 100) * 100)
        
        # 确保订单大小不超过需要交易的数量
        size = min(size, base_size)
        
        return max(size, self.start_size)

    def calculate_order_price(self, mid_price, spread, side, prediction, t):
        # 基础价格：买入时略低于中间价，卖出时略高于中间价
        base_price = mid_price
        
        if t < 1500:
            price_adjustment = spread * 0.3
        else:
            price_adjustment = spread * 0.7
            
        
        logger.debug(f"mid price: {mid_price}, prediction: {prediction}")
        
        # TODO 如果是买的话，就低价速速买进
        if side == "buy":
            price = base_price + price_adjustment
        else:  # sell
            price = base_price - price_adjustment
        
        # 确保价格在合理范围内
        return price

    def get_order_params(self, lob, remain_position, t, side, prediction):
        mid_price = (lob["AskPrice1"] * lob['AskVolume1'] + lob["BidPrice1"] * lob['BidVolume1']) / (lob['AskVolume1'] + lob['BidVolume1'])
        spread = lob["AskPrice1"] - lob["BidPrice1"]
        
        size = self.calculate_order_size(remain_position, t, prediction)
        price = self.calculate_order_price(mid_price, spread, side, prediction, t)
        
        return side, size , round(price,2)

class LSTMTradingStrategy(BaseStrategy):
    def __init__(self, model_path="./trained_lstm_stock_model_180.pth", lookback_period=20, sequence_length=10):
        super().__init__()
        self.model = self.load_model(model_path)
        self.scaler = StandardScaler()
        self.lookback_period = lookback_period
        self.sequence_length = sequence_length
        self.instruments = []
        self.order_strategy = OrderStrategy()
        self.nlargest = 10
        
        # 新增：用于时间冷却期和动态阈值调整
        self.initial_cooldown = 40
        self.final_cooldown = 10  # 最终冷却期更短
        # TODO 可以提高一点阈值，来降低初期的爆发交易量
        self.initial_threshold = 0.01
        self.final_threshold = 0.001  # 最终阈值更低
        self.max_order_size_ratio = 0.2  # 初始最大订单比例
        self.emergency_time = 2700  # 紧急模式触发时间（交易日的最后300秒）
        self.critical_emergency_time = 2930
        
        self.last_trade_time = {}
        self.trade_count = {}
        
        # Dedeprecated var
        self.target_positions = []
        self.price_history = {}
        self.volume_history = {}
        self.order_book_history = {}
        self.factor_history = {}
        # self.order_strategy = NaiveOrderStrategy()
        # self.hft_order_strategy = AdvancedHFTStrategy(20)

    def load_model(self, model_path):
        input_dim = 10  # 更新为新的特征数量
        hidden_dim = 32
        num_layers = 2
        output_dim = 1
        model = LSTMModel(input_dim, hidden_dim, num_layers, output_dim)
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        return model
    
    def factorDF2Tensor(self, df: pd.DataFrame, stock_num=50):
        r"""
        把pd的表格形式的数字转化为tensor的形式
        要求df有StockID,Tick列，其余均为特征列
        
        """
        logger.debug(f"df.shape:{df.shape}, stock_num: {stock_num}")
        assert df.shape[0] % stock_num == 0
        
        
        tick_num, feature_num = df.shape[0] // stock_num, df.shape[1]
        temp_df = df.sort_values(['StockID', 'Tick'])
        features = temp_df.columns.drop(['StockID', 'Tick'])
        
        # 将数据转换为PyTorch tensor
        tensor_data = torch.tensor(temp_df[features].values, dtype=torch.float32)
        
        # 重塑tensor为(50, tick_num, 46)的形状 (46是特征数，不包括StockID和Tick)
        reshaped_tensor = tensor_data.view(stock_num, tick_num, -1)
        return reshaped_tensor
    
    def calculate_factors(self, df : pd.DataFrame, window_short=5, window_medium=10, window_long=30):
        factors = pd.DataFrame(index=df.index)
        factors['Tick'] = df['Tick']
        factors['StockID'] = df['StockID']
        
        # 假设我们使用 last_price 作为价格序列
        price_series = df['last_price']
        
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
        
        # 成交量相关因子 (假设使用 TotalTradeVolume)
        volume_series = df['TotalTradeVolume']
        factors['volume_momentum'] = volume_series.pct_change(window_short)
        factors['volume_ma_ratio'] = volume_series / volume_series.rolling(window=window_medium).mean()
        
        # 价量相关性
        factors['price_volume_corr'] = price_series.rolling(window=window_medium).corr(volume_series)
        
        # 资金流量指标 (MFI)，假设你有 high_series 和 low_series (使用 BidPrice1 和 AskPrice1)
        high_series = df['BidPrice1']
        low_series = df['AskPrice1']
        typical_price = (price_series + high_series + low_series) / 3
        raw_money_flow = typical_price * volume_series
        positive_flow = raw_money_flow.where(typical_price > typical_price.shift(1), 0).rolling(window=window_medium).sum()
        negative_flow = raw_money_flow.where(typical_price < typical_price.shift(1), 0).rolling(window=window_medium).sum()
        mfi_ratio = positive_flow / negative_flow
        factors['mfi'] = 100 - (100 / (1 + mfi_ratio))
        
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
        
        # expected_factors  = [
        #     'Tick', 'StockID',
        #     'momentum_short',
        #     'momentum_medium',
        #     'momentum_long',
        #     'ma_short',
        #     'ma_medium',
        #     'ma_long',
        #     'ma_cross_short_medium',
        #     'ma_cross_short_long',
        #     'ma_cross_medium_long',
        #     'volatility_short',
        #     'volatility_medium',
        #     'volatility_long',
        #     'rsi',
        #     'bollinger_upper',
        #     'bollinger_lower',
        #     'bollinger_percent',
        #     'price_acceleration',
        #     'volume_momentum',
        #     'volume_ma_ratio',
        #     'price_volume_corr',
        #     'mfi',
        #     'atr',
        #     'channel_upper',
        #     'channel_lower',
        #     'channel_position',
        #     'trend_strength'
        # ]
        expected_factors  = [
            'Tick', 'StockID',
            'mfi', 'rsi',
            'ma_long',
            'channel_upper',
            'channel_lower',
            'bollinger_upper',
            'ma_medium',
            'bollinger_lower',
            'ma_short',
            'price_volume_corr'
            ]
        
        if 'Tick' not in expected_factors:
            expected_factors.append('Tick')
        if 'StockID' not in expected_factors:
            expected_factors.append('StockID')
        for factor in expected_factors:
            if factor not in factors.columns:
                factors[factor] = 0  # 或者使用其他合适的默认值

        factors = factors.replace([np.inf, -np.inf], np.nan).fillna(0)

        return factors[expected_factors]  # 只返回预期的因子
    
    def initialize_instruments(self, instruments):
        self.instruments = instruments
        for instrument in instruments:
            self.price_history[instrument] = deque(maxlen=self.lookback_period)
            self.volume_history[instrument] = deque(maxlen=self.lookback_period)
            self.order_book_history[instrument] = deque(maxlen=self.lookback_period)
            self.factor_history[instrument] = deque(maxlen=self.sequence_length)
    
    @async_timer
    async def update_market_data(self, api, token_ub):
        try:
            # 创建两组并发请求，每组3个
            tasks_lob = [api.sendGetAllLimitOrderBooks(token_ub) for _ in range(3)]
            tasks_trades = [api.sendGetUserInfo(token_ub) for _ in range(3)]
            
            # 分别等待每组请求中的第一个成功响应
            async def wait_for_first_success(tasks):
                while tasks:
                    done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                    for task in done:
                        try:
                            result = task.result()
                            if result["status"] == "Success":
                                for t in pending:
                                    t.cancel()
                                return result
                        except Exception:
                            pass
                    tasks = pending
                return None

            # 并行等待两组请求
            lob_result, trades_result = await asyncio.gather(
                wait_for_first_success(tasks_lob),
                wait_for_first_success(tasks_trades)
            )

            # 检查是否两种数据都获取到了
            if lob_result is None or trades_result is None:
                raise Exception("Failed to get either LOB or Trades data")
            
            new_data = self.convert_api_to_df_format(lob_result["lobs"])
            
            self._cache_user_info = trades_result['rows']
            
            if self._cache_lobs is None:
                self._cache_lobs = new_data
                logger.info(f"Added {len(new_data)} new records at tick: {new_data['Tick'].max()}.")
                return
            
            # 使用更高效的方法更新数据
            max_existing_tick = self._cache_lobs['Tick'].max()
            new_data = new_data[new_data['Tick'] > max_existing_tick]
            
            if not new_data.empty:
                self._cache_lobs = pd.concat([self._cache_lobs, new_data], ignore_index=True)
                logger.info(f"Added {len(new_data)} new records at tick: {new_data['Tick'].max()}.")
            
            # 使用更高效的方法保留最新数据
            self._cache_lobs = (self._cache_lobs.groupby('StockID')
                                .apply(lambda x: x.nlargest(self.lookback_period, 'Tick'))
                                .reset_index(drop=True))

        except Exception as e:
            logger.error(f"Error updating market data: {str(e)}")
    
    def estimate_market_volatility(self, instrument):
        """
        简单的市场波动性估计，可以根据历史价格波动或价差来计算。
        """
        recent_prices = self.price_history[instrument]
        if len(recent_prices) < 2:
            return 0
        
        # 简单的历史价格标准差作为波动性的估计
        return np.std(recent_prices)

    def calculate_dynamic_params(self, t):
        """计算动态参数"""
        progress = min(t / 3000, 1)  # 交易日进度，范围 0 到 1
        cooldown = int(self.initial_cooldown - (self.initial_cooldown - self.final_cooldown) * progress)
        threshold = self.initial_threshold - (self.initial_threshold - self.final_threshold) * progress
        order_size_ratio = self.max_order_size_ratio + (1 - self.max_order_size_ratio) * progress
        return cooldown, threshold, order_size_ratio

    @async_timer
    async def execute_trades(self, api, token_ub, user_info, t):
        try:
            if isinstance(t, float):
                t = int(t)
            stock_df = self._cache_lobs
            
            # stock_df = stock_df[~stock_df['StockID'].isin(except_stock)]
            # logger.debug(f"current stock: {stock_df['StockID'].unique()}")
            factors_df = self.calculate_factors(stock_df)
            
            input_tensor = self.factorDF2Tensor(factors_df)
            
            with torch.no_grad():
                predictions = self.model(input_tensor).numpy().flatten()

            prediction_df = pd.DataFrame({
                'instrument': self.instruments,
                'instrument_id': [int(instr[-3:]) for instr in self.instruments],
                'prediction': predictions / 10000
            })
            
            # Add target position information
            user_info_df = pd.DataFrame(user_info)
            prediction_df = prediction_df.merge(user_info_df[['remain_volume', 'target_volume', 'frozen_volume']], 
                                                left_on='instrument_id', right_index=True)
            
            prediction_df['tradable_volume'] = prediction_df['remain_volume'] - prediction_df['frozen_volume']
            
            cooldown, threshold, order_size_ratio = self.calculate_dynamic_params(t)

            def can_trade(instrument, prediction):
                last_trade = self.last_trade_time.get(instrument, 0)
                if t - last_trade < cooldown:
                    return False
                if t > self.critical_emergency_time:
                    return True
                logger.debug(f"prediction: {prediction}, threshold: {threshold}")
                return abs(prediction) > threshold
            
            # 紧急模式：如果接近收盘且还有大量未完成的目标仓位
            is_emergency = t >= self.emergency_time
            if is_emergency:
                logger.warning(f"Entering emergency mode at tick {t}")
                threshold *= 0.5  # 在紧急模式下降低阈值

            prediction_df['valid_trade'] = prediction_df.apply(
                lambda row: (
                    (is_emergency or can_trade(row['instrument'], row['prediction'])) and
                    (
                        (row['target_volume'] > 0 and row['tradable_volume'] > 0) or
                        (row['target_volume'] < 0 and row['tradable_volume'] < 0)
                    )
                ),
                axis=1
            )
            logger.debug(f"tradable stock: {prediction_df['valid_trade'].sum()}")
            logger.debug(f"Predict:{prediction_df['prediction'].describe()}")
            logger.debug(f"PredictionDF target_volume:{prediction_df['target_volume'].describe()}")
            logger.debug(f"PredictionDF tradable_volume:{prediction_df['tradable_volume'].describe()}")
            
            valid_trades = prediction_df[prediction_df['valid_trade']]
            logger.debug(f"valid_trades_df : {valid_trades.columns}")
            
            # TODO 选取前几个比较厉害的来算
            buy_candidates = valid_trades[valid_trades['target_volume'] > 0].nlargest(self.nlargest, 'prediction')
            sell_candidates = valid_trades[valid_trades['target_volume'] < 0].nsmallest(self.nlargest, 'prediction')
            
            order_tasks = []
            order_info_list = []
            chosen_stock_list = []

            logger.debug(f"Valid trades: {len(valid_trades)}")
            logger.debug(f"Buy candidates: {len(buy_candidates)}")
            logger.debug(f"Sell candidates: {len(sell_candidates)}")

            for side, candidates in [("buy", buy_candidates), ("sell", sell_candidates)]:
                for _, row in candidates.iterrows():
                    instrument = row['instrument']
                    chosen_stock_list.append(instrument)
                    
                    self.last_trade_time[instrument] = t
                    self.trade_count[instrument] = self.trade_count.get(instrument, 0) + 1
                    
                    logger.debug(f"Processing {instrument} for {side} trade")
                    
                    lob = stock_df[(stock_df['StockID'] == instrument) & (stock_df['Tick'] == t)]
                    temp_t = max(stock_df['Tick'].max(), t)
                    while lob.empty and temp_t >= 0:
                        temp_t -= 1
                        lob = stock_df[(stock_df['StockID'] == instrument) & (stock_df['Tick'] == temp_t)]
                        
                    if temp_t != t:
                        logger.debug(f"current tick {t} does not exist, using {temp_t}")
                    
                    if lob.empty:
                        logger.warning(f"No LOB data for {instrument} at tick {t}. Skipping.")
                        logger.debug(f"current t:{t}, df: {stock_df.head()}")
                        logger.debug(f"current t:{t}, lobs t: {stock_df['Tick'].min()} - {stock_df['Tick'].max()}")
                        break
                    lob = lob.iloc[0].squeeze()
                    
                    max_order_size = abs(row['tradable_volume']) * order_size_ratio
                    if is_emergency:
                        max_order_size = abs(row['tradable_volume'])  # 紧急模式下可以下全部剩余量
                    
                    target_position = prediction_df[prediction_df['instrument'] == instrument]["target_volume"].iloc[0]
                    remain_volume = prediction_df[prediction_df['instrument'] == instrument]["remain_volume"].iloc[0]
                    frozen_volume = prediction_df[prediction_df['instrument'] == instrument]["frozen_volume"].iloc[0]
                    tradable_volume = prediction_df[prediction_df['instrument'] == instrument]['tradable_volume'].iloc[0]
                    
                    current_position = target_position - remain_volume - frozen_volume
                    
                    logger.debug(f"single lob is {lob}")
                    
                    # TODO 拆分订单量，上一个tick的LOB，流动性。不同档位的量对应不同档位的订单价格
                    side, size, price = self.order_strategy.get_order_params(
                        lob, tradable_volume, t, side, row['prediction']
                    )
                    
                    price = round(price, 2)
                    logger.info(f"Placing order for {instrument}: {side} {size} @ {price}")
                    
                    if size > 0:
                        logger.info(f"Placing order for {instrument}: {side} {size} @ {price}")
                        order_tasks.append(api.sendOrder(token_ub, instrument, t, side, price, size))
                        order_info_list.append((instrument, t, side, price, size))

            results = await asyncio.gather(*order_tasks, return_exceptions=True)
            cancel_list = []

            for idx, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Order failed: {str(result)}")
                elif result["status"] == "Success":
                    logger.info(f"Order placed successfully: {result}")
                elif result["status"] == "Volume Exceeds Target":
                    order = order_tasks[idx]
                    logger.error(f"{result['status']} Order task: {order}")
                elif result["status"] == "Too Many Active Order":
                    order = order_tasks[idx]
                    logger.error(f"{result['status']} Order task: {order}")
                else:
                    logger.error(f"Order failed: {result}")
                    

        except Exception as e:
            logger.error(f"Error executing trades: {str(e)}", exc_info=True)

    async def send_all_order(self, api, token_ub):
        active_orders = await api.sendGetActiveOrder(token_ub)
        index_list = []
        if active_orders["status"] == "Success":
            for idx, instrument_info in enumerate(active_orders["instruments"]):
                if instrument_info['active_orders']:
                    for order in instrument_info['active_orders']:
                        index_list.append((instrument_info['instrument_name'] ,order['order_index']))
        else:
            logger.error(f"Failed to get active orders: {active_orders}")
        return index_list

    async def cancel_all_order(self, api, token_ub):
        active_orders = await api.sendGetActiveOrder(token_ub)
        index_list = []
        if active_orders["status"] == "Success":
            for idx, instrument_info in enumerate(active_orders["instruments"]):
                if instrument_info['active_orders']:
                    for order in instrument_info['active_orders']:
                        index_list.append((instrument_info['instrument_name'] ,order['order_index']))
        else:
            logger.error(f"Failed to get active orders: {active_orders}")
        return index_list
    
    async def work(self, api, token_ub, t):
        try:
            
            # user_info = await api.sendGetUserInfo(token_ub)
            logger.info(f"Work time: {round(t)}")
            await self.update_market_data(api, token_ub)
            # if t >= 2950 and t < 3000:  # 在交易日最后50秒执行EOD策略
            #     await self.execute_eod_strategy(api, token_ub, self._cache_user_info, t)
            # else:
                # await self.update_market_data(api, token_ub)
            await self.execute_trades(api, token_ub, self._cache_user_info, t)
            if int(t) % 10 == 0:
                logger.debug("Begin cancel")
                index_list = await self.cancel_all_order(api, token_ub)
                cancel_response = await asyncio.gather(*[api.sendCancel(token_ub, index[0], t, index[1]) for index in index_list])
                for idx, resp in enumerate(cancel_response):
                    order_info = index_list[idx]
                    if resp["status"] == "Success":
                        logger.info(f"Cancelled order for {order_info[0]}: {order_info[1]}")
                    else:
                        logger.error(f"Failed to cancel order for {order_info[0]}: {resp}\n {order_info}, {t}")

        except Exception as e:
            logger.error(f"Error in work method: {str(e)}")

    def new_day(self):
        self._cache_lobs = None
        self._cache_user_info = None
        self.last_trade_time = {}
        self.trade_count = {}

class AsyncBotsDemoClass:
    def __init__(self, username, password, port):
        self.username = username
        self.password = password
        self.api = AsyncInterfaceClass(f"http://8.147.116.35:{port}")
        self.token_ub = None
        self.instruments = []
        self.strategy = LSTMTradingStrategy()

    async def login(self):
        response = await self.api.sendLogin(self.username, self.password)
        if response["status"] == "Success":
            self.token_ub = response["token_ub"]
            logger.info(f"Login Success: {self.token_ub}")
        else:
            logger.error(f"Login Error: {response}")
            raise Exception("Login failed")

    async def GetInstruments(self):
        response = await self.api.sendGetInstrumentInfo(self.token_ub)
        if response["status"] == "Success":
            self.instruments = [instrument["instrument_name"] for instrument in response["instruments"]]
            self.strategy.initialize_instruments(self.instruments)
            logger.info(f"Get Instruments: {self.instruments}")
        else:
            logger.error(f"Get Instruments Error: {response}")
            raise Exception("Failed to get instruments")

    async def init(self):
        await self.login()

        response = await asyncio.gather(*[
            self.GetInstruments(),
            self.api.sendGetGameInfo(self.token_ub)
            ])
        
        response = response[-1]
        if response["status"] == "Success":
            self.start_time = response["next_game_start_time"]
            self.running_days = response["next_game_running_days"]
            self.running_time = response["next_game_running_time"]
            self.time_ratio = response["next_game_time_ratio"]
        else:
            logger.error(f"Get Game Info Error: {response}")
            raise Exception("Failed to get game info")
        
        self.day = 0

    async def work(self):
        t = ConvertToSimTime_us(self.start_time, self.time_ratio, self.day, self.running_time)
        await self.strategy.work(self.api, self.token_ub, t)


async def main(username, password):
    bot = AsyncBotsDemoClass(username, password, 30020)
    await bot.init()
    
    SimTimeLen = 3000
    endWaitTime = 600

    while ConvertToSimTime_us(bot.start_time, bot.time_ratio, bot.day, bot.running_time) >= SimTimeLen:
        bot.day += 1
    logger.info(f"current day: {bot.day}")
    
    while bot.day <= bot.running_days:
        while ConvertToSimTime_us(bot.start_time, bot.time_ratio, bot.day, bot.running_time) <= -900:
            await asyncio.sleep(0.1)
        
        bot.strategy.new_day()
        now = round(ConvertToSimTime_us(bot.start_time, bot.time_ratio, bot.day, bot.running_time))
        for s in range(now, SimTimeLen + endWaitTime):
            while True:
                if ConvertToSimTime_us(bot.start_time, bot.time_ratio, bot.day, bot.running_time) >= s:
                    break
            t = ConvertToSimTime_us(bot.start_time, bot.time_ratio, bot.day, bot.running_time)
            # logger.info(f"Work Time: {s} {t}")
            if t < SimTimeLen:
                await bot.work()
        bot.day += 1

    await bot.api.close_session()

if __name__ == "__main__":
    username = 'UBIQ_TEAM179'
    password = 'ANfgOwr3SvpN'
    asyncio.run(main(username, password))