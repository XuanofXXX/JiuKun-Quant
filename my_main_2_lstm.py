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

# from train_LSTM_more_factors import preprocess_dataframe

from functools import wraps

def async_timer(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = await func(*args, **kwargs)
        end = time.perf_counter()
        logger.info(f"{func.__name__} took {end - start:.4f} seconds")
        return result
    return wrapper

if os.path.exists('./quant_info_log.log'):
    os.remove('./quant_info_log.log')
if os.path.exists('./quant_log.log'):
    os.remove('./quant_log.log')
# 设置日志
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

file_handler = logging.FileHandler('quant_log.log', mode='w')
file_handler.setLevel(logging.DEBUG)
file_info_handler = logging.FileHandler('quant_info_log.log', mode='w')
file_info_handler.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)
file_info_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(file_info_handler)
logger.addHandler(console_handler)

def instrument2id(instrument: str):
    return int(instrument[-3:])

def ConvertToSimTime_us(start_time, time_ratio, day, running_time):
    return (time.time() - start_time - (day - 1) * running_time) * time_ratio

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
            async with self.session.post(url, json=data) as response:
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
            "localtime": localtime,
            "direction": direction,
            "price": price,
            "volume": volume,
        }
        logger.debug(data)
        return await self.send_request("/TradeAPI/Order", data)

    async def sendCancel(self, token_ub, instrument, localtime, index):
        data = {
            "token_ub": token_ub,
            "user_info": "",
            "instrument": instrument,
            "localtime": localtime,
            "index": index
        }
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

    def calculate_order_size(self, remain_position, t, prediction):
        # 基础订单大小就是remain_position的绝对值
        base_size = abs(remain_position)
        
        # 时间因子：随着时间推移逐渐增加订单大小
        # TODO softmax
        time_factor = (t / self.time_window)
        
        # 预测因子：根据预测的强度调整订单大小
        prediction_factor = 1 + self.gamma * abs(prediction)
        
        # 计算最终订单大小
        size = base_size * time_factor * prediction_factor
        
        # 四舍五入到最接近的100的倍数
        size = int(round(size / 100) * 100)
        
        # 确保订单大小不超过需要交易的数量
        size = min(size, base_size)
        
        return max(size, self.start_size)

    def calculate_order_price(self, mid_price, spread, side, prediction):
        # 基础价格：买入时略低于中间价，卖出时略高于中间价
        base_price = mid_price - (0.5 * spread) if side == "buy" else mid_price + (0.5 * spread)
        
        # 根据预测调整价格
        price_adjustment = self.alpha * spread * prediction
        
        if side == "buy":
            price = base_price - price_adjustment
        else:  # sell
            price = base_price + price_adjustment
        
        # 确保价格在合理范围内
        return max(mid_price - spread, min(mid_price + spread, price))

    def get_order_params(self, lob, remain_position, t, side, prediction):
        mid_price = (lob["AskPrice1"] * lob['AskVolume1'] + lob["BidPrice1"] * lob['BidVolume1']) / (lob['AskVolume1'] + lob['BidVolume1'])
        spread = lob["AskPrice1"] - lob["BidPrice1"]
        
        size = self.calculate_order_size(remain_position, t, prediction)
        price = self.calculate_order_price(mid_price, spread, side, prediction)
        
        return side, round(size, 2), price

class CancelOrderStrategy:
    def __init__(self, max_order_age: int = 5, price_threshold: float = 0.005, volume_threshold: float = 0.5):
        self.max_order_age = max_order_age  # 最大订单年龄（秒）
        self.price_threshold = price_threshold  # 价格偏离阈值
        self.volume_threshold = volume_threshold  # 成交量阈值

    async def check_and_cancel_orders(self, api, token_ub, active_orders: List[Dict], current_lob: Dict, current_time: int):
        cancel_tasks = []

        for order in active_orders:
            if self.should_cancel_order(order, current_lob, current_time):
                cancel_tasks.append(self.cancel_order(api, token_ub, order, current_time))

        if cancel_tasks:
            await asyncio.gather(*cancel_tasks)

    def should_cancel_order(self, order: Dict, current_lob: Dict, current_time: int) -> bool:
        # 检查订单年龄
        if current_time - order['localtime'] > self.max_order_age:
            logger.info(f"Order {order['index']} exceeded max age, cancelling.")
            return True

        # 检查价格偏离
        if order['direction'] == 'buy':
            current_price = current_lob['askprice'][0]
        else:
            current_price = current_lob['bidprice'][0]

        price_deviation = abs(order['price'] - current_price) / current_price
        if price_deviation > self.price_threshold:
            logger.info(f"Order {order['index']} price deviation {price_deviation:.2%} exceeded threshold, cancelling.")
            return True

        # 检查成交量
        if order['volume'] > current_lob['askvolume'][0] * self.volume_threshold:
            logger.info(f"Order {order['index']} volume {order['volume']} exceeded {self.volume_threshold:.0%} of market volume, cancelling.")
            return True

        return False

    async def cancel_order(self, api, token_ub: str, order: Dict, current_time: int):
        try:
            response = await api.sendCancel(token_ub, order['instrument'], current_time, order['index'])
            if response['status'] == 'Success':
                logger.info(f"Successfully cancelled order {order['index']} for {order['instrument']}")
            else:
                logger.error(f"Failed to cancel order {order['index']}: {response}")
        except Exception as e:
            logger.error(f"Error cancelling order {order['index']}: {str(e)}")

    async def update_and_cancel(self, api, token_ub: str, current_time: int):
        try:
            # 获取活跃订单
            active_orders_response = await api.sendGetActiveOrder(token_ub)
            if active_orders_response['status'] != 'Success':
                logger.error(f"Failed to get active orders: {active_orders_response}")
                return

            active_orders = active_orders_response['rows']

            # 获取最新的限价订单簿
            lob_response = await api.sendGetAllLimitOrderBooks(token_ub)
            if lob_response['status'] != 'Success':
                logger.error(f"Failed to get limit order books: {lob_response}")
                return

            lobs = lob_response['lobs']

            # 检查并取消订单
            for order in active_orders:
                instrument_id = int(order['instrument'][-3:])
                current_lob = lobs[instrument_id]
                await self.check_and_cancel_orders(api, token_ub, [order], current_lob, current_time)

        except Exception as e:
            logger.error(f"Error in update_and_cancel: {str(e)}")

class LSTMTradingStrategy(BaseStrategy):
    def __init__(self, model_path="./trained_lstm_stock_model_120.pth", lookback_period=20, sequence_length=10):
        super().__init__()
        self.model = self.load_model(model_path)
        self.scaler = StandardScaler()
        self.lookback_period = lookback_period
        self.sequence_length = sequence_length
        self.price_history = {}
        self.volume_history = {}
        self.order_book_history = {}
        self.factor_history = {}
        self.instruments = []
        self.target_positions = []
        self.order_strategy = OrderStrategy()
        self.cancel_strategy = CancelOrderStrategy()
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
    
    def factorDF2Tensor(self, df: pd.DataFrame):
        r"""
        把pd的表格形式的数字转化为tensor的形式
        要求df有StockID,Tick列，其余均为特征列
        
        """
        assert df.shape[0] % 50 == 0
        
        tick_num, feature_num = df.shape[0] // 50, df.shape[1]
        temp_df = df.sort_values(['StockID', 'Tick'])
        features = temp_df.columns.drop(['StockID', 'Tick'])
        
        # 将数据转换为PyTorch tensor
        tensor_data = torch.tensor(temp_df[features].values, dtype=torch.float32)
        
        # 重塑tensor为(50, tick_num, 46)的形状 (46是特征数，不包括StockID和Tick)
        reshaped_tensor = tensor_data.view(50, tick_num, -1)
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

    @async_timer
    async def execute_trades(self, api, token_ub, user_info, t):
        try:
            if isinstance(t, float):
                t = int(t)
            stock_df = self._cache_lobs
            factors_df = self.calculate_factors(self._cache_lobs)
            input_tensor = self.factorDF2Tensor(factors_df)
            
            with torch.no_grad():
                predictions = self.model(input_tensor).numpy().flatten()

            prediction_df = pd.DataFrame({
                'instrument': self.instruments,
                'instrument_id': [int(instr[-3:]) for instr in self.instruments],
                'prediction': predictions / 100000
            })
            
            # Add target position information
            user_info_df = pd.DataFrame(user_info)
            prediction_df = prediction_df.merge(user_info_df[['remain_volume', 'target_volume', 'frozen_volume']], 
                                                left_on='instrument_id', right_index=True)
            
            prediction_df['tradable_volume'] = prediction_df['remain_volume'] - prediction_df['target_volume']
            prediction_df_mean = prediction_df['prediction'].mean()
            prediction_df_std = prediction_df['prediction'].std()
            
            prediction_df['valid_trade'] = (
                ((prediction_df['prediction'] > prediction_df_mean + prediction_df_std / 3) & (prediction_df['target_volume'] > 0) & (prediction_df['tradable_volume'] != 0)) | \
                ((prediction_df['prediction'] < prediction_df_mean - prediction_df_std / 3) & (prediction_df['target_volume'] < 0) & (prediction_df['tradable_volume'] != 0))
            )
            
            logger.debug(f"Predict:{prediction_df['prediction'].describe()}")
            logger.debug(f"PredictionDF target_volume:{prediction_df['target_volume'].describe()}")
            logger.debug(f"PredictionDF tradable_volume:{prediction_df['tradable_volume'].describe()}")
            
            valid_trades = prediction_df[prediction_df['valid_trade']]
            
            # Sort by absolute prediction value and select top candidates
            buy_candidates = valid_trades[valid_trades['target_volume'] > 0].nlargest(10, 'prediction')
            sell_candidates = valid_trades[valid_trades['target_volume'] < 0].nsmallest(10, 'prediction')

            order_tasks = []

            for side, candidates in [("buy", buy_candidates), ("sell", sell_candidates)]:
                for _, row in candidates.iterrows():
                    instrument = row['instrument']
                    instrument_id = row['instrument_id']
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
                    
                    target_position = prediction_df[prediction_df['instrument'] == instrument]["target_volume"].iloc[0]
                    remain_volume = prediction_df[prediction_df['instrument'] == instrument]["remain_volume"].iloc[0]
                    frozen_volume = prediction_df[prediction_df['instrument'] == instrument]["frozen_volume"].iloc[0]
                    tradable_volume = prediction_df[prediction_df['instrument'] == instrument]['tradable_volume'].iloc[0]
                    
                    current_position = target_position - remain_volume - frozen_volume
                    
                    side, size, price = self.order_strategy.get_order_params(
                        lob, tradable_volume, t, side, row['prediction']
                    )
                    
                    price = round(price, 2)
                    logger.info(f"Placing order for {instrument}: {side} {size} @ {price}")
                    
                    if size > 0:
                        logger.info(f"Placing order for {instrument}: {side} {size} @ {price}")
                        order_tasks.append(api.sendOrder(token_ub, instrument, t, side, price, size))

            results = await asyncio.gather(*order_tasks, return_exceptions=True)

            for result in results:
                if isinstance(result, Exception):
                    logger.error(f"Order failed: {str(result)}")
                elif result["status"] != "Success":
                    logger.error(f"Order failed: {result}")
                else:
                    logger.info(f"Order placed successfully: {result}")

        except Exception as e:
            logger.error(f"Error executing trades: {str(e)}", exc_info=True)
            
    async def execute_eod_strategy(self, api, token_ub, user_info, t):
        try:
            logger.info("Executing EOD strategy")
            
            all_lobs = await api.sendGetAllLimitOrderBooks(token_ub)
            if all_lobs["status"] != "Success":
                raise Exception("Failed to get LOB data for EOD strategy")

            for instrument in self.instruments:
                instrument_id = int(instrument[-3:])
                current_position = user_info["rows"][instrument_id]["remain_volume"]
                target_position = user_info["rows"][instrument_id]["target_volume"]
                
                if current_position == target_position:
                    continue

                lob = all_lobs["lobs"][self.instruments.index(instrument)]
                volume_to_trade = abs(target_position - current_position)
                
                if volume_to_trade > 0:
                    side = "buy" if target_position > current_position else "sell"
                    price = float(lob["askprice"][0]) * 1.001 if side == "buy" else float(lob["bidprice"][0]) * 0.999
                    
                    batch_size = min(volume_to_trade, 1000)
                    while volume_to_trade > 0:
                        order_volume = min(batch_size, volume_to_trade)
                        response = await api.sendOrder(token_ub, instrument, t, side, price, order_volume)
                        
                        if response["status"] == "Success":
                            logger.info(f"EOD order placed for {instrument}: {side} {order_volume} @ {price}")
                            volume_to_trade -= order_volume
                        else:
                            logger.error(f"EOD order failed for {instrument}: {response}")
                        
                        await asyncio.sleep(0.1)
            
            logger.info("EOD strategy execution completed")
        except Exception as e:
            logger.error(f"Error executing EOD strategy: {str(e)}")

    async def work(self, api, token_ub, t):
        try:
            # user_info = await api.sendGetUserInfo(token_ub)
            logger.info(f"Work time: {round(t)}")
            await self.update_market_data(api, token_ub)
            if t >= 2950 and t < 3000:  # 在交易日最后50秒执行EOD策略
                await self.execute_eod_strategy(api, token_ub, self._cache_user_info, t)
            else:
                # await self.update_market_data(api, token_ub)
                await self.execute_trades(api, token_ub, self._cache_user_info, t)
            # await self.cancel_strategy.update_and_cancel(api, token_ub, t)

        except Exception as e:
            logger.error(f"Error in work method: {str(e)}")

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

    while bot.day <= bot.running_days:
        while ConvertToSimTime_us(bot.start_time, bot.time_ratio, bot.day, bot.running_time) <= -900:
            await asyncio.sleep(0.1)

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