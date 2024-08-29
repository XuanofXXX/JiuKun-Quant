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
import warnings

from scipy import stats
from functools import wraps
from typing import List, Dict
from collections import deque
from scipy.stats import linregress
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler


from utils.logger_config import setup_logger
from utils.convert import (
    ConvertToSimTime_us,
    convert_userinfo_response_to_df_format,
    convert_LOB_response_to_df_format
)


def async_timer(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = await func(*args, **kwargs)
        end = time.perf_counter()
        logger.info(f"{func.__name__} took {end - start:.4f} seconds")
        return result
    return wrapper

def async_retry(max_retries=3, base_delay=0.05, max_delay=0.5):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            retries = 0
            while retries < max_retries:
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    retries += 1
                    if retries == max_retries:
                        logging.error(f"Failed after {max_retries} retries. Error: {str(e)}")
                        raise
                    
                    # 计算延迟时间，使用指数退避策略
                    delay = min(base_delay * (1.5 ** (retries - 1)), max_delay)
                    # 添加一些随机性以避免多个请求同时重试
                    jitter = random.uniform(0, 0.1 * delay)
                    total_delay = delay + jitter
                    
                    logging.warning(f"Attempt {retries} failed. Retrying in {total_delay:.4f} seconds. Error: {str(e)}")
                    await asyncio.sleep(total_delay)
        return wrapper
    return decorator

logger = setup_logger('quant_log')
warnings.filterwarnings("ignore", category=DeprecationWarning)

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

    @async_retry(20)
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

class OrderStrategy:
    def __init__(self, time_window=3000, start_size = 100, alpha=0.1, beta=0.2, gamma=0.3):
        self.time_window = time_window
        self.start_size = start_size
        self.alpha = alpha  # 价格影响因子
        self.beta = beta    # 时间影响因子
        self.gamma = gamma  # 预测影响因子
        self.min_order_size = 100  # 最小订单大小
        self.max_price_levels = 3 # 最多使用的价格档位数

    # def split_orders(self, current_lob, prev_lob, tradable_size, remaining_time, target_position, t, side, prediction):
    #     orders = []
    #     remaining_size = abs(tradable_size)
    #     adjusted_size = self.calculate_order_size(tradable_size, remaining_time, target_position, prediction)
        
    #     if side == "buy":
    #         price_levels = list(zip(current_lob['BidPrice1':'BidPrice10'], current_lob['BidVolume1':'BidVolume10']))
    #         price_levels.sort(key=lambda x: x[0], reverse=True)  # 从高到低排序买价
    #     else:
    #         price_levels = list(zip(current_lob['AskPrice1':'AskPrice10'], current_lob['AskVolume1':'AskVolume10']))
    #         price_levels.sort(key=lambda x: x[0])  # 从低到高排序卖价

    #     for i, (price, volume) in enumerate(price_levels[:self.max_price_levels]):
    #         if adjusted_size <= 0:
    #             break

    #         # 计算在这个价格级别的订单大小
    #         level_liquidity = volume
    #         order_size = min(adjusted_size, level_liquidity, max(self.min_order_size, adjusted_size * 0.3))
    #         order_size = int(round(order_size / 100) * 100)  # 取整到100的倍数

    #         if order_size >= self.min_order_size:
    #             # 根据预测和之前的LOB调整价格
    #             adjusted_price = self.adjust_price(price, side, prediction, prev_lob)
    #             orders.append((side, order_size, round(adjusted_price, 2)))
    #             adjusted_size -= order_size

    #     return orders
    
    def split_orders(self, current_lob, prev_lob, tradable_size, remaining_time, target_position, t, side, prediction):
        orders = []
        remaining_size = abs(tradable_size)
        
        total_trade_volume = self.calculate_order_size(tradable_size, remaining_time, target_position, prediction )

        # # 确保总交易量至少为1000，并且是100的倍数
        # if remaining_size < 1000 or remaining_size % 100 != 0:
        #     return orders

        # 计算每批订单的大小
        batch_sizes = self.calculate_batch_sizes(total_trade_volume)
        logger.debug(f"total_trade_volume: {total_trade_volume}")

        if side == "buy":
            price_levels = list(zip(current_lob['BidPrice1':'BidPrice10'], current_lob['BidVolume1':'BidVolume10']))
            price_levels.sort(key=lambda x: x[0], reverse=True)  # 从高到低排序买价
        else:
            price_levels = list(zip(current_lob['AskPrice1':'AskPrice10'], current_lob['AskVolume1':'AskVolume10']))
            price_levels.sort(key=lambda x: x[0])  # 从低到高排序卖价
        
        for i, batch_size in enumerate(batch_sizes):
            if i >= len(price_levels):
                break

            # price, volume = price_levels[i]
            spread = current_lob['AskPrice1'] - current_lob['BidPrice1']
            
            # 第一批订单使用更保守的价格
            # TODO 调一下分批的价格和量
            if i == 0:
                price = current_lob['BidPrice1'] if side == "buy" else current_lob['AskPrice1']
                adjusted_price = self.adjust_price_conservative(price, side, prev_lob)
            elif i == 1:
                price = current_lob['BidPrice2'] if side == "buy" else current_lob['AskPrice2']
                adjusted_price = self.adjust_price_aggressive(price, spread, side, prediction, prev_lob)
            else:
                # todo 赌的可能有点大
                price = current_lob['BidPrice5'] if side == "buy" else current_lob['AskPrice5']
                adjusted_price = self.adjust_price_aggressive(price, spread, side, prediction, prev_lob)
            

            # 确保价格是2位小数
            adjusted_price = round(adjusted_price, 2)
            orders.append((side, batch_size, adjusted_price))

        
        logger.debug(f"拆单之后的单子：{(side, batch_size, adjusted_price)}")
        return orders

    def calculate_batch_sizes(self, total_size):
        batch_sizes = []
        remaining = total_size

        # 第一批：40%的总量，但不少于100，并且是100的倍数
        first_batch = max(int(total_size * 0.4) // 100 * 100, 100)
        batch_sizes.append(first_batch)
        remaining -= first_batch

        # 如果还有剩余，分配第二批
        if remaining >= 1000:
            second_batch = min(remaining, max(int(total_size * 0.3) // 100 * 100, 1000))
            batch_sizes.append(second_batch)
            remaining -= second_batch

        # 如果还有剩余，分配第三批
        if remaining >= 1000:
            batch_sizes.append(remaining // 100 * 100)  # 确保是100的倍数

        return batch_sizes

    # todo 更迅速地把货出完
    def adjust_price_conservative(self, base_price, side, prev_lob):
        # 更保守的价格调整，更接近当前市场价格
        if side == "buy":
            # return round(min(base_price, prev_lob['AskPrice1']), 2)
            return round(max(base_price, prev_lob['AskPrice1']), 2)
        else:  # sell
            # return round(max(base_price, prev_lob['BidPrice1']), 2)
            return round(min(base_price, prev_lob['BidPrice1']), 2)

    def adjust_price_aggressive(self, base_price, spread, side, prediction, prev_lob):
        # 更激进的价格调整，考虑预测因素
        price_adjustment = 0.5 * spread
        logger.debug(f"aggressive price_adjustment: {price_adjustment}")
        if side == "buy":
            # prev_ask = prev_lob['AskPrice2']
            # price_adjustment = self.alpha * prediction * (base_price - prev_ask)
            return round(base_price - price_adjustment, 2)
        else:  # sell
            # prev_bid = prev_lob['BidPrice2']
            # price_adjustment = self.alpha * prediction * (prev_bid - base_price)
            # price_adjustment = 0.5 * spread
            # logger.debug(f"aggressive price_adjustment: {price_adjustment}")
            return round(base_price + price_adjustment, 2)
    
    def adjust_price(self, base_price, side, prediction, prev_lob):
        # 根据预测和之前的LOB调整价格
        if side == "buy":
            prev_ask = prev_lob['AskPrice1']
            price_adjustment = prediction * (base_price - prev_ask)
            price = base_price + price_adjustment
        else:  # sell
            prev_bid = prev_lob['BidPrice1']
            price_adjustment = prediction * (prev_bid - base_price)
            price = base_price - price_adjustment

        logger.debug(f"base_price: {base_price}, prediction: {prediction}, adjusted_price: {price}")
        return price

    def calculate_order_size(self, tradable_size, remaining_time, target_position, prediction):
        # 基础订单大小
        base_size = abs(tradable_size)
        
        # 时间因子：随着剩余时间减少，增加订单大小
        time_factor = max(0.01, 1 - (remaining_time / self.time_window))
        
        # 目标仓位因子：距离目标仓位越远，订单大小越小
        position_factor = abs(target_position) / (abs(target_position) + abs(tradable_size))
        
        # 预测因子：根据预测的强度调整订单大小
        # todo 这里的abs(prediction) * 100是为了将预测值映射到0.5-1.5之间
        prediction_factor = 0.5 + self.gamma * abs(prediction) * 100
        
        # 计算最终订单大小
        size = base_size * time_factor * position_factor * prediction_factor
        
        logger.debug(f"calculating order size: base_size: {base_size}, time_factor: {time_factor}, position_factor: {position_factor}, prediction_factor: {prediction_factor}, calculate size:{size}")
        
        # 四舍五入到最接近的100的倍数
        size = int(round(size / 100) * 100)
        
        # 确保订单大小不超过需要交易的数量
        size = min(size, base_size)
        
        return max(size, self.start_size)

    def get_order_params(self, lob, remain_position, t, side, prediction):
        mid_price = (lob["AskPrice1"] * lob['AskVolume1'] + lob["BidPrice1"] * lob['BidVolume1']) / (lob['AskVolume1'] + lob['BidVolume1'])
        spread = lob["AskPrice1"] - lob["BidPrice1"]
        
        size = self.calculate_order_size(remain_position, t, prediction)
        price = self.calculate_order_price(mid_price, spread, side, prediction, t)
        
        return side, size , round(price,2)

class LSTMTradingStrategy:
    def __init__(self, model_path="./trained_lstm_stock_model_180.pth", lookback_period=20, sequence_length=10):
        super().__init__()
        self.model = self.load_model(model_path)
        self._cache_lobs = None
        self._cache_user_info = None
        self._cache_past_trade_price = None
        # Tick  StockID   Price Volume OrderIndex
        # 0   2716  UBIQ001    6.49    100     153027
        # ...
        # 80  2741  UBIQ042  -92.75    100      95958
        self.scaler = StandardScaler()
        self.lookback_period = lookback_period
        self.sequence_length = sequence_length
        self.instruments = []
        self.order_strategy = OrderStrategy()
        self.nlargest = 3
        
        # 新增：用于时间冷却期和动态阈值调整
        self.initial_cooldown = 40
        self.final_cooldown = 5  # 最终冷却期更短
        # TODO 可以提高一点阈值，来降低初期的爆发交易量
        self.initial_threshold = 0.01
        self.final_threshold = 0.001  # 最终阈值更低
        self.max_order_size_ratio = 0.2  # 初始最大订单比例
        self.emergency_time = 2700  # 紧急模式触发时间（交易日的最后300秒）
        self.critical_emergency_time = 2950
        
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
            tasks_user_info = [api.sendGetUserInfo(token_ub) for _ in range(3)]
            
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
            lob_result, user_info_result = await asyncio.gather(
                wait_for_first_success(tasks_lob),
                wait_for_first_success(tasks_user_info)
            )

            # 检查是否两种数据都获取到了
            if lob_result is None or user_info_result is None:
                raise Exception("Failed to get either LOB or Trades data")
            
            new_data = convert_LOB_response_to_df_format(lob_result["lobs"])
            new_data['vwap'] = new_data['TotalTradeValue'] / new_data['TotalTradeVolume']
            
            max_tick = new_data['Tick'].max()
            
            if self._cache_lobs is None:
                self._cache_lobs = new_data
                logger.info(f"Added {len(new_data)} new records at tick: {new_data['Tick'].max()}.")
                return
            
            self._cache_user_info = convert_userinfo_response_to_df_format(user_info_result['rows'], max_tick)
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
        # order_size_ratio = self.max_order_size_ratio + (1 - self.max_order_size_ratio) * progress
        return cooldown, threshold

    def pre_processing(self):
        stock_df = self._cache_lobs
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
        user_info_df = self._cache_user_info
        logger.debug(f'user info df: {user_info_df}')
        prediction_df = prediction_df.merge(user_info_df[['remain_volume', 'target_volume', 'frozen_volume']], 
                                            left_on='instrument_id', right_index=True)
        
        prediction_df['tradable_volume'] = prediction_df['remain_volume'] - prediction_df['frozen_volume']
        logger.debug(f"Prediction_df columns: {prediction_df.columns}")
        return stock_df, prediction_df

    def choose_stocks(self, prediction_df: pd.DataFrame, t):
        '''
        The prediction_df.columns:
            'instrument', 'instrument_id', 'prediction', 'remain_volume',
            'target_volume', 'frozen_volume', 'tradable_volume'
        '''
        
        cooldown, threshold = self.calculate_dynamic_params(t)

        # TODO 对机会的定义更严格
        # 1. 看过去的10|20|30|100个成交金额
        # 2. 过去波动是否剧烈
        # 3. 均线穿越的策略（VWAP和twap）
        # 4. VWAP， twap
        # 5. 综合的策略，比如5满足3即可
        def can_trade(instrument, prediction, tradable_volume, target_volume):
            last_trade = self.last_trade_time.get(instrument, 0)
            if t - last_trade < cooldown:
                return False
            if t > self.critical_emergency_time:
                return True
            if abs(prediction) <= threshold:
                return False
            # return abs(tradable_volume) / abs(target_volume) > 0.1  # 如果剩余交易量占目标的10%以上
            return False

        # 紧急模式：如果接近收盘且还有大量未完成的目标仓位
        is_emergency = t >= self.emergency_time
        if is_emergency:
            logger.warning(f"Entering emergency mode at tick {t}")
            threshold *= 0.5  # 在紧急模式下降低阈值
        
        def calculate_combined_score(row, current_price, prediction, model_weight, twap_weight, vwap_weight):
            twap_diff = (current_price - row['twap']) / row['twap']
            vwap_diff = (current_price - row['vwap']) / row['vwap']
            
            if row['target_volume'] > 0:  # 买入
                return model_weight * prediction - twap_weight * twap_diff - vwap_weight * vwap_diff
            else:  # 卖出
                return model_weight * prediction + twap_weight * twap_diff + vwap_weight * vwap_diff
        
        prediction_df['valid_trade'] = prediction_df.apply(
            lambda row: (
                (is_emergency or can_trade(row['instrument'], row['prediction'], row['tradable_volume'], row['target_volume'])) and
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
        
        logger.debug(f"Valid trades: {len(valid_trades)}")
        logger.debug(f"Buy candidates: {len(buy_candidates)}")
        logger.debug(f"Sell candidates: {len(sell_candidates)}")
        
        return buy_candidates, sell_candidates

    @async_timer
    async def execute_trades(self, api, token_ub, t):
        try:
            if isinstance(t, float):
                t = int(t)
            remaining_time = max(0, 3000 - t)
            stock_df, prediction_df = self.pre_processing()
            buy_candidates, sell_candidates = self.choose_stocks(prediction_df, t)
            
            order_tasks = []
            order_info_list = []
            chosen_stock_list = []

            for side, candidates in [("buy", buy_candidates), ("sell", sell_candidates)]:
                for _, row in candidates.iterrows():
                    instrument = row['instrument']
                    chosen_stock_list.append(instrument)
                    
                    self.last_trade_time[instrument] = t
                    self.trade_count[instrument] = self.trade_count.get(instrument, 0) + 1
                    
                    logger.debug(f"Processing {instrument} for {side} trade")
                    

                    current_lob = stock_df[(stock_df['StockID'] == instrument) & (stock_df['Tick'] == t)]
                    prev_lob = None
                    temp_t = max(stock_df['Tick'].max(), t)
                    while current_lob.empty and temp_t >= 0:
                        temp_t -= 1
                        current_lob = stock_df[(stock_df['StockID'] == instrument) & (stock_df['Tick'] == temp_t)]
                    if temp_t != t:
                        logger.debug(f"current tick {t} does not exist, using {temp_t}")
                    
                    prev_lob = stock_df[(stock_df['StockID'] == instrument) & (stock_df['Tick'] == t-1)]
                    temp_t = max(stock_df['Tick'].max(), t)-1
                    while prev_lob.empty and temp_t >= 0:
                        temp_t -= 1
                        prev_lob = stock_df[(stock_df['StockID'] == instrument) & (stock_df['Tick'] == temp_t)]
                    
                    if current_lob.empty:
                        logger.warning(f"No LOB data for {instrument} at tick {t}. Skipping.")
                        logger.debug(f"current t:{t}, df: {stock_df.head()}")
                        logger.debug(f"current t:{t}, lobs t: {stock_df['Tick'].min()} - {stock_df['Tick'].max()}")
                        break
                    
                    current_lob = current_lob.iloc[0]
                    prev_lob = prev_lob.iloc[0]
                    
                    tradable_volume = row['tradable_volume']
                    target_position = row['target_volume']
                    
                    split_orders = self.order_strategy.split_orders(
                        current_lob, prev_lob, tradable_volume, remaining_time, target_position, t, side, row['prediction']
                    )
                    
                    for split_side, size, price in split_orders:
                        if size > 0:
                            logger.info(f"Placing split order for {instrument}: {split_side} {size} @ {price}")
                            order_tasks.append(api.sendOrder(token_ub, instrument, t, split_side, price, size))
                            order_info_list.append((t, instrument, split_side, price, size))

            await self.send_all_order(order_tasks, order_info_list)

        except Exception as e:
            logger.error(f"Error executing trades: {str(e)}", exc_info=True)
    
    async def send_all_order(self, order_list_coroutine, order_info_list):
        results = await asyncio.gather(*order_list_coroutine, return_exceptions=True)

        for idx, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Order failed: {str(result)}")
            elif result["status"] == "Success":
                order_detail = order_info_list[idx]
                order_index = result['index']
                logger.info(f"Order placed successfully: {result}")
                if self._cache_past_trade_price is None:
                    self._cache_past_trade_price = pd.DataFrame(
                        columns=['Tick', 'StockID', 'Price', 'Volume', 'OrderIndex']
                    )
                
                logger.debug(f'new_row_append:{list(order_detail) + [order_index]}')
                # new_row_append:[766, 'UBIQ049', 'buy', 57.91, 100, 62695]
                new_row = pd.DataFrame([list(order_detail) + [order_index]], columns=['Tick', 'StockID','Side', 'Price', 'Volume', 'OrderIndex'])
                logger.debug(f'new_row:{new_row}')
                # The truth value of a Series is ambiguous. Use a.empty, a.bool(), a.item(), a.any() or a.all().
                new_row['Price'] = -new_row['Price'] if new_row['Side'].item() == 'sell' else new_row['Price']
                new_row = new_row.drop('Side', axis=1)
                self._cache_past_trade_price = pd.concat([self._cache_past_trade_price, new_row], ignore_index=True)
                
            elif result["status"] == "Volume Exceeds Target":
                order_detail = order_info_list[idx]
                logger.error(f"{result['status']} Order task: {order_detail}")
            elif result["status"] == "Too Many Active Order":
                order_detail = order_info_list[idx]
                logger.error(f"{result['status']} Order task: {order_detail}")
            else:
                logger.error(f"Order failed: {result}")

    async def cancel_all_order(self, api, token_ub, t):
        active_orders = await api.sendGetActiveOrder(token_ub)
        index_list = []
        if active_orders["status"] == "Success":
            for idx, instrument_info in enumerate(active_orders["instruments"]):
                if instrument_info['active_orders']:
                    for order in instrument_info['active_orders']:
                        index_list.append((instrument_info['instrument_name'] ,order['order_index']))
        else:
            logger.error(f"Failed to get active orders: {active_orders}")
        cancel_response = await asyncio.gather(*[api.sendCancel(token_ub, index[0], t, index[1]) for index in index_list])
        for idx, resp in enumerate(cancel_response):
            order_info = index_list[idx]
            if resp["status"] == "Success":
                logger.info(f"Cancelled order for {order_info[0]}: {order_info[1]}")
                self._cache_past_trade_price = self._cache_past_trade_price[self._cache_past_trade_price['index'] != order_info[1]]
            else:
                logger.error(f"Failed to cancel order for {order_info[0]}: {resp}\n {order_info}, {t}")
    
    async def work(self, api, token_ub, t):
        try:
            logger.info(f"Work time: {round(t)}")
            await self.update_market_data(api, token_ub)
            if int(t) % 5 == 0:
                await self.cancel_all_order(api, token_ub, t)
            await self.execute_trades(api, token_ub, t)
            logger.debug(f"current past_trade_price_df:{self._cache_past_trade_price}")

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
    
    while bot.day <= bot.running_days:
        logger.info(f"current day: {bot.day}")
        while ConvertToSimTime_us(bot.start_time, bot.time_ratio, bot.day, bot.running_time) <= -1:
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