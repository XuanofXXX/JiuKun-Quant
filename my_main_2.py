import asyncio
import aiohttp
import json
import time
import logging
import random
import sys
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from collections import deque
import lightgbm as lgb

# 设置日志
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

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

class AdvancedTradingStrategy:
    def __init__(self, lookback_period=60, num_stocks=5, max_position=2000):
        self.lookback_period = lookback_period
        self.num_stocks = num_stocks
        self.max_position = max_position
        self.price_history = {}
        self.volume_history = {}
        self.order_book_history = {}
        self.instruments = []
        self.model = None
        self.market_state = "normal"
        self.factor_data = {}

    def initialize_instruments(self, instruments):
        self.instruments = instruments
        for instrument in instruments:
            self.price_history[instrument] = deque(maxlen=self.lookback_period)
            self.volume_history[instrument] = deque(maxlen=self.lookback_period)
            self.order_book_history[instrument] = deque(maxlen=self.lookback_period)

    async def update_market_data(self, api, token_ub):
        try:
            all_lobs = await api.sendGetAllLimitOrderBooks(token_ub)
            if all_lobs["status"] != "Success":
                raise Exception("Failed to get LOB data")
            
            for i, instrument in enumerate(self.instruments):
                lob = all_lobs["lobs"][i]
                mid_price = (float(lob["askprice"][0]) + float(lob["bidprice"][-1])) / 2
                volume = float(lob["askvolume"][0]) + float(lob["bidvolume"][-1])
                
                self.price_history[instrument].append(mid_price)
                self.volume_history[instrument].append(volume)
                self.order_book_history[instrument].append(lob)

            self.update_market_state()
        except Exception as e:
            logger.error(f"Error updating market data: {str(e)}")

    def update_market_state(self):
        try:
            total_volatility = sum(self.calculate_volatility(inst) for inst in self.instruments)
            if total_volatility > 0.02:
                self.market_state = "volatile"
            elif total_volatility < 0.005:
                self.market_state = "calm"
            else:
                self.market_state = "normal"
        except Exception as e:
            logger.error(f"Error updating market state: {str(e)}")
            self.market_state = "normal"

    def calculate_factors(self, instrument):
        prices = np.array(self.price_history[instrument])
        volumes = np.array(self.volume_history[instrument])
        
        # 动量因子
        momentum = (prices[-1] / prices[0]) - 1 if prices[0] != 0 else 0
        
        # 波动率因子
        returns = np.diff(prices) / prices[:-1]
        volatility = np.std(returns)
        
        # 成交量趋势因子
        volume_trend = stats.linregress(range(len(volumes)), volumes).slope
        
        # 价格压力因子
        price_pressure = (prices[-1] - np.mean(prices)) / np.mean(prices)
        
        # RSI因子
        delta = np.diff(prices)
        gain = (delta > 0) * delta
        loss = (delta < 0) * -delta
        avg_gain = np.mean(gain)
        avg_loss = np.mean(loss)
        rs = avg_gain / avg_loss if avg_loss != 0 else 0
        rsi = 100 - (100 / (1 + rs))
        
        return {
            'momentum': momentum,
            'volatility': volatility,
            'volume_trend': volume_trend,
            'price_pressure': price_pressure,
            'rsi': rsi
        }

    def calculate_returns(self, instrument, n_days=5):
        prices = np.array(self.price_history[instrument])
        returns = (prices[n_days:] - prices[:-n_days]) / prices[:-n_days]
        return returns

    def update_factor_data(self):
        for instrument in self.instruments:
            factors = self.calculate_factors(instrument)
            returns = self.calculate_returns(instrument)
            
            if instrument not in self.factor_data:
                self.factor_data[instrument] = []
            
            self.factor_data[instrument].append({**factors, 'returns': returns[-1] if len(returns) > 0 else 0})

    def evaluate_factors(self):
        all_factors = []
        all_returns = []
        for instrument, data in self.factor_data.items():
            factors = pd.DataFrame(data)
            all_factors.append(factors.iloc[:, :-1])  # 除了 'returns' 列
            all_returns.append(factors['returns'])
        
        all_factors = pd.concat(all_factors, axis=0)
        all_returns = pd.concat(all_returns, axis=0)
        
        ic = all_factors.corrwith(all_returns)
        ir = ic / ic.std()
        sharpe = ic.mean() / ic.std() * np.sqrt(252)  # 假设一年有252个交易日
        
        return ic, ir, sharpe

    def train_model(self, pretrain=False):
        X = []
        y = []
        for instrument, data in self.factor_data.items():
            df = pd.DataFrame(data)
            X.append(df.iloc[:, :-1])  # 除了 'returns' 列
            y.append(df['returns'])
        
        X = pd.concat(X, axis=0)
        y = pd.concat(y, axis=0)
        
        params = {
            'objective': 'regression',
            'metric': 'mse',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': 0
        }
        
        if pretrain:
            # 使用全部数据进行预训练
            train_data = lgb.Dataset(X, label=y)
            self.model = lgb.train(params, train_data, num_boost_round=100)
        else:
            # 使用最新的数据进行增量学习（Adapter）
            if self.model is None:
                raise ValueError("Model not pretrained. Call train_model with pretrain=True first.")
            
            # 只使用最新的一批数据进行增量学习
            recent_data = lgb.Dataset(X.iloc[-100:], label=y.iloc[-100:])
            
            # 使用现有模型作为初始模型，进行少量的增量训练
            self.model = lgb.train(params, recent_data, num_boost_round=10, init_model=self.model)

    def predict_returns(self, factors):
        return self.model.predict(factors)

    def calculate_order_price(self, instrument, side, lob):
        try:
            spread = float(lob["askprice"][0]) - float(lob["bidprice"][-1])
            mid_price = (float(lob["askprice"][0]) + float(lob["bidprice"][-1])) / 2
            
            if self.market_state == "volatile":
                # 在波动性市场中更积极
                if side == "buy":
                    return mid_price - spread * 0.4
                else:
                    return mid_price + spread * 0.4
            elif self.market_state == "calm":
                # 在平静市场中更保守
                if side == "buy":
                    return mid_price - spread * 0.1
                else:
                    return mid_price + spread * 0.1
            else:
                # 正常市场条件
                if side == "buy":
                    return mid_price - spread * 0.25
                else:
                    return mid_price + spread * 0.25
        except Exception as e:
            logger.error(f"Error calculating order price for {instrument}: {str(e)}")
            return mid_price  # 出错时返回中间价

    def calculate_order_size(self, instrument, user_info, price):
        try:
            instrument_id = int(instrument[-3:])
            current_position = user_info["rows"][instrument_id]["remain_volume"]
            target_position = user_info["rows"][instrument_id]["target_volume"]
            
            if abs(current_position) >= self.max_position:
                return 0
            
            available_position = self.max_position - abs(current_position)
            order_size = min(abs(target_position - current_position), available_position)
            
            # 根据价格和波动性调整订单大小
            volatility = self.calculate_volatility(instrument)
            price_level = price / np.mean(self.price_history[instrument]) if len(self.price_history[instrument]) > 0 else 1
            
            if volatility > 0.02 or price_level > 1.05:
                order_size *= 0.5  # 在高波动性或高价格情况下减少订单大小
            elif volatility < 0.005 or price_level < 0.95:
                order_size *= 1.5  # 在低波动性或低价格情况下增加订单大小
            
            return int(order_size / 100) * 100  # 舍入到最接近的100的倍数
        except Exception as e:
            logger.error(f"Error calculating order size for {instrument}: {str(e)}")
            return 0

    def calculate_volatility(self, instrument):
        try:
            prices = list(self.price_history[instrument])
            if len(prices) < 2:
                return 0
            returns = np.diff(prices) / prices[:-1]
            return np.std(returns)
        except Exception as e:
            logger.error(f"Error calculating volatility for {instrument}: {str(e)}")
            return 0

    async def execute_trade(self, api, token_ub, user_info, t, instrument, side):
        try:
            lob = (await api.sendGetLimitOrderBook(token_ub, instrument))
            instrument_id = int(instrument[-3:])
            current_position = user_info["rows"][instrument_id]["remain_volume"]
            target_position = user_info["rows"][instrument_id]["target_volume"]
            
            if (side == "buy" and target_position > current_position) or (side == "sell" and target_position < current_position):
                price = self.calculate_order_price(instrument, side, lob)
                size = self.calculate_order_size(instrument, user_info, price)
                
                if size > 