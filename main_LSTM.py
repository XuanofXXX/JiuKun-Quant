import asyncio
import aiohttp
import logging
import time
import re
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from collections import deque
from sklearn.preprocessing import StandardScaler
from scipy.stats import linregress

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

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
        return await self.send_request("/TradeAPI/Order", data)

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

class LSTMTradingStrategy:
    def __init__(self, model_path, sequence_length=10, lookback_period=30):
        self.model = self.load_model(model_path)
        self.scaler = StandardScaler()
        self.sequence_length = sequence_length
        self.lookback_period = lookback_period
        self.price_history = {}
        self.volume_history = {}
        self.factor_history = {}
        self.high_history = {}
        self.low_history = {}

    def load_model(self, model_path):
        input_dim = 26  # 根据您的因子数量调整
        hidden_dim = 32
        num_layers = 2
        output_dim = 1
        model = LSTMModel(input_dim, hidden_dim, num_layers, output_dim)
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        return model

    def initialize_instruments(self, instruments):
        for instrument in instruments:
            self.price_history[instrument] = deque(maxlen=self.lookback_period)
            self.volume_history[instrument] = deque(maxlen=self.lookback_period)
            self.factor_history[instrument] = deque(maxlen=self.sequence_length)
            self.high_history[instrument] = deque(maxlen=self.lookback_period)
            self.low_history[instrument] = deque(maxlen=self.lookback_period)

    def update_data(self, instrument, price, volume, high, low):
        self.price_history[instrument].append(price)
        self.volume_history[instrument].append(volume)
        self.high_history[instrument].append(high)
        self.low_history[instrument].append(low)
        
        price_series = pd.Series(self.price_history[instrument])
        volume_series = pd.Series(self.volume_history[instrument])
        high_series = pd.Series(self.high_history[instrument])
        low_series = pd.Series(self.low_history[instrument])
        
        print(price_series)
        price_series.to_csv('price_series.csv', index=False)
        factors = self.calculate_factors(price_series, volume_series, high_series, low_series)
        self.factor_history[instrument].append(factors.iloc[-1].values)

    def calculate_factors(self, price_series, volume_series=None, high_series=None, low_series=None, window_short=5, window_medium=10, window_long=30):
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
        
        expected_factors  = [
            'momentum_short',
            'momentum_medium',
            'momentum_long',
            'ma_short',
            'ma_medium',
            'ma_long',
            'ma_cross_short_medium',
            'ma_cross_short_long',
            'ma_cross_medium_long',
            'volatility_short',
            'volatility_medium',
            'volatility_long',
            'rsi',
            'bollinger_upper',
            'bollinger_lower',
            'bollinger_percent',
            'price_acceleration',
            'volume_momentum',
            'volume_ma_ratio',
            'price_volume_corr',
            'mfi',
            'atr',
            'channel_upper',
            'channel_lower',
            'channel_position',
            'trend_strength'
        ]
        
        for factor in expected_factors:
            if factor not in factors.columns:
                factors[factor] = 0  # 或者使用其他合适的默认值

        factors = factors.replace([np.inf, -np.inf], np.nan).fillna(0)

        return factors[expected_factors]  # 只返回预期的因子

    def predict(self, instrument):
        if len(self.factor_history[instrument]) < self.sequence_length:
            logger.warning(f"Not enough factor history for {instrument}. Current: {len(self.factor_history[instrument])}, Required: {self.sequence_length}")
            return 0
        
        sequence = np.array(list(self.factor_history[instrument]))
        scaled_sequence = self.scaler.fit_transform(sequence)
        
        with torch.no_grad():
            input_tensor = torch.tensor(scaled_sequence).unsqueeze(0).float()
            prediction = self.model(input_tensor).item()
        
        logger.info(f"Prediction for {instrument}: {prediction}")
        return prediction

    def make_decision(self, instrument, current_price, current_position, target_position):
        prediction = self.predict(instrument)
        
        threshold = 50  # 设置一个小的阈值
        if prediction > threshold and target_position > 0:
            return "buy", min(current_position , 1000)
        elif prediction < -threshold and target_position < 0:
            return "sell", min(current_position , 1000)
        else:
            return None, 0

    async def execute_trade(self, api, token_ub, instrument, action, volume, current_price, t):
        if action == "buy":
            price = round(current_price * 1.01, 2)  # 略高于当前价格买入
        elif action == "sell":
            price = round(current_price * 0.99, 2)  # 略低于当前价格卖出
        else:
            return
        
        response = await api.sendOrder(token_ub, instrument, t, action, price, volume)
        if response["status"] == "Success":
            logger.info(f"Order placed for {instrument}: {action} {volume} @ {price}")
        else:
            logger.error(f"Order failed for {instrument}: {response}")

    async def trade(self, api, token_ub, instruments, t):
        for instrument in instruments:
            try:
                # 获取当前市场数据
                lob_response = await api.sendGetLimitOrderBook(token_ub, instrument)
                if lob_response["status"] != "Success" or "lob" not in lob_response:
                    logger.error(f"Failed to get LOB for {instrument}: {lob_response}")
                    continue

                lob = lob_response["lob"]
                if not lob or "askprice" not in lob or "bidprice" not in lob:
                    logger.error(f"Invalid LOB data for {instrument}: {lob}")
                    continue

                ask_price = float(lob["askprice"][0])
                bid_price = float(lob["bidprice"][-1])
                mid_price = (ask_price + bid_price) / 2
                volume = float(lob["askvolume"][0]) + float(lob["bidvolume"][-1])
                
                # 使用 ask_price 作为 high，bid_price 作为 low
                self.update_data(instrument, mid_price, volume, ask_price, bid_price)
                
                # 获取当前持仓信息
                user_info_response = await api.sendGetUserInfo(token_ub)
                if user_info_response["status"] != "Success":
                    logger.error(f"Failed to get user info: {user_info_response}")
                    continue

                user_info = user_info_response["rows"]
                instrument_id = int(instrument[-3:])

                current_position = abs(user_info[instrument_id]["remain_volume"]) - abs(user_info[instrument_id]["frozen_volume"])
                target_position = user_info[instrument_id]["target_volume"]
                
                # 做出交易决策
                action, volume = self.make_decision(instrument, mid_price, current_position, target_position)
                
                logger.info(f"Decision for {instrument}: action={action}, volume={volume}, current_position={current_position}, target_position={target_position}")
                
                # 执行交易
                if action and volume > 0:
                    await self.execute_trade(api, token_ub, instrument, action, volume, mid_price, t)
                else:
                    logger.info(f"No trade needed for {instrument}")

            except Exception as e:
                logger.error(f"Error processing {instrument}: {str(e)}")

class LSTMTradingBot:
    def __init__(self, username, password, port):
        self.username = username
        self.password = password
        self.api = AsyncInterfaceClass(f"http://8.147.116.35:{port}")
        self.token_ub = None
        self.instruments = []
        self.strategy = LSTMTradingStrategy("./trained_lstm_stock_model.pth")

    async def login(self):
        response = await self.api.sendLogin(self.username, self.password)
        if response["status"] == "Success":
            self.token_ub = response["token_ub"]
            logger.info(f"Login Success: {self.token_ub}")
        else:
            logger.error(f"Login Error: {response}")
            raise Exception("Login failed")

    async def get_instruments(self):
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
            await self.get_instruments()
            
            response = await self.api.sendGetGameInfo(self.token_ub)
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
        await self.strategy.trade(self.api, self.token_ub, self.instruments, t)

async def main(username, password):
    bot = LSTMTradingBot(username, password, 30020)
    try:
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
                while ConvertToSimTime_us(bot.start_time, bot.time_ratio, bot.day, bot.running_time) < s:
                    await asyncio.sleep(0.001)
                t = ConvertToSimTime_us(bot.start_time, bot.time_ratio, bot.day, bot.running_time)
                logger.info(f"Work Time: {s} {t}")
                if t < SimTimeLen - 30:
                    await bot.work()
            bot.day += 1
    except Exception as e:
        logger.error(f"An error occurred in main loop: {str(e)}")
    finally:
        await bot.api.close_session()
        logger.info("Session closed")

if __name__ == "__main__":
    username = 'UBIQ_TEAM179'
    password = 'ANfgOwr3SvpN'
    try:
        asyncio.run(main(username, password))
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")