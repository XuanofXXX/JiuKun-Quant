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
from collections import deque
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

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


logging.basicConfig(level=logging.DEBUG, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

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
    def __init__(self, model_path="trained_lstm_stock_model.pth", lookback_period=20, sequence_length=10):
        self.model = self.load_model(model_path)
        self.scaler = StandardScaler()
        self.lookback_period = lookback_period
        self.sequence_length = sequence_length
        self.price_history = {}
        self.volume_history = {}
        self.order_book_history = {}
        self.factor_history = {}
        self.instruments = []

    def load_model(self, model_path):
        input_dim = 9  # number of features
        hidden_dim = 32
        num_layers = 2
        output_dim = 1
        model = LSTMModel(input_dim, hidden_dim, num_layers, output_dim)
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        return model

    def initialize_instruments(self, instruments):
        self.instruments = instruments
        for instrument in instruments:
            self.price_history[instrument] = deque(maxlen=self.lookback_period)
            self.volume_history[instrument] = deque(maxlen=self.lookback_period)
            self.order_book_history[instrument] = deque(maxlen=self.lookback_period)
            self.factor_history[instrument] = deque(maxlen=self.sequence_length)

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

                factors = self.calculate_factors(instrument)
                self.factor_history[instrument].append(factors)

        except Exception as e:
            logger.error(f"Error updating market data: {str(e)}")

    def calculate_factors(self, instrument):
        if len(self.price_history[instrument]) < self.lookback_period:
            return np.zeros(9)  # Return zeros if not enough data

        df = pd.DataFrame({
            'price': list(self.price_history[instrument]),
            'volume': list(self.volume_history[instrument])
        })
        
        factors = pd.DataFrame({
            'price_momentum_short': df['price'].pct_change(5),
            'price_momentum_mid': df['price'].pct_change(30),
            'volume_change_short': df['volume'].pct_change(5),
            'volume_change_mid': df['volume'].pct_change(30),
            'spread': (self.order_book_history[instrument][-1]['askprice'][0] - self.order_book_history[instrument][-1]['bidprice'][-1]) / df['price'].iloc[-1],
            'depth_imbalance': (self.order_book_history[instrument][-1]['bidvolume'][0] - self.order_book_history[instrument][-1]['askvolume'][0]) / (self.order_book_history[instrument][-1]['bidvolume'][0] + self.order_book_history[instrument][-1]['askvolume'][0]),
            'volatility': df['price'].rolling(30).std() / df['price'],
            'rsi': self.calculate_rsi(df['price']),
            'ma_cross': df['price'].rolling(window=5).mean() / df['price'].rolling(window=30).mean() - 1
        })
        
        return factors.iloc[-1].values

    def calculate_rsi(self, prices, period=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs)).iloc[-1]

    def predict_price_movement(self, instrument):
        if len(self.factor_history[instrument]) < self.sequence_length:
            return 0
        
        sequence = np.array(list(self.factor_history[instrument]))
        scaled_sequence = self.scaler.fit_transform(sequence)
        
        with torch.no_grad():
            input_tensor = torch.tensor(scaled_sequence).unsqueeze(0).float()
            prediction = self.model(input_tensor).item()
        
        return prediction

    def calculate_order_price(self, instrument, side, lob):
        price_movement = self.predict_price_movement(instrument)
        mid_price = (float(lob["askprice"][0]) + float(lob["bidprice"][-1])) / 2
        spread = float(lob["askprice"][0]) - float(lob["bidprice"][-1])
        
        if side == "buy":
            return mid_price - spread * 0.25 * (1 - price_movement * 0.1)
        else:
            return mid_price + spread * 0.25 * (1 + price_movement * 0.1)

    def calculate_order_size(self, instrument, user_info, price, max_position=2000):
        instrument_id = int(instrument[-3:])
        current_position = user_info["rows"][instrument_id]["remain_volume"]
        target_position = user_info["rows"][instrument_id]["target_volume"]
        
        if abs(current_position) >= max_position:
            return 0
        
        available_position = max_position - abs(current_position)
        order_size = min(abs(target_position - current_position), available_position)
        
        return int(order_size / 100) * 100  # Round to nearest 100

    async def execute_trades(self, api, token_ub, user_info, t):
        try:
            order_tasks = []
            for instrument in self.instruments:
                lob = (await api.sendGetAllLimitOrderBooks(token_ub))["lobs"][self.instruments.index(instrument)]
                instrument_id = int(instrument[-3:])
                current_position = user_info["rows"][instrument_id]["remain_volume"]
                target_position = user_info["rows"][instrument_id]["target_volume"]
                
                price_movement = self.predict_price_movement(instrument)
                if price_movement != 0:
                    logger.debug(f"Predict_price_movement for {instrument}: {price_movement}")
                
                if price_movement > 0.001 and current_position < target_position:
                    side = "buy"
                elif price_movement < -0.001 and current_position > target_position:
                    side = "sell"
                else:
                    continue
                
                price = self.calculate_order_price(instrument, side, lob)
                size = self.calculate_order_size(instrument, user_info, price)
                
                logger.debug(f"Calculated for {instrument}: price:{price}, size:{size}")
                
                if size > 0:
                    order_tasks.append(api.sendOrder(token_ub, instrument, t, side, price, size))
            
            # 并发执行所有订单任务
            results = await asyncio.gather(*order_tasks, return_exceptions=True)
            
            # 处理结果
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Order failed for {self.instruments[i]}: {str(result)}")
                elif result["status"] != "Success":
                    logger.error(f"Order failed for {self.instruments[i]}: {result}")
                else:
                    logger.info(f"Order placed successfully for {self.instruments[i]}: {result}")
        
        except Exception as e:
            logger.error(f"Error executing trades: {str(e)}")
    
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
            user_info = await api.sendGetUserInfo(token_ub)
            if t >= 2950 and t < 3000:  # 在交易日最后50秒执行EOD策略
                await self.execute_eod_strategy(api, token_ub, user_info, t)
            else:
                await self.update_market_data(api, token_ub)
                await self.execute_trades(api, token_ub, user_info, t)
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
        await self.GetInstruments()
        
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
            while ConvertToSimTime_us(bot.start_time, bot.time_ratio, bot.day, bot.running_time) < s:
                await asyncio.sleep(0.001)
            t = ConvertToSimTime_us(bot.start_time, bot.time_ratio, bot.day, bot.running_time)
            logger.info(f"Work Time: {s} {t}")
            if t < SimTimeLen - 30:
                await bot.work()
        bot.day += 1

    await bot.api.close_session()

if __name__ == "__main__":
    username = 'UBIQ_TEAM179'
    password = 'ANfgOwr3SvpN'
    asyncio.run(main(username, password))