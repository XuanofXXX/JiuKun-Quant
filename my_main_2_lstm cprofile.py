import asyncio
import aiohttp
import sys
import json
import time
import logging
import random
import torch
import cProfile
import pstats
import io
import os
from pstats import SortKey

import torch.nn as nn
import pandas as pd
import numpy as np
from scipy import stats
from collections import deque
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# from train_LSTM_more_factors import preprocess_dataframe


# 设置日志
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

file_handler = logging.FileHandler('quant_log.log', mode='w')
file_handler.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

logger.addHandler(file_handler)
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
        # self._cache_user_info = {}.
    
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

class LSTMTradingStrategy(BaseStrategy):
    def __init__(self, model_path="trained_lstm_stock_model.pth", lookback_period=20, sequence_length=10):
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
        self.profiler = cProfile.Profile()
        self.profile_counter = 0
        self.profile_interval = 100  # 每100次调用保存一次性能数据
        self.profile_dir = "performance_profiles"
        os.makedirs(self.profile_dir, exist_ok=True)

    def load_model(self, model_path):
        input_dim = 9  # 更新为新的特征数量
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
    
    def calculate_factors(self, df):
        return df
    
    def initialize_instruments(self, instruments):
        self.instruments = instruments
        for instrument in instruments:
            self.price_history[instrument] = deque(maxlen=self.lookback_period)
            self.volume_history[instrument] = deque(maxlen=self.lookback_period)
            self.order_book_history[instrument] = deque(maxlen=self.lookback_period)
            self.factor_history[instrument] = deque(maxlen=self.sequence_length)
            
    async def update_market_data(self, api, token_ub):
        try:
            all_lobs_list = await asyncio.gather(*[api.sendGetAllLimitOrderBooks(token_ub) for _ in range(3)])

            all_lobs = next((response for response in all_lobs_list 
                             if not isinstance(response, Exception) and response["status"] == "Success"), None)
            if all_lobs is None:
                raise Exception("Failed to get LOB data")
            
            new_data = self.convert_api_to_df_format(all_lobs["lobs"])
            
            if self._cache_lobs is None:
                self._union_lobs(new_data)
                return
            
            # 检查是否有新的时间戳
            if not self._cache_lobs.empty:
                max_existing_tick = self._cache_lobs['Tick'].max()
                new_data = new_data[new_data['Tick'] > max_existing_tick]
            
            if not new_data.empty:
                self._union_lobs(new_data)
                logger.info(f"Added {len(new_data)} new records at tick: {new_data['Tick'].max()}.")
            else:
                logger.info("No new data to add.")
            
            # 保持每只股票最新的 lookback_period 条记录
            if not self._cache_lobs.empty:
                self._cache_lobs = (self._cache_lobs.sort_values(['StockID', 'Tick'])
                    .groupby('StockID')
                    .tail(self.lookback_period)
                    .reset_index(drop=True))
            
        except Exception as e:
            logger.error(f"Error updating market data: {str(e)}")
    
    def calculate_order_price(self, instrument, side, lob, price_movement):
        mid_price = (float(lob["AskPrice1"].iloc[0]) + float(lob["BidPrice1"].iloc[0])) / 2
        spread = float(lob["AskPrice1"].iloc[0]) - float(lob["BidPrice1"].iloc[0])
        
        if side == "buy":
            return mid_price - spread * 0.25 * (1 - price_movement * 0.1)
        else:
            return mid_price + spread * 0.25 * (1 + price_movement * 0.1)

    def calculate_order_size(self, instrument, user_info, price, max_position=2000, t=0, pred_price_limit=None):
        instrument_id = int(instrument[-3:])
        current_position = user_info["rows"][instrument_id]["remain_volume"]
        target_position = user_info["rows"][instrument_id]["target_volume"]
        
        if abs(current_position) >= max_position:
            return 0
        
        # 基础订单大小
        base_order_size = min(abs(target_position - current_position), max_position - abs(current_position))
        
        # 根据距离目标仓位的远近调整订单大小
        position_diff_ratio = abs(target_position - current_position) / max_position
        size_multiplier = 1 + position_diff_ratio  # 差距越大，订单越大
        
        # 根据当前时间调整订单大小
        time_factor = min(1, max(0.1, t / 2700))  # 假设交易日为2700秒，随时间增加而增大
        
        # 根据预测价格调整订单大小
        price_factor = 1
        if pred_price_limit and price:
            price_diff = abs(pred_price_limit - price) / price
            price_factor = 1 + min(1, price_diff * 10)  # 价格差异越大，订单越大，但最多翻倍
        
        # 计算最终订单大小
        adjusted_size = base_order_size * size_multiplier * time_factor * price_factor
        
        # 确保订单大小不超过最大允许仓位
        final_size = min(adjusted_size, max_position - abs(current_position))
        
        # 将订单大小四舍五入到最接近的100的倍数
        return int(round(final_size / 100) * 100)
    
    async def execute_trades(self, api, token_ub, user_info, t):
        try:
            if isinstance(t, float):
                t = int(t)
            stock_df = self._cache_lobs
            stock_df = self.calculate_factors(stock_df)
            input_tensor = self.factorDF2Tensor(stock_df)
            
            input_tensor = input_tensor[:, :, :9]
        
            with torch.no_grad():
                predictions = self.model(input_tensor).numpy().flatten()

            prediction_df = pd.DataFrame({
                'instrument': self.instruments,
                'instrument_id': [int(instr[-3:]) for instr in self.instruments],
                'prediction': predictions / 100000
            })
            
                    # Add target position information
            user_info_df = pd.DataFrame(user_info['rows'])
            prediction_df = prediction_df.merge(user_info_df[['remain_volume', 'target_volume']], 
                                                left_on='instrument_id', right_index=True)
            
            prediction_df['valid_trade'] = (
                ((prediction_df['prediction'] > 0) & (prediction_df['target_volume'] > 0)) | \
                ((prediction_df['prediction'] < 0) & (prediction_df['target_volume'] < 0))
            )
            
            valid_trades = prediction_df[prediction_df['valid_trade']]
            
            # Sort by absolute prediction value and select top candidates
            buy_candidates = valid_trades[valid_trades['prediction'] > 0].nlargest(10, 'prediction')
            sell_candidates = valid_trades[valid_trades['prediction'] < 0].nsmallest(10, 'prediction')

            order_tasks = []

            for side, candidates in [("buy", buy_candidates), ("sell", sell_candidates)]:
                for _, row in candidates.iterrows():
                    instrument = row['instrument']
                    logger.debug(f"Processing {instrument} for {side} trade")
                    lob = stock_df[(stock_df['StockID'] == instrument) & (stock_df['Tick'] == t)]
                    
                    # logger.debug(f"LOB data for {instrument} at tick {t}: {lob}")
                    if lob.empty:
                        logger.warning(f"No LOB data for {instrument} at tick {t}. Skipping.")
                        logger.debug(f'LOB at tick{t} is {stock_df[stock_df["Tick"] == t]}')
                        logger.debug(f'LOB at instrument {instrument} is {stock_df[stock_df["StockID"] == instrument]}')
                        continue

                    price = self.calculate_order_price(instrument, side, lob, row['prediction'])
                    size = min(abs(row['target_volume']), 
                            self.calculate_order_size(instrument, user_info, price))

                    # logger.debug(f"Calculated for {instrument}: side:{side}, price:{price}, size:{size}, prediction:{row['prediction']}")

                    if size > 0:
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
        self.profiler.enable()
        try:
            user_info = await api.sendGetUserInfo(token_ub)
            if t >= 2950 and t < 3000:  # 在交易日最后50秒执行EOD策略
                await self.execute_eod_strategy(api, token_ub, user_info, t)
            else:
                await self.update_market_data(api, token_ub)
                await self.execute_trades(api, token_ub, user_info, t)
        except Exception as e:
            logger.error(f"Error in work method: {str(e)}")
        finally:
            self.profiler.disable()
            s = io.StringIO()
            sortby = SortKey.CUMULATIVE
            ps = pstats.Stats(self.profiler, stream=s).sort_stats(sortby)
            ps.print_stats(10)  # 打印前10个最耗时的函数
            logger.info(f"Performance profile:\n{s.getvalue()}")


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