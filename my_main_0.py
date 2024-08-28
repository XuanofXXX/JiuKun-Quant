import asyncio
import aiohttp
import json
import time
import logging
import random
import sys
import numpy as np
from scipy import stats

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
    # TODO  每一次访问instrument的时候有last_price, trade_volume, trade_value, 和twap 这么多信息能不能用起来
    # TODO 没有针对取消挂单的策略
    # TODO 可以增加cache，防止多次相同请求，耗费时间
    def __init__(self, lookback_period=10):
        self.lookback_period = lookback_period
        self.price_history = {}
        self.volume_history = {}
        self.instruments = []

    def initialize_instruments(self, instruments: str):
        self.instruments = instruments
        for instrument in instruments:
            self.price_history[instrument] = []
            self.volume_history[instrument] = []

    async def update_market_data(self, api, token_ub):
        all_lobs = await api.sendGetAllLimitOrderBooks(token_ub)
        if all_lobs["status"] != "Success":
            raise Exception("Failed to get LOB data")
        
        for i, instrument in enumerate(self.instruments):
            lob = all_lobs["lobs"][i]
            # TODO 这里只选择了最优bid和ask的，但是收到的有前十个bid和ask？
            mid_price = (float(lob["askprice"][0]) + float(lob["bidprice"][-1])) / 2
            volume = float(lob["askvolume"][0]) + float(lob["bidvolume"][-1])
            
            self.price_history[instrument].append(mid_price)
            self.volume_history[instrument].append(volume)
            
            if len(self.price_history[instrument]) > self.lookback_period:
                self.price_history[instrument].pop(0)
                self.volume_history[instrument].pop(0)

    def calculate_momentum(self, instrument: str):
        prices = self.price_history[instrument]
        if len(prices) < self.lookback_period:
            return 0
        return (prices[-1] / prices[0]) - 1

    def calculate_volatility(self, instrument: str):
        prices = self.price_history[instrument]
        if len(prices) < self.lookback_period:
            return 0
        return np.std(prices) / np.mean(prices)

    def calculate_volume_trend(self, instrument: str):
        volumes = self.volume_history[instrument]
        if len(volumes) < self.lookback_period:
            return 0
        slope, _, _, _, _ = stats.linregress(range(len(volumes)), volumes)
        return slope

    def select_stocks(self, n=60):
        scores = []
        # TODO 没有看到买卖的区别（）
        for instrument in self.instruments:
            momentum = self.calculate_momentum(instrument)
            volatility = self.calculate_volatility(instrument)
            volume_trend = self.calculate_volume_trend(instrument)
            
            score = momentum + volume_trend - volatility
            scores.append((instrument, score))
        
        return [instrument for instrument, _ in sorted(scores, key=lambda x: x[1], reverse=True)[:n]]

    def calculate_order_price(self, instrument, side, lob):
        spread = float(lob["askprice"][0]) - float(lob["bidprice"][-1])
        mid_price = (float(lob["askprice"][0]) + float(lob["bidprice"][-1])) / 2
        
        if side == "buy":
            return mid_price - spread * 0.25
        else:
            return mid_price + spread * 0.25

    def calculate_order_size(self, instrument: str, user_info, max_position=2000, order_size_limit=100):
        instrument_id = int(instrument[-3:])
        tradable_position = abs(user_info["rows"][instrument_id]["remain_volume"]) - abs(user_info["rows"][instrument_id]["frozen_volume"])
        if tradable_position == 0:
            return 0
        target_position = user_info["rows"][instrument_id]["target_volume"]
        direction = 'buy' if target_position > 0 else 'sell'
        
        target_position = abs(target_position)

        order_size = max(abs(tradable_position), order_size_limit)
        order_size = (order_size / 100) * 100
        if order_size == 0:
            logger.debug(f"Volume: 0 | target_position: {target_position} | tradable_position: {tradable_position}")
        return order_size

    async def execute_trades(self, api, token_ub, user_info, t):
        selected_stocks = self.select_stocks()
        
        for instrument in selected_stocks:
            lob = (await api.sendGetAllLimitOrderBooks(token_ub))["lobs"][self.instruments.index(instrument)]
            logger.debug(f"The instrument is {instrument}")
            instrument_id = instrument2id(instrument)
            target_position = user_info["rows"][instrument_id]["target_volume"]
            tradable_position = abs(user_info["rows"][instrument_id]["remain_volume"]) - abs(user_info["rows"][instrument_id]["frozen_volume"])
            
            if target_position > 0:
                side = "buy"
            elif target_position < 0:
                side = "sell"
            else:
                continue
            
            price = self.calculate_order_price(instrument, side, lob)
            size = abs(self.calculate_order_size(instrument, user_info))
            
            price = round(price, 2)
            size = int(abs(size))
            
            response = await api.sendOrder(token_ub, instrument, t, side, price, size)
            
            if response["status"] != "Success":
                logger.error(f"Order failed for {instrument}: {response}")
                logger.debug(f"volume: {size}, tradable_position: {tradable_position}")
            else:
                logger.info(f"Order placed for {instrument}: {side} {size} @ {price}")

    async def work(self, api, token_ub, t):
        try:
            await self.update_market_data(api, token_ub)
            user_info = await api.sendGetUserInfo(token_ub)
            await self.execute_trades(api, token_ub, user_info, t)
        except Exception as e:
            logger.error(f"Error in work method: {str(e)}")

class ImprovedAdvancedTradingStrategy:
    def __init__(self, lookback_period=10):
        self.lookback_period = lookback_period
        self.price_history = {}
        self.volume_history = {}
        self.twap_history = {}
        self.instruments = []

    def initialize_instruments(self, instruments):
        self.instruments = instruments
        for instrument in instruments:
            self.price_history[instrument] = []
            self.volume_history[instrument] = []
            self.twap_history[instrument] = []

    async def update_market_data(self, api, token_ub):
        all_lobs = await api.sendGetAllLimitOrderBooks(token_ub)
        if all_lobs["status"] != "Success":
            raise Exception("Failed to get LOB data")

        for i, instrument in enumerate(self.instruments):
            lob = all_lobs["lobs"][i]
            mid_price = (float(lob["askprice"][0]) + float(lob["bidprice"][-1])) / 2
            volume = float(lob["askvolume"][0]) + float(lob["bidvolume"][-1])
            twap = float(lob.get("twap", mid_price))  # Use mid_price if twap is not available

            self.price_history[instrument].append(mid_price)
            self.volume_history[instrument].append(volume)
            self.twap_history[instrument].append(twap)

            if len(self.price_history[instrument]) > self.lookback_period:
                self.price_history[instrument].pop(0)
                self.volume_history[instrument].pop(0)
                self.twap_history[instrument].pop(0)

    def calculate_momentum(self, instrument):
        prices = self.price_history[instrument]
        if len(prices) < self.lookback_period:
            return 0
        return (prices[-1] / prices[0]) - 1

    def calculate_volatility(self, instrument):
        prices = self.price_history[instrument]
        if len(prices) < self.lookback_period:
            return 0
        return np.std(prices) / np.mean(prices)

    def calculate_volume_trend(self, instrument):
        volumes = self.volume_history[instrument]
        if len(volumes) < self.lookback_period:
            return 0
        slope, _, _, _, _ = stats.linregress(range(len(volumes)), volumes)
        return slope

    def calculate_twap_deviation(self, instrument):
        prices = self.price_history[instrument]
        twaps = self.twap_history[instrument]
        if len(prices) < self.lookback_period or len(twaps) < self.lookback_period:
            return 0
        return (prices[-1] / twaps[-1]) - 1

    def select_stocks(self, user_info, n=60):
        scores = []
        for instrument in self.instruments:
            momentum = self.calculate_momentum(instrument)
            volatility = self.calculate_volatility(instrument)
            volume_trend = self.calculate_volume_trend(instrument)
            twap_deviation = self.calculate_twap_deviation(instrument)
            
            instrument_id = int(instrument[-3:])
            target_position = user_info["rows"][instrument_id]["target_volume"]
            current_position = user_info["rows"][instrument_id]["remain_volume"]
            position_distance = abs(target_position - current_position) / max(abs(target_position), 1)

            score = momentum + volume_trend - volatility + twap_deviation + position_distance
            scores.append((instrument, score))
        
        return [instrument for instrument, _ in sorted(scores, key=lambda x: x[1], reverse=True)[:n]]

    def calculate_order_price(self, instrument, side, lob):
        spread = float(lob["askprice"][0]) - float(lob["bidprice"][-1])
        mid_price = (float(lob["askprice"][0]) + float(lob["bidprice"][-1])) / 2
        twap = float(lob.get("twap", mid_price))
        
        if side == "buy":
            return min(mid_price - spread * 0.25, twap)
        else:
            return max(mid_price + spread * 0.25, twap)

    def calculate_order_size(self, instrument, user_info, remaining_time, max_position=2000):
        instrument_id = int(instrument[-3:])
        target_position = user_info["rows"][instrument_id]["target_volume"]
        current_position = user_info["rows"][instrument_id]["remain_volume"]
        frozen_volume = user_info["rows"][instrument_id]["frozen_volume"]

        tradable_position = target_position - current_position - frozen_volume
        
        # Adjust order size based on remaining time
        time_factor = max(0.1, min(1, remaining_time / 300))  # Assuming 300 seconds trading time
        base_order_size = abs(tradable_position) * (1 - time_factor)

        # Ensure order size is within limits and rounded to 100
        order_size = min(max(100, base_order_size), max_position)
        order_size = round(order_size / 100) * 100

        return int(order_size)

    async def execute_trades(self, api, token_ub, user_info, t, remaining_time):
        selected_stocks = self.select_stocks(user_info)
        all_lobs = await api.sendGetAllLimitOrderBooks(token_ub)
        
        for instrument in selected_stocks:
            lob = all_lobs["lobs"][self.instruments.index(instrument)]
            instrument_id = int(instrument[-3:])
            target_position = user_info["rows"][instrument_id]["target_volume"]
            current_position = user_info["rows"][instrument_id]["remain_volume"]
            
            if target_position == current_position:
                continue
            
            side = "buy" if target_position > current_position else "sell"
            
            price = self.calculate_order_price(instrument, side, lob)
            size = self.calculate_order_size(instrument, user_info, remaining_time)
            
            price = round(price, 2)
            
            response = await api.sendOrder(token_ub, instrument, t, side, price, size)
            
            if response["status"] != "Success":
                print(f"Order failed for {instrument}: {response}")
            else:
                print(f"Order placed for {instrument}: {side} {size} @ {price}")

    async def cancel_stale_orders(self, api, token_ub, t, max_order_age=60):
        user_info = await api.sendGetUserInfo(token_ub)
        for instrument in self.instruments:
            instrument_id = int(instrument[-3:])
            orders = user_info["rows"][instrument_id]["orders"]
            for order in orders:
                if t - order["localtime"] > max_order_age:
                    await api.sendCancel(token_ub, instrument, t, order["index"])

    async def work(self, api, token_ub, t, total_time=300):
        try:
            await self.update_market_data(api, token_ub)
            user_info = await api.sendGetUserInfo(token_ub)
            remaining_time = max(0, total_time - t)
            await self.execute_trades(api, token_ub, user_info, t, remaining_time)
            await self.cancel_stale_orders(api, token_ub, t)
        except Exception as e:
            print(f"Error in work method: {str(e)}")

class AsyncBotsDemoClass:
    def __init__(self, username, password, port):
        self.username = username
        self.password = password
        self.api = AsyncInterfaceClass(f"http://8.147.116.35:{port}")
        self.token_ub = None
        self.instruments = []
        self.strategy = AdvancedTradingStrategy()

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