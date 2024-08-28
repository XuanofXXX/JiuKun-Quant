import asyncio
import aiohttp
import json
import time
import logging
import random
import sys
import numpy as np
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from collections import deque

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


logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

class AdvancedTradingStrategy:
    def __init__(self, lookback_period=20, num_stocks=5, max_position=2000):
        """
        初始化交易策略
        :param lookback_period: 历史数据回顾期
        :param num_stocks: 选择的股票数量
        :param max_position: 每个股票的最大持仓量
        """
        self.lookback_period = lookback_period
        self.num_stocks = num_stocks
        self.max_position = max_position
        self.price_history = {}  # 存储每个股票的价格历史
        self.volume_history = {}  # 存储每个股票的成交量历史
        self.order_book_history = {}  # 存储每个股票的订单簿历史
        self.instruments = []  # 存储所有可交易的股票代码
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)  # 初始化随机森林模型
        self.market_state = "normal"  # 市场状态，可以是 "normal", "volatile", 或 "calm"
        
    def initialize_instruments(self, instruments):
        """
        初始化可交易的股票列表，并为每个股票创建历史数据存储
        :param instruments: 股票代码列表
        """
        self.instruments = instruments
        for instrument in instruments:
            self.price_history[instrument] = deque(maxlen=self.lookback_period)
            self.volume_history[instrument] = deque(maxlen=self.lookback_period)
            self.order_book_history[instrument] = deque(maxlen=self.lookback_period)

    async def update_market_data(self, api, token_ub):
        """
        更新市场数据，包括价格、成交量和订单簿
        :param api: API接口对象
        :param token_ub: 用户token
        """
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
        """
        根据当前市场波动性更新市场状态
        """
        try:
            total_volatility = sum(self.calculate_volatility(inst) for inst in self.instruments)
            if total_volatility > 0.02:  # 高波动性阈值
                self.market_state = "volatile"
            elif total_volatility < 0.005:  # 低波动性阈值
                self.market_state = "calm"
            else:
                self.market_state = "normal"
        except Exception as e:
            logger.error(f"Error updating market state: {str(e)}")
            self.market_state = "normal"

    def calculate_features(self, instrument):
        """
        计算给定股票的特征
        :param instrument: 股票代码
        :return: 特征字典，包括动量、波动性、成交量趋势和订单簿不平衡
        """
        try:
            prices = list(self.price_history[instrument])
            volumes = list(self.volume_history[instrument])
            
            if len(prices) < self.lookback_period:
                return None
            
            returns = np.diff(prices) / prices[:-1]
            
            features = {
                "momentum": (prices[-1] / prices[0]) - 1 if prices[0] != 0 else 0,
                "volatility": np.std(returns),
                "volume_trend": stats.linregress(range(len(volumes)), volumes).slope,
                "order_book_imbalance": self.calculate_order_book_imbalance(instrument)
            }
            
            return features
        except Exception as e:
            logger.error(f"Error calculating features for {instrument}: {str(e)}")
            return None

    def calculate_order_book_imbalance(self, instrument):
        """
        计算订单簿的买卖不平衡程度
        :param instrument: 股票代码
        :return: 不平衡指标，范围为 [-1, 1]
        """
        try:
            lob = self.order_book_history[instrument][-1]
            bid_volume = sum(float(vol) for vol in lob["bidvolume"])
            ask_volume = sum(float(vol) for vol in lob["askvolume"])
            total_volume = bid_volume + ask_volume
            return (bid_volume - ask_volume) / total_volume if total_volume != 0 else 0
        except Exception as e:
            logger.error(f"Error calculating order book imbalance for {instrument}: {str(e)}")
            return 0

    def train_model(self):
        """
        使用历史数据训练随机森林模型
        """
        try:
            start_time = time.time()
            X = []
            y = []
            for instrument in self.instruments:
                features = self.calculate_features(instrument)
                if features is not None:
                    X.append(list(features.values()))
                    y.append(self.price_history[instrument][-1] / self.price_history[instrument][0] - 1)
            
            if X and y:
                self.model.fit(X, y)
            end_time = time.time()
            training_time = end_time - start_time
            logger.info(f"Model training completed in {training_time:.4f} seconds")
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")

    def select_stocks(self):
        """
        选择表现最好的股票
        :return: 选中的股票代码列表
        """
        try:
            self.train_model()
            
            scores = []
            for instrument in self.instruments:
                features = self.calculate_features(instrument)
                if features is not None:
                    score = self.model.predict([list(features.values())])[0]
                    scores.append((instrument, score))
            
            return [instrument for instrument, _ in sorted(scores, key=lambda x: x[1], reverse=True)[:self.num_stocks]]
        except Exception as e:
            logger.error(f"Error selecting stocks: {str(e)}")
            return []

    def calculate_order_price(self, instrument, side, lob):
        """
        根据当前市场状态和买卖方向计算订单价格
        :param instrument: 股票代码
        :param side: 买卖方向 ("buy" 或 "sell")
        :param lob: 限价订单簿数据
        :return: 计算得出的订单价格
        """
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
        """
        计算订单大小
        :param instrument: 股票代码
        :param user_info: 用户信息，包含当前持仓等
        :param price: 订单价格
        :return: 计算得出的订单大小
        """
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
        """
        计算给定股票的波动性
        :param instrument: 股票代码
        :return: 波动性指标
        """
        try:
            prices = list(self.price_history[instrument])
            if len(prices) < 2:
                return 0
            returns = np.diff(prices) / prices[:-1]
            return np.std(returns)
        except Exception as e:
            logger.error(f"Error calculating volatility for {instrument}: {str(e)}")
            return 0

    async def execute_trades(self, api, token_ub, user_info, t):
        """
        执行交易逻辑
        :param api: API接口对象
        :param token_ub: 用户token
        :param user_info: 用户信息
        :param t: 当前时间戳
        """
        try:
            selected_stocks = self.select_stocks()
            
            for instrument in selected_stocks:
                lob = (await api.sendGetAllLimitOrderBooks(token_ub))["lobs"][self.instruments.index(instrument)]
                instrument_id = int(instrument[-3:])
                current_position = user_info["rows"][instrument_id]["remain_volume"]
                target_position = user_info["rows"][instrument_id]["target_volume"]
                
                if target_position > current_position:
                    side = "buy"
                elif target_position < current_position:
                    side = "sell"
                else:
                    continue
                
                price = self.calculate_order_price(instrument, side, lob)
                size = self.calculate_order_size(instrument, user_info, price)
                
                if size > 0:
                    response = await api.sendOrder(token_ub, instrument, t, side, price, size)
                    
                    if response["status"] != "Success":
                        logger.error(f"Order failed for {instrument}: {response}")
                    else:
                        logger.info(f"Order placed for {instrument}: {side} {size} @ {price}")
        except Exception as e:
            logger.error(f"Error executing trades: {str(e)}")
            
    async def execute_eod_strategy(self, api, token_ub, user_info, t):
        """
        执行EOD（End of Day）交易策略
        :param api: API接口对象
        :param token_ub: 用户token
        :param user_info: 用户信息
        :param t: 当前时间戳
        """
        try:
            logger.info("Executing EOD strategy")
            
            # 获取所有股票的最新行情
            all_lobs = await api.sendGetAllLimitOrderBooks(token_ub)
            if all_lobs["status"] != "Success":
                raise Exception("Failed to get LOB data for EOD strategy")

            for instrument in self.instruments:
                instrument_id = int(instrument[-3:])
                current_position = user_info["rows"][instrument_id]["remain_volume"]
                target_position = user_info["rows"][instrument_id]["target_volume"]
                
                if current_position == target_position:
                    continue  # 已达到目标仓位，无需操作
                
                lob = all_lobs["lobs"][self.instruments.index(instrument)]
                
                # 计算需要交易的数量
                volume_to_trade = abs(target_position - current_position)
                
                if volume_to_trade > 0:
                    # 决定交易方向
                    side = "buy" if target_position > current_position else "sell"
                    
                    # 计算更激进的价格，以提高成交概率
                    if side == "buy":
                        price = float(lob["askprice"][0]) * 1.001  # 略高于卖一价
                    else:
                        price = float(lob["bidprice"][0]) * 0.999  # 略低于买一价
                    
                    # 分批下单，提高成交概率
                    batch_size = min(volume_to_trade, 1000)  # 每批最多1000股
                    while volume_to_trade > 0:
                        order_volume = min(batch_size, volume_to_trade)
                        response = await api.sendOrder(token_ub, instrument, t, side, price, order_volume)
                        
                        if response["status"] == "Success":
                            logger.info(f"EOD order placed for {instrument}: {side} {order_volume} @ {price}")
                            volume_to_trade -= order_volume
                        else:
                            logger.error(f"EOD order failed for {instrument}: {response}")
                        
                        # 短暂等待，避免频繁下单
                        await asyncio.sleep(0.1)
                    
                    # 检查订单是否全部成交，如果没有，考虑调整价格重新下单
                    user_info = await api.sendGetUserInfo(token_ub)
                    updated_position = user_info["rows"][instrument_id]["remain_volume"]
                    if updated_position != target_position:
                        logger.warning(f"EOD target not met for {instrument}. Adjusting strategy.")
                        # 这里可以添加更激进的策略，如继续提高买入价或降低卖出价
            
            logger.info("EOD strategy execution completed")
        except Exception as e:
            logger.error(f"Error executing EOD strategy: {str(e)}")

    async def work(self, api, token_ub, t):
        """
        策略的主要工作流程
        :param api: API接口对象
        :param token_ub: 用户token
        :param t: 当前时间戳
        """
        try:
            user_info = await api.sendGetUserInfo(token_ub)
            if t >= 2950 and t < 3000:  # 在交易日最后50秒执行EOD策略
                await self.execute_eod_strategy(api,token_ub, user_info, t)
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