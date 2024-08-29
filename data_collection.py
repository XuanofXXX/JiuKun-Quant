import asyncio
import logging.handlers
import aiohttp
import csv
import time
import pandas as pd
from datetime import datetime
from utils.logger_config import setup_logger
from functools import wraps
import random

logger = setup_logger('data_collector')

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
            async with self.session.post(url, json=data) as response:
                return await response.json()
        except aiohttp.ClientError as e:
            logger.error(f"Error sending request to {endpoint}: {str(e)}")
            return {"status": "Error", "message": str(e)}

    async def sendLogin(self, username, password):
        return await self.send_request("/Login", {"user": username, "password": password})

    async def sendGetAllLimitOrderBooks(self, token_ub):
        return await self.send_request("/TradeAPI/GetAllLimitOrderBooks", {"token_ub": token_ub})

    async def sendGetGameInfo(self, token_ub):
        return await self.send_request("/TradeAPI/GetGameInfo", {"token_ub": token_ub})

    async def sendGetInstrumentInfo(self, token_ub):
        return await self.send_request("/TradeAPI/GetInstrumentInfo", {"token_ub": token_ub})
    
    async def sendGetInstrumentInfo(self, token_ub):
        return await self.send_request("/TradeAPI/GetInstrumentInfo", {"token_ub": token_ub})
    
    async def sendGetUserInfo(self, token_ub):
        return await self.send_request("/TradeAPI/GetUserInfo", {"token_ub": token_ub})

class DataCollector:
    def __init__(self, username, password, port):
        self.username = username
        self.password = password
        self.api = AsyncInterfaceClass(f"http://8.147.116.35:{port}")
        self.token_ub = None
        self.instruments = []
        self.start_time = None
        self.time_ratio = None
        self.running_time = None
        self.running_days = None
        self.day = 1
        self.data_buffer = []

    async def init(self):
        await self.login()
        await self.get_game_info()
        await self.get_instruments()

    async def login(self):
        response = await self.api.sendLogin(self.username, self.password)
        if response["status"] == "Success":
            self.token_ub = response["token_ub"]
            logger.info(f"Login Success: {self.token_ub}")
        else:
            logger.error(f"Login Error: {response}")
            raise Exception("Login failed")

    async def get_game_info(self):
        response = await self.api.sendGetGameInfo(self.token_ub)
        if response["status"] == "Success":
            self.start_time = response["next_game_start_time"]
            self.time_ratio = response["next_game_time_ratio"]
            self.running_time = response["next_game_running_time"]
            self.running_days = response["next_game_running_days"]
        else:
            logger.error(f"Get Game Info Error: {response}")
            raise Exception("Failed to get game info")

    async def get_instruments(self):
        response = await self.api.sendGetUserInfo(self.token_ub)
        if response["status"] == "Success":
            rows = response['rows']
            self.instruments = [instrument["instrument_name"] for instrument in rows]
            self.target_positions = {instrument["instrument_name"]: instrument['target_volume'] for instrument in rows}
            logger.info(f"Get Instruments: {self.instruments}")
        else:
            logger.error(f"Get Instruments Error: {response}")
            raise Exception("Failed to get instruments")

    def ConvertToSimTime_us(self, day):
        return (time.time() - self.start_time - (day - 1) * self.running_time) * self.time_ratio

    def initialize_file(self):
        timestamp = datetime.fromtimestamp(self.start_time).strftime("%m%d%H")
        filename = f"./snapshots/{timestamp}-day{self.day}_all_stocks.csv"
        headers = ['Tick', 'StockID'] + [f"{side}{typ}{num}" for side in ["Ask", "Bid"] for typ in ["Price", "Volume"] for num in range(1, 11)] + ["TotalTradeVolume", "TotalTradeValue"] + ['share_holding', 'orders', 'error_orders', 'order_value','trade_value','target_volume','remain_volume', 'frozen_volume']
        
        logger.debug(f"Init: header:{headers}")
        
        self.file = open(filename, 'w', newline='')
        self.writer = csv.writer(self.file)
        self.writer.writerow(headers)

    def close_file(self):
        if hasattr(self, 'file'):
            self.file.close()

    async def collect_data(self, t):
        try:
            t = round(t)
            tasks_lob = [self.api.sendGetAllLimitOrderBooks(self.token_ub) for _ in range(3)]
            tasks_user_info = [self.api.sendGetUserInfo(self.token_ub) for _ in range(3)]

            async def wait_for_first_success(tasks):
                while tasks:
                    done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                    for task in done:
                        try:
                            result = task.result()
                            if result["status"] == "Success":
                                for p in pending:
                                    p.cancel()
                                return result
                        except Exception:
                            pass
                    tasks = pending
                return None

            lob_result, user_info_result = await asyncio.gather(
                wait_for_first_success(tasks_lob),
                wait_for_first_success(tasks_user_info)
            )
            

            if lob_result["status"] == "Success" and user_info_result["status"] == "Success":
                for i, lob in enumerate(lob_result["lobs"]):
                    instr_info = user_info_result['rows'][i]
                    instrument = self.instruments[i]

                    row = [t, instrument]
                    row.extend(lob["askprice"])
                    row.extend(lob["askvolume"])
                    row.extend(lob["bidprice"])
                    row.extend(lob["bidvolume"])
                    row.append(lob["trade_volume"])
                    row.append(lob["trade_value"])
                    row.append(instr_info['share_holding'])
                    row.append(instr_info['orders'])
                    row.append(instr_info['error_orders'])
                    row.append(instr_info['order_value'])
                    row.append(instr_info['trade_value'])
                    row.append(instr_info['target_volume'])
                    row.append(instr_info['remain_volume'])
                    row.append(instr_info['frozen_volume'])
                    
                    self.data_buffer.append(row)

                if len(self.data_buffer) >= 1000:  # 每1000行写入一次文件
                    self.writer.writerows(self.data_buffer)
                    self.data_buffer.clear()
                    logger.info(f"Data save at day:{self.day} time: {t}")

                logger.info(f"Data collected at day:{self.day} time: {t}")
                logger.debug(f"Data collected {row}")
            else:
                logger.error(f"Failed to get data: LOB status: {lob_result['status']}, UserInfo status: {user_info_result['status']}")
        except Exception as e:
            logger.error(f"Error collecting data: {str(e)}")

    async def run(self):
        SimTimeLen = 3000
        endWaitTime = 600
        
        while self.ConvertToSimTime_us(self.day) >= SimTimeLen:
            self.day += 1

        while self.day <= self.running_days:
            while self.ConvertToSimTime_us(self.day) <= -900:
                await asyncio.sleep(0.1)

            self.initialize_file()
            now = round(self.ConvertToSimTime_us(self.day))
            for s in range(now, SimTimeLen + endWaitTime):
                while self.ConvertToSimTime_us(self.day) < s:
                    await asyncio.sleep(0.001)
                t = self.ConvertToSimTime_us(self.day)
                # logger.info(f"Work Time: {s} {t}")
                if t <= SimTimeLen:
                    await self.collect_data(t)
            
            # 写入剩余的数据
            if self.data_buffer:
                self.writer.writerows(self.data_buffer)
                self.data_buffer.clear()
            
            self.close_file()
            self.day += 1

        await self.api.close_session()
        
async def main():
    username = 'UBIQ_TEAM179'
    password = 'ANfgOwr3SvpN'
    collector = DataCollector(username, password, 30020)
    
    await collector.init()
    await collector.run()

if __name__ == "__main__":
    asyncio.run(main())