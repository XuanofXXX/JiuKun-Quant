import requests
import socket
import json
import time
import logging
import random
import sys


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

def ConvertToSimTime_us(start_time, time_ratio, day, running_time):
    return (time.time() - start_time - (day - 1) * running_time) * time_ratio

class BotsClass:
    def __init__(self, username, password):
        self.username = username
        self.password = password
    def login(self):
        pass
    def init(self):
        pass
    def bod(self):
        pass
    def work(self):
        pass
    def eod(self):
        pass
    def final(self):
        pass

class BotsDemoClass(BotsClass):
    def __init__(self, username, password, port):
        super().__init__(username, password)
        self.api = InterfaceClass(f"http://8.147.116.35:{port}")
    def login(self):
        response = self.api.sendLogin(self.username, self.password)
        if response["status"] == "Success":
            self.token_ub = response["token_ub"]
            logger.info("Login Success: {}".format(self.token_ub))
        else:
            print('\033[31mlogin error \033[0m',response)
            logger.info("Login Error: ", response)
    def GetInstruments(self):
        response = self.api.sendGetInstrumentInfo(self.token_ub)
        print(response)
        if response["status"] == "Success":
            self.instruments = []
            for instrument in response["instruments"]:
                self.instruments.append(instrument["instrument_name"])
            logger.info("Get Instruments: {}".format(self.instruments))
    def init(self):
        response = self.api.sendGetGameInfo(self.token_ub)
        if response["status"] == "Success":
            self.start_time = response["next_game_start_time"]
            self.running_days = response["next_game_running_days"]
            self.running_time = response["next_game_running_time"]
            self.time_ratio = response["next_game_time_ratio"]
        print(response)
        
        self.GetInstruments()
        self.day = 0
    def bod(self):
        pass
    
    def work(self): 
        userinfo = self.api.sendGetUserInfo(self.token_ub)
        stockID = random.randint(0, len(self.instruments) - 1)
        # print(stockID, self.instruments[stockID])
        LOB = self.api.sendGetLimitOrderBook(self.token_ub, self.instruments[stockID])
        if LOB["status"] == "Success":
            askprice_1 = float(LOB["lob"]["askprice"][0])
            bidprice_1 = float(LOB["lob"]["bidprice"][0])
            t = ConvertToSimTime_us(self.start_time, self.time_ratio, self.day, self.running_time)
            #target volume
            if userinfo["rows"][stockID]["target_volume"] > 0:
                response = self.api.sendOrder(self.token_ub, self.instruments[stockID], t, "buy", askprice_1, 100)
            else:
                response = self.api.sendOrder(self.token_ub, self.instruments[stockID], t, "sell", bidprice_1, 100)
            if response["index"] is not None:
                resp = self.api.sendCancel(self.token_ub, self.instruments[stockID], t, response["index"])
                print(resp)
            else:
                print('index none')
    
    def eod(self):
        pass
    
    def final(self):
        pass

class InterfaceClass:
    def __init__(self, domain_name):
        self.domain_name = domain_name
        self.session = requests.Session()

    def sendLogin(self, username, password):
        url = self.domain_name + "/Login"
        data = {
            "user": username,
            "password": password
        }
        response = self.session.post(url, data=json.dumps(data)).json()
        return response

    def sendOrder(self, token_ub, instrument, localtime, direction, price, volume):
        logger.debug("Order: Instrument: {}, Direction:{}, Price: {}, Volume:{}".format(instrument, direction, price, volume))
        url = self.domain_name + "/TradeAPI/Order"
        data = {
            "token_ub": token_ub,
            "user_info": "NULL",
            "instrument": instrument,
            "localtime": localtime,
            "direction": direction,
            "price": price,
            "volume": volume,
        }
        print(data)
        response = self.session.post(url, data=json.dumps(data)).json()
        return response

    def sendCancel(self, token_ub, instrument, localtime, index):
        logger.debug("Cancel: Instrument: {}, index:{}".format(instrument, index))
        url = self.domain_name + "/TradeAPI/Cancel"
        data = {
            "token_ub": token_ub,
            "user_info": "NULL",
            "instrument": instrument,
            "localtime": 0,
            "index": index
        }
        response = self.session.post(url, data=json.dumps(data)).json()
        return response

    def sendGetLimitOrderBook(self, token_ub, instrument):
        logger.debug("GetLimitOrderBOok: Instrument: {}".format(instrument))
        url = self.domain_name + "/TradeAPI/GetLimitOrderBook"
        data = {
            "token_ub": token_ub,
            "instrument": instrument
        }
        response = self.session.post(url, data=json.dumps(data)).json()
        return response

    def sendGetUserInfo(self, token_ub):
        logger.debug("GetUserInfo: ")
        url = self.domain_name + "/TradeAPI/GetUserInfo"
        data = {
            "token_ub": token_ub,
        }
        response = self.session.post(url, data=json.dumps(data)).json()
        return response

    def sendGetGameInfo(self, token_ub):
        logger.debug("GetGameInfo: ")
        url = self.domain_name + "/TradeAPI/GetGameInfo"
        data = {
            "token_ub": token_ub,
        }
        response = self.session.post(url, data=json.dumps(data)).json()
        return response

    def sendGetInstrumentInfo(self, token_ub):
        logger.debug("GetInstrumentInfo: ")
        url = self.domain_name + "/TradeAPI/GetInstrumentInfo"
        data = {
            "token_ub": token_ub,
        }
        response = self.session.post(url, data=json.dumps(data)).json()
        return response

    def sendGetTrade(self, token_ub, instrument):
        logger.debug("GetTrade: Instrment: {}".format(instrument))
        url = self.domain_name + "/TradeAPI/GetTrade"
        data = {
            "token_ub": token_ub,
            "instrument_name": instrument
        }
        response = self.session.post(url, data=json.dumps(data)).json()
        return response

    def sendGetActiveOrder(self, token_ub):
        logger.debug("GetActiveOrder: ")
        url = self.domain_name + "/TradeAPI/GetActiveOrder"
        data = {
            "token_ub": token_ub,
        }
        response = self.session.post(url, data=json.dumps(data)).json()
        return response

# 命令行传入用户名和密码：python main.py username password
username = 'UBIQ_TEAM179'
password = 'ANfgOwr3SvpN'

bot = BotsDemoClass(username, password, 30020)
bot.login()
bot.init()
SimTimeLen = 3000 # 60*5 * 10
endWaitTime = 600 # 60*1 * 10
while True:
    if ConvertToSimTime_us(bot.start_time, bot.time_ratio, bot.day, bot.running_time) < SimTimeLen:
        break
    else:
        bot.day += 1

while bot.day <= bot.running_days:
    while True:
        if ConvertToSimTime_us(bot.start_time, bot.time_ratio, bot.day, bot.running_time) > -900:
            break
    bot.bod()
    now = round(ConvertToSimTime_us(bot.start_time, bot.time_ratio, bot.day, bot.running_time))
    for s in range(now, SimTimeLen + endWaitTime):
        while True:
            if ConvertToSimTime_us(bot.start_time, bot.time_ratio, bot.day, bot.running_time) >= s:
                break
        t = ConvertToSimTime_us(bot.start_time, bot.time_ratio, bot.day, bot.running_time)
        logger.info("Work Time: {} {}".format(s, t))
        if t < SimTimeLen - 30:
            bot.work()
    bot.eod()
    bot.day += 1
bot.final()
