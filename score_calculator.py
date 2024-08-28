import asyncio
import pandas as pd
import numpy as np
from datetime import datetime
from utils.logger_config import setup_logger
from utils.Interface import AsyncInterfaceClass
from utils.convert import convert_LOB_response_to_df_format, convert_userinfo_response_to_df_format, ConvertToSimTime_us

logger = setup_logger('score_calculator_bot')

class ScoreCalculatorBot:
    def __init__(self, username, password, host='http://8.147.116.35:30020', csv_filename='trading_scores.csv'):
        self.username = username
        self.password = password
        self.api = AsyncInterfaceClass(host, logger)
        self.csv_filename = csv_filename
        self.lobs_df = None
        self.user_info_df = None
        self.token_ub = None
        self.start_time = None
        self.running_days = None
        self.running_time = None
        self.time_ratio = None

    async def login(self):
        response = await self.api.sendLogin(self.username, self.password)
        if response["status"] == "Success":
            self.token_ub = response["token_ub"]
            logger.info(f"Login Success: {self.token_ub}")
        else:
            logger.error(f"Login Error: {response}")
            raise Exception("Login failed")

    async def init(self):
        await self.login()
        response = await self.api.sendGetGameInfo(self.token_ub)
        if response["status"] == "Success":
            self.start_time = response["next_game_start_time"]
            self.running_days = response["next_game_running_days"]
            self.running_time = response["next_game_running_time"]
            self.time_ratio = response["next_game_time_ratio"]
        logger.info(response)
        self.day = 0

    async def get_market_data(self, t):
        try:
            lob_response, user_info_response = await asyncio.gather(
                self.api.sendGetAllLimitOrderBooks(self.token_ub),
                self.api.sendGetUserInfo(self.token_ub)
            )

            new_lob_df = convert_LOB_response_to_df_format(lob_response['lobs'])
            new_user_info_df = convert_userinfo_response_to_df_format(user_info_response['rows'], t)

            if lob_response['status'] != 'Success' or user_info_response['status'] != 'Success':
                raise Exception("Failed to get market data")

            # Update historical data
            self.lobs_df = pd.concat([self.lobs_df, new_lob_df], ignore_index=True)
            self.user_info_df = pd.concat([self.user_info_df, new_user_info_df], ignore_index=True)

            return new_lob_df, new_user_info_df
        except Exception as e:
            logger.error(f"Error getting market data: {str(e)}")
            raise


    async def calculate_daily_score(self, t):
        try:
            lob_df, user_info_df = await self.get_market_data(t)

            scores = []
            data = []
            for _, row in user_info_df.iterrows():
                instrument = row['StockID']
                lob = lob_df[lob_df['StockID'] == instrument].iloc[0]

                vwap = row['trade_value'] / abs(row['share_holding']) if row['share_holding'] != 0 else 0
                twap = lob['twap']
                remaining_volume = row['remain_volume']

                trade_direction = 1 if row['target_volume'] > 0 else -1
                if vwap != 0 and twap != 0:
                    score = (1 - vwap / twap) * trade_direction + 0.0004
                else:
                    score = 0.0004

                if remaining_volume != 0:
                    current_price = lob['AskPrice1'] if remaining_volume > 0 else lob['BidPrice1']
                    remaining_value = abs(remaining_volume) * current_price
                    total_value = row['trade_value'] + remaining_value
                    total_volume = abs(row['share_holding']) + abs(remaining_volume)
                    adjusted_vwap = total_value / total_volume
                    score = (1 - adjusted_vwap / twap) * trade_direction + 0.0004

                scores.append(score)

                data.append({
                    'Date': datetime.now().strftime('%Y-%m-%d'),
                    'Tick': t,
                    'Instrument': instrument,
                    'VWAP': vwap,
                    'TWAP': twap,
                    'Target_Volume': row['target_volume'],
                    'Share_Holding': row['share_holding'],
                    'Remaining_Volume': remaining_volume,
                    'Trade_Value': row['trade_value'],
                    'Score': score
                })

            mean_score = np.mean(scores)
            std_score = np.std(scores)
            ir = mean_score / std_score if std_score != 0 else 0
            daily_score = abs(mean_score) * ir

            data.append({
                'Date': datetime.now().strftime('%Y-%m-%d'),
                'Tick': t,
                'Instrument': 'TOTAL',
                'VWAP': None,
                'TWAP': None,
                'Target_Volume': None,
                'Share_Holding': None,
                'Remaining_Volume': None,
                'Trade_Value': None,
                'Score': daily_score
            })

            return daily_score, data
        except Exception as e:
            logger.error(f"Error calculating daily score: {str(e)}")
            raise

    async def save_to_csv(self, data):
        try:
            df = pd.DataFrame(data)
            df.to_csv(self.csv_filename, mode='a', header=not pd.io.common.file_exists(self.csv_filename), index=False)
            logger.info(f"Data saved to {self.csv_filename}")
        except Exception as e:
            logger.error(f"Error saving to CSV: {str(e)}")
            raise

    @staticmethod
    def calculate_final_score(daily_scores):
        mean_score = np.mean(daily_scores)
        std_score = np.std(daily_scores)
        ir = mean_score / std_score if std_score != 0 else 0
        final_score = abs(mean_score) * ir
        return final_score

    def calculate_final_score_from_csv(self):
        try:
            df = pd.read_csv(self.csv_filename)
            daily_scores = df[df['Instrument'] == 'TOTAL']['Score'].tolist()
            final_score = self.calculate_final_score(daily_scores)
            logger.info(f"Final score calculated: {final_score}")
            return final_score
        except Exception as e:
            logger.error(f"Error calculating final score from CSV: {str(e)}")
            raise

    async def work(self, t):
        daily_score, data = await self.calculate_daily_score(t)
        await self.save_to_csv(data)
        logger.info(f"Daily score at tick {t}: {daily_score}")

async def main(username, password):
    bot = ScoreCalculatorBot(username, password)
    await bot.init()

    SimTimeLen = 3000
    endWaitTime = 600

    while ConvertToSimTime_us(bot.start_time, bot.time_ratio, bot.day, bot.running_time) >= SimTimeLen:
        bot.day += 1

    while bot.day <= bot.running_days:
        while ConvertToSimTime_us(bot.start_time, bot.time_ratio, bot.day, bot.running_time) <= -1:
            await asyncio.sleep(0.1)

        now = round(ConvertToSimTime_us(bot.start_time, bot.time_ratio, bot.day, bot.running_time))
        for s in range(now, SimTimeLen + endWaitTime):
            while True:
                if ConvertToSimTime_us(bot.start_time, bot.time_ratio, bot.day, bot.running_time) >= s:
                    break
            t = ConvertToSimTime_us(bot.start_time, bot.time_ratio, bot.day, bot.running_time)
            logger.info(f"Work Time: {s} {t}")
            if t <= SimTimeLen:
                await bot.work(t)
        bot.day += 1

    final_score = bot.calculate_final_score_from_csv()
    logger.info(f"Final score: {final_score}")

    await bot.api.close_session()

if __name__ == "__main__":
    username = 'UBIQ_TEAM179'
    password = 'ANfgOwr3SvpN'
    asyncio.run(main(username, password))