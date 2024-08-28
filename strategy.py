import numpy as np
from scipy import stats
from logger_config import strategy_logger as logger


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
            logger.error(f"Error in work method: {str(e)}")\

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
        logger.info(f"Initialized {len(instruments)} instruments")

    async def update_market_data(self, api, token_ub):
        all_lobs = await api.sendGetAllLimitOrderBooks(token_ub)
        if all_lobs["status"] != "Success":
            logger.error("Failed to get LOB data")
            raise Exception("Failed to get LOB data")

        for i, instrument in enumerate(self.instruments):
            lob = all_lobs["lobs"][i]
            mid_price = (float(lob["askprice"][0]) + float(lob["bidprice"][-1])) / 2
            volume = float(lob["askvolume"][0]) + float(lob["bidvolume"][-1])
            twap = float(lob.get("twap", mid_price))

            self.price_history[instrument].append(mid_price)
            self.volume_history[instrument].append(volume)
            self.twap_history[instrument].append(twap)

            if len(self.price_history[instrument]) > self.lookback_period:
                self.price_history[instrument].pop(0)
                self.volume_history[instrument].pop(0)
                self.twap_history[instrument].pop(0)

        logger.debug("Updated market data for all instruments")

    # ... (其他方法保持不变，但添加适当的日志记录)

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
                logger.error(f"Order failed for {instrument}: {response}")
            else:
                logger.info(f"Order placed for {instrument}: {side} {size} @ {price}")

    async def cancel_stale_orders(self, api, token_ub, t, max_order_age=60):
        user_info = await api.sendGetUserInfo(token_ub)
        for instrument in self.instruments:
            instrument_id = int(instrument[-3:])
            orders = user_info["rows"][instrument_id]["orders"]
            for order in orders:
                if t - order["localtime"] > max_order_age:
                    await api.sendCancel(token_ub, instrument, t, order["index"])
                    logger.info(f"Cancelled stale order for {instrument}: {order['index']}")

    async def work(self, api, token_ub, t, total_time=300):
        try:
            await self.update_market_data(api, token_ub)
            user_info = await api.sendGetUserInfo(token_ub)
            remaining_time = max(0, total_time - t)
            await self.execute_trades(api, token_ub, user_info, t, remaining_time)
            await self.cancel_stale_orders(api, token_ub, t)
        except Exception as e:
            logger.exception(f"Error in work method: {str(e)}")