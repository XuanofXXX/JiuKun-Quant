
class AdvancedHFTStrategy:
    def __init__(self, lookback=100, alpha=0.01, gamma=0.1, vwap_window=20, time_window=3000):
        self.lookback = lookback
        self.alpha = alpha
        self.gamma = gamma
        self.vwap_window = vwap_window
        self.time_window = time_window
        self.price_history = deque(maxlen=lookback)
        self.volume_history = deque(maxlen=lookback)
        self.order_flow_imbalance = deque(maxlen=lookback)
        self.vwap_history = deque(maxlen=vwap_window)

    def update(self, lob):
        mid_price = (lob['AskPrice1'] + lob['BidPrice1']) / 2
        volume = lob['TotalTradeVolume']
        order_imbalance = self.calculate_order_book_imbalance(lob)
        
        self.price_history.append(mid_price)
        self.volume_history.append(volume)
        self.order_flow_imbalance.append(order_imbalance)
        
        if len(self.price_history) >= 2:
            prices = list(self.price_history)
            volumes = list(self.volume_history)
            vwap = np.average(prices[-self.vwap_window:], 
                              weights=volumes[-self.vwap_window:])
            self.vwap_history.append(vwap)

    def calculate_order_book_imbalance(self, lob):
        bid_volume = sum(lob[f'BidVolume{i}'] for i in range(1, 6))
        ask_volume = sum(lob[f'AskVolume{i}'] for i in range(1, 6))
        return (bid_volume - ask_volume) / (bid_volume + ask_volume)

    def estimate_price_impact(self, order_size, side, lob):
        if side == 'buy':
            cumulative_volume = 0
            for i in range(1, 6):
                cumulative_volume += lob[f'AskVolume{i}']
                logger.debug(f"cumulative_volume: {cumulative_volume}, order_size: {order_size}")
                if cumulative_volume >= order_size:
                    return (lob[f'AskPrice{i}'] - lob['AskPrice1']) / lob['AskPrice1']
        else:
            cumulative_volume = 0
            for i in range(1, 6):
                cumulative_volume += lob[f'BidVolume{i}']
                if cumulative_volume >= order_size:
                    return (lob['BidPrice1'] - lob[f'BidPrice{i}']) / lob['BidPrice1']
        return 0.01  # 如果订单大小超过可见深度，返回一个默认值

    def calculate_optimal_order_size(self, remain_position, lob, prediction, t):
        market_impact = self.estimate_price_impact(abs(remain_position), 'buy' if remain_position > 0 else 'sell', lob)
        volatility = np.std(list(self.price_history)) if len(self.price_history) > 1 else 0.001
        
        optimal_size = abs(remain_position) * np.sqrt(self.alpha / (volatility * market_impact))
        optimal_size = min(optimal_size, abs(remain_position))
        
        # 加入时间因子
        time_factor = self.calculate_time_factor(t)
        
        adjusted_size = optimal_size * (1 + self.gamma * abs(prediction)) * time_factor
        
        return int(round(adjusted_size / 100) * 100)

    def calculate_price_trend(self):
        if len(self.price_history) < self.lookback:
            return 0
        prices = list(self.price_history)
        x = np.arange(len(prices))
        y = np.array(prices)
        slope, _, _, _, _ = linregress(x, y)
        return slope

    def calculate_order_price(self, lob, side, prediction, t):
        mid_price = (lob['AskPrice1'] + lob['BidPrice1']) / 2
        spread = lob['AskPrice1'] - lob['BidPrice1']
        
        vwap = np.mean(list(self.vwap_history)) if self.vwap_history else mid_price
        price_trend = self.calculate_price_trend()
        order_book_imbalance = self.calculate_order_book_imbalance(lob)
        
        # 加入时间因子
        time_factor = self.calculate_time_factor(t)
        
        price_adjustment = (
            0.2 * spread * prediction +
            0.3 * spread * price_trend / max(abs(price_trend), 1e-6) +
            0.2 * spread * order_book_imbalance +
            0.3 * (vwap - mid_price)
        ) * time_factor
        
        if side == 'buy':
            price = mid_price - 0.5 * spread + price_adjustment
        else:
            price = mid_price + 0.5 * spread + price_adjustment
        
        return max(lob['BidPrice1'], min(lob['AskPrice1'], price))

    def calculate_time_factor(self, t):
        # 基础时间因子
        base_time_factor = t / self.time_window
        
        # 添加一些非线性特性
        if base_time_factor < 0.2:
            return 0.5 + base_time_factor  # 开盘时段较为保守
        elif base_time_factor > 0.8:
            return 1.5 + (base_time_factor - 0.8) * 2.5  # 收盘前更积极
        else:
            return 1 + (base_time_factor - 0.5) ** 2  # 中间时段平稳增加

    def get_order_params(self, lob, remain_position, prediction, t):
        # logger.debug(f"lob.shape: {lob.shape}")
        # lob.to_csv('temp.csv', index=False)
        self.update(lob)
        
        side = 'buy' if remain_position > 0 else 'sell'
        size = self.calculate_optimal_order_size(remain_position, lob, prediction, t)
        price = self.calculate_order_price(lob, side, prediction, t)
        
        return side, size, price


class CancelOrderStrategy:
    def __init__(self, max_order_age: int = 5, price_threshold: float = 0.005, volume_threshold: float = 0.5):
        self.max_order_age = max_order_age  # 最大订单年龄（秒）
        self.price_threshold = price_threshold  # 价格偏离阈值
        self.volume_threshold = volume_threshold  # 成交量阈值

    async def check_and_cancel_orders(self, api, token_ub, active_orders: List[Dict], current_lob: Dict, current_time: int):
        cancel_tasks = []

        for order in active_orders:
            if self.should_cancel_order(order, current_lob, current_time):
                cancel_tasks.append(self.cancel_order(api, token_ub, order, current_time))

        if cancel_tasks:
            await asyncio.gather(*cancel_tasks)

    def should_cancel_order(self, order: Dict, current_lob: Dict, current_time: int) -> bool:
        # 检查订单年龄
        if current_time - order['localtime'] > self.max_order_age:
            logger.info(f"Order {order['index']} exceeded max age, cancelling.")
            return True

        # 检查价格偏离
        if order['direction'] == 'buy':
            current_price = current_lob['askprice'][0]
        else:
            current_price = current_lob['bidprice'][0]

        price_deviation = abs(order['price'] - current_price) / current_price
        if price_deviation > self.price_threshold:
            logger.info(f"Order {order['index']} price deviation {price_deviation:.2%} exceeded threshold, cancelling.")
            return True

        # 检查成交量
        if order['volume'] > current_lob['askvolume'][0] * self.volume_threshold:
            logger.info(f"Order {order['index']} volume {order['volume']} exceeded {self.volume_threshold:.0%} of market volume, cancelling.")
            return True

        return False

    async def cancel_order(self, api, token_ub: str, order: Dict, current_time: int):
        try:
            response = await api.sendCancel(token_ub, order['instrument'], current_time, order['index'])
            if response['status'] == 'Success':
                logger.info(f"Successfully cancelled order {order['index']} for {order['instrument']}")
            else:
                logger.error(f"Failed to cancel order {order['index']}: {response}")
        except Exception as e:
            logger.error(f"Error cancelling order {order['index']}: {str(e)}")

    async def update_and_cancel(self, api, token_ub: str, current_time: int):
        try:
            # 获取活跃订单
            active_orders_response = await api.sendGetActiveOrder(token_ub)
            if active_orders_response['status'] != 'Success':
                logger.error(f"Failed to get active orders: {active_orders_response}")
                return

            active_orders = active_orders_response['rows']

            # 获取最新的限价订单簿
            lob_response = await api.sendGetAllLimitOrderBooks(token_ub)
            if lob_response['status'] != 'Success':
                logger.error(f"Failed to get limit order books: {lob_response}")
                return

            lobs = lob_response['lobs']

            # 检查并取消订单
            for order in active_orders:
                instrument_id = int(order['instrument'][-3:])
                current_lob = lobs[instrument_id]
                await self.check_and_cancel_orders(api, token_ub, [order], current_lob, current_time)

        except Exception as e:
            logger.error(f"Error in update_and_cancel: {str(e)}")


class NaiveOrderStrategy:
    def __init__(self, time_window=3000, start_size = 100,):
        self.time_window = time_window
        self.start_size = start_size

    def calculate_order_size(self, tradable_size, t, prediction):
        # 基础订单大小就是remain_position的绝对值
        base_size = abs(tradable_size)
        
        # 时间因子：随着时间推移逐渐增加订单大小
        # TODO softmax
        
        time_factor = np.sqrt(t / self.time_window)

        # 计算最终订单大小
        size = base_size * time_factor
        
        # 四舍五入到最接近的100的倍数
        size = int(round(size / 100) * 100)
        
        # 确保订单大小不超过需要交易的数量
        size = min(size, base_size)
        
        return max(size, self.start_size)

    def calculate_order_price(self, mid_price, spread, side, prediction, t):
        # 基础价格：买入时略低于中间价，卖出时略高于中间价
        base_price = mid_price
        
        if t < 1500:
            price_adjustment = spread * 0.3
        else:
            price_adjustment = spread * 0.7
            
        
        logger.debug(f"mid price: {mid_price}, prediction: {prediction}")
        
        # TODO 如果是买的话，就低价速速买进
        if side == "buy":
            price = base_price + price_adjustment
        else:  # sell
            price = base_price - price_adjustment
        
        # 确保价格在合理范围内
        return price

    def get_order_params(self, lob, remain_position, t, side, prediction):
        mid_price = (lob["AskPrice1"] * lob['AskVolume1'] + lob["BidPrice1"] * lob['BidVolume1']) / (lob['AskVolume1'] + lob['BidVolume1'])
        spread = lob["AskPrice1"] - lob["BidPrice1"]
        
        size = self.calculate_order_size(remain_position, t, prediction)
        price = self.calculate_order_price(mid_price, spread, side, prediction, t)
        
        return side, size , round(price,2)
