
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
