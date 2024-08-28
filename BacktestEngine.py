import pandas as pd
import numpy as np
from typing import List, Dict, Callable

class Order:
    def __init__(self, ticker: str, side: str, price: float, volume: int):
        self.ticker = ticker
        self.side = side  # 'buy' or 'sell'
        self.price = price
        self.volume = volume
        self.filled_volume = 0
        self.status = 'open'  # 'open', 'filled', or 'canceled'

class Position:
    def __init__(self, ticker: str):
        self.ticker = ticker
        self.volume = 0
        self.cost_basis = 0

class BacktestEngine:
    def __init__(self, data: pd.DataFrame, initial_capital: float):
        self.data = data
        self.capital = initial_capital
        self.positions: Dict[str, Position] = {}
        self.orders: List[Order] = []
        self.trades: List[Dict] = []
        self.current_tick = 0
        
    def place_order(self, order: Order):
        self.orders.append(order)
    
    def cancel_order(self, order: Order):
        if order in self.orders and order.status == 'open':
            order.status = 'canceled'
            self.orders.remove(order)
    
    def match_orders(self):
        current_lob = self.data.iloc[self.current_tick]
        for order in self.orders:
            if order.status == 'open':
                if order.side == 'buy':
                    if order.price >= current_lob[f'AskPrice1_{order.ticker}']:
                        filled_price = current_lob[f'AskPrice1_{order.ticker}']
                        filled_volume = min(order.volume, current_lob[f'AskVolume1_{order.ticker}'])
                    else:
                        continue
                else:  # sell order
                    if order.price <= current_lob[f'BidPrice1_{order.ticker}']:
                        filled_price = current_lob[f'BidPrice1_{order.ticker}']
                        filled_volume = min(order.volume, current_lob[f'BidVolume1_{order.ticker}'])
                    else:
                        continue
                
                order.filled_volume += filled_volume
                if order.filled_volume == order.volume:
                    order.status = 'filled'
                
                self.execute_trade(order.ticker, order.side, filled_price, filled_volume)
    
    def execute_trade(self, ticker: str, side: str, price: float, volume: int):
        if ticker not in self.positions:
            self.positions[ticker] = Position(ticker)
        
        position = self.positions[ticker]
        
        if side == 'buy':
            cost = price * volume
            if self.capital >= cost:
                self.capital -= cost
                position.volume += volume
                position.cost_basis = (position.cost_basis * position.volume + cost) / (position.volume + volume)
            else:
                # Not enough capital, handle this case (e.g., partial fill or rejection)
                return
        else:  # sell
            if position.volume >= volume:
                self.capital += price * volume
                position.volume -= volume
            else:
                # Not enough stocks to sell, handle this case
                return
        
        self.trades.append({
            'ticker': ticker,
            'side': side,
            'price': price,
            'volume': volume,
            'timestamp': self.data.index[self.current_tick]
        })
    
    def run(self, strategy: Callable):
        for tick in range(len(self.data)):
            self.current_tick = tick
            strategy(self, self.data.iloc[tick])
            self.match_orders()
    
    def get_portfolio_value(self):
        portfolio_value = self.capital
        current_prices = self.data.iloc[self.current_tick]
        for ticker, position in self.positions.items():
            portfolio_value += position.volume * current_prices[f'BidPrice1_{ticker}']
        return portfolio_value

# 示例策略
def simple_strategy(engine: BacktestEngine, current_data: pd.Series):
    for ticker in engine.positions.keys():
        bid_price = current_data[f'BidPrice1_{ticker}']
        ask_price = current_data[f'AskPrice1_{ticker}']
        mid_price = (bid_price + ask_price) / 2
        
        # 简单的均值回归策略
        if mid_price > engine.positions[ticker].cost_basis * 1.01:  # 1% 利润
            engine.place_order(Order(ticker, 'sell', bid_price, engine.positions[ticker].volume))
        elif mid_price < engine.positions[ticker].cost_basis * 0.99:  # 1% 亏损
            engine.place_order(Order(ticker, 'buy', ask_price, 100))  # 买入100股

# 使用示例
if __name__ == "__main__":
    # 假设我们有一个包含LOB数据的DataFrame
    data = pd.read_csv('lob_data.csv', index_col='timestamp', parse_dates=True)
    
    engine = BacktestEngine(data, initial_capital=1000000)  # 100万初始资金
    engine.run(simple_strategy)
    
    final_portfolio_value = engine.get_portfolio_value()
    print(f"Final portfolio value: {final_portfolio_value}")
    print(f"Total return: {(final_portfolio_value - 1000000) / 1000000 * 100:.2f}%")
    print(f"Number of trades: {len(engine.trades)}")