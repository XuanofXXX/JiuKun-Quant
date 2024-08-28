import time 
import pandas as pd

def ConvertToSimTime_us(start_time, time_ratio, day, running_time):
    return (time.time() - start_time - (day - 1) * running_time) * time_ratio

def convert_LOB_response_to_df_format(api_response):
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

def convert_userinfo_response_to_df_format(api_response, t):
    # 创建一个字典来存储转换后的数据
    base_data = {
        'Tick': [], 'StockID': [],
        'share_holding': [],
        'orders': [],
        'error_orders': [], 'order_value': [],
        'trade_value': [],
        'target_volume': [],'remain_volume': [],
        'frozen_volume': []
    }

    for idx, item in enumerate(api_response):
        # 填充总成交量和总成交额
        base_data['Tick'].append(t)
        base_data['StockID'].append(f'UBIQ{idx:03}')
        
        base_data['share_holding'].append(item['share_holding'])
        base_data['orders'].append(item['orders'])
        base_data['error_orders'].append(item['error_orders'])
        base_data['order_value'].append(item['order_value'])
        base_data['trade_value'].append(item['trade_value'])
        base_data['target_volume'].append(item['target_volume'])
        base_data['remain_volume'].append(item['remain_volume'])
        base_data['frozen_volume'].append(item['frozen_volume'])

    # 创建DataFrame
    df = pd.DataFrame(base_data)
    return df