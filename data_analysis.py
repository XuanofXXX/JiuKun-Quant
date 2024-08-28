import pandas as pd
import numpy as np

def analyze_liquidity(data, target_positions):
    """
    分析股票流动性并评估目标仓位相对于历史成交量的规模。

    :param data: DataFrame，包含每只股票的历史成交量数据
    :param target_positions: dict，键为股票代码，值为目标仓位
    :return: DataFrame，包含每只股票的流动性分析结果
    """
    results = []
    
    for stock in data.columns:
        stock_data = data[stock].dropna()
        
        # 计算日均成交量和标准差
        mean_volume = stock_data.mean()
        std_volume = stock_data.std()
        
        # 获取目标仓位
        target_position = abs(target_positions.get(stock, 0))
        
        # 计算比率
        ratio_to_mean = target_position / mean_volume if mean_volume != 0 else np.inf
        ratio_to_std = target_position / std_volume if std_volume != 0 else np.inf
        
        # 将比率映射到0-10的范围
        scaled_ratio_mean = min(ratio_to_mean * 10, 10)
        scaled_ratio_std = min(ratio_to_std * 10, 10)
        
        results.append({
            'stock': stock,
            'mean_daily_volume': mean_volume,
            'volume_std': std_volume,
            'target_position': target_position,
            'ratio_to_mean_volume': ratio_to_mean,
            'ratio_to_volume_std': ratio_to_std,
            'scaled_ratio_mean': scaled_ratio_mean,
            'scaled_ratio_std': scaled_ratio_std
        })
    
    results_df = pd.DataFrame(results)
    
    # 添加描述性统计
    stats = results_df[['ratio_to_mean_volume', 'ratio_to_volume_std', 'scaled_ratio_mean', 'scaled_ratio_std']].describe()
    
    return results_df, stats

# 使用示例
# data = pd.DataFrame({...})  # 您的历史成交量数据
# target_positions = {...}  # 您的目标仓位字典
# results, stats = analyze_liquidity(data, target_positions)
# print(results)
# print(stats)