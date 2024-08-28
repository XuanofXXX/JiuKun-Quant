import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy import stats
import math

def load_and_process_data(file_path):
    df = pd.read_csv(file_path, sep='|')
    df['TotalVolume'] = df['AskVolume1'] + df['AskVolume2'] + df['AskVolume3'] + df['AskVolume4'] + df['AskVolume5'] + \
                        df['BidVolume1'] + df['BidVolume2'] + df['BidVolume3'] + df['BidVolume4'] + df['BidVolume5']
    return df

def calculate_liquidity_metrics(data):
    daily_volume = data['TotalTradeVolume'].sum()
    volume_volatility = data['TotalTradeVolume'].std()
    volume_distribution = data['TotalTradeVolume'].tolist()  # 改为列表，而不是 stats.describe
    
    return {
        'daily_volume': daily_volume,
        'volume_volatility': volume_volatility,
        'volume_distribution': volume_distribution
    }

def analyze_stocks(folder_path):
    results = {}
    for file in os.listdir(folder_path):
        if file.endswith('.csv'):
            stock_id = file.split('_')[1].split('.')[0]
            data = load_and_process_data(os.path.join(folder_path, file))
            metrics = calculate_liquidity_metrics(data)
            results[stock_id] = metrics
    
    return results

def calculate_target_ratios(results, target_volume):
    ratios = {}
    for stock_id, metrics in results.items():
        mean_volume = metrics['daily_volume']
        std_volume = metrics['volume_volatility']
        
        ratio_mean = abs(target_volume) / mean_volume if mean_volume != 0 else 0
        ratio_std = abs(target_volume) / std_volume if std_volume != 0 else 0
        
        ratios[stock_id] = {
            'ratio_to_mean': min(ratio_mean, 10),
            'ratio_to_std': min(ratio_std, 10)
        }
    
    return ratios

def generate_report(results, ratios):
    report = "股票流动性分析报告\n\n"
    
    for stock_id in results.keys():
        report += f"股票 {stock_id}:\n"
        report += f"  日均成交量: {results[stock_id]['daily_volume']:.2f}\n"
        report += f"  成交量波动性: {results[stock_id]['volume_volatility']:.2f}\n"
        report += f"  成交量分布:\n"
        report += f"    最小值: {min(results[stock_id]['volume_distribution']):.2f}\n"
        report += f"    最大值: {max(results[stock_id]['volume_distribution']):.2f}\n"
        report += f"    均值: {np.mean(results[stock_id]['volume_distribution']):.2f}\n"
        report += f"    方差: {np.var(results[stock_id]['volume_distribution']):.2f}\n"
        report += f"    偏度: {stats.skew(results[stock_id]['volume_distribution']):.2f}\n"
        report += f"    峰度: {stats.kurtosis(results[stock_id]['volume_distribution']):.2f}\n"
        report += f"  目标仓位/日均成交量比率: {ratios[stock_id]['ratio_to_mean']:.2f}\n"
        report += f"  目标仓位/成交量标准差比率: {ratios[stock_id]['ratio_to_std']:.2f}\n\n"
    
    return report

def plot_volume_distribution(results):
    num_stocks = len(results)
    num_cols = 3
    num_rows = math.ceil(num_stocks / num_cols)
    
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(20, 5 * num_rows))
    fig.suptitle('股票成交量分布')
    
    results = sorted(results, key=lambda x: x[0])
    
    for i, (stock_id, metrics) in enumerate(results.items()):
        row = i // num_cols
        col = i % num_cols
        
        volume_data = metrics['volume_distribution']
        if num_rows > 1:
            axs[row, col].hist(volume_data, bins=30, edgecolor='black')
            axs[row, col].set_title(f'股票 {stock_id}')
            axs[row, col].set_xlabel('成交量')
            axs[row, col].set_ylabel('频率')
        else:
            axs[col].hist(volume_data, bins=30, edgecolor='black')
            axs[col].set_title(f'股票 {stock_id}')
            axs[col].set_xlabel('成交量')
            axs[col].set_ylabel('频率')
    
    # 移除多余的子图
    if num_stocks < num_rows * num_cols:
        for i in range(num_stocks, num_rows * num_cols):
            row = i // num_cols
            col = i % num_cols
            if num_rows > 1:
                fig.delaxes(axs[row, col])
            else:
                fig.delaxes(axs[col])
    
    plt.tight_layout()
    plt.savefig('volume_distribution.png')

def main():
    folder_path = 'snapshots'
    target_volume = 10000  # 假设的目标仓位
    
    results = analyze_stocks(folder_path)
    ratios = calculate_target_ratios(results, target_volume)
    
    report = generate_report(results, ratios)
    print(report)
    
    with open('liquidity_report.txt', 'w') as f:
        f.write(report)
    
    plot_volume_distribution(results)

if __name__ == "__main__":
    main()