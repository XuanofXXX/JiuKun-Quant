import pandas as pd
import glob
import os
import re
from collections import defaultdict

def extract_day_and_stock(filename):
    # 使用正则表达式提取日期和股票ID
    match = re.search(r'(\d{8}-\d{6})-(\d+)_UBIQ(\d+)', filename)
    if match:
        timestamp, day, stock_id = match.groups()
        return int(day), f'UBIQ{stock_id}'
    return None, None

def merge_csv_files(input_dir, output_dir):
    # 获取所有CSV文件
    all_files = glob.glob(os.path.join(input_dir, '*.csv'))
    
    # 按天数分组文件
    day_files = defaultdict(list)
    for file in all_files:
        day, stock_id = extract_day_and_stock(os.path.basename(file))
        if day is not None:
            day_files[day].append((file, stock_id))
    
    # 处理每一天的文件
    for day, files in day_files.items():
        df_list = []
        for file, stock_id in files:
            df = pd.read_csv(file)
            df['StockID'] = stock_id
            df_list.append(df)
        
        # 合并当天的所有数据
        combined_df = pd.concat(df_list, ignore_index=True)
        
        # 按时间戳和股票ID排序
        if 'localtime' not in combined_df.columns:
            combined_df['localtime'] = combined_df.index
        combined_df = combined_df.sort_values(['localtime', 'StockID'])
        
        # 创建输出目录（如果不存在）
        os.makedirs(output_dir, exist_ok=True)
        
        # 生成输出文件名
        output_file = os.path.join(output_dir, f'Day_{day}_ALL_STOCKS.csv')
        
        # 保存合并后的数据
        combined_df.to_csv(output_file, index=False)
        print(f"Day {day} 合并完成，输出文件：{output_file}")

# 使用示例
input_dir = './data_collection/snapshots'  # 输入目录
output_dir = './data_collection/Merged_Data'  # 输出目录

merge_csv_files(input_dir, output_dir)