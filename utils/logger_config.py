import logging
import os

def setup_logger(log_name:str):
    if os.path.exists(f'./log/{log_name}.log'):
        os.remove(f'./log/{log_name}.log')
    if os.path.exists(f'./log/{log_name}_info.log'):
        os.remove(f'./log/{log_name}_info.log')

    logger = logging.getLogger('quant_logger')
    logger.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler(f'./log/{log_name}.log', mode='w')
    file_handler.setLevel(logging.DEBUG)
    file_info_handler = logging.FileHandler(f'./log/{log_name}_info.log', mode='w')
    file_info_handler.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    file_info_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(file_info_handler)
    logger.addHandler(console_handler)

    return logger