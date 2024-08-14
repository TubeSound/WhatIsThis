# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 21:44:13 2023

@author: docs9
"""

import os
import pandas as pd
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from mt5_api import Mt5Api
from common import TimeFrame
from datetime import datetime, timedelta
from dateutil import tz
from time_utils import TimeFilter, TimeUtils
from data_loader import DataLoader

JST = tz.gettz('Asia/Tokyo')
UTC = tz.gettz('utc') 

def download(symbols, save_holder):
    api = Mt5Api()
    api.connect()
    for symbol in symbols:
        for tf in [TimeFrame.M1, TimeFrame.M5, TimeFrame.M15, TimeFrame.M30, TimeFrame.H1, TimeFrame.H4, TimeFrame.D1]:
            for year in range(2024, 2025):
                for month in range(8, 9):
                    t0 = datetime(year, month, 1, 7)
                    t0 = t0.replace(tzinfo=JST)
                    t1 = t0 + relativedelta(months=1) - timedelta(seconds=1)
                    if tf == 'TICK':
                        rates = api.get_ticks(t0, t1)
                    else:
                        rates = api.get_rates_jst(symbol, tf, t0, t1)
                    path = os.path.join(save_holder, symbol, tf)
                    os.makedirs(path, exist_ok=True)
                    path = os.path.join(path, symbol + '_' + tf + '_' + str(year) + '_' + str(month).zfill(2) + '.csv')
                    df = pd.DataFrame(rates)
                    if len(df) > 10:
                        df.to_csv(path, index=False)
                    print(symbol, tf, year, '-', month, 'size: ', len(df))
    
    pass


def download_tick(symbols, save_holder):
    api = Mt5Api()
    api.connect()
    for symbol in symbols:
        year = 2024
        month = 7
        tf = 'TICK'
        t0 = datetime(year, month, 25, 7)
        t0 = t0.replace(tzinfo=JST)
        t1 = datetime.now(JST)
        df = api.get_ticks(symbol, t0, t1)
        path = save_holder
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, symbol + '_' + tf + '_' + str(year) + '_' + str(month).zfill(2) + '.pkl')
        #print(path, symbol, len(df))
        save(path, df)
        print(path, symbol, tf, year, '-', month, 'size: ', len(df))
    
    pass


def dl1():
    symbols = ['NIKKEI', 'DOW', 'NSDQ', 'SP', 'HK50', 'DAX', 'FTSE', 'XAUUSD']
    symbols += ['CL', 'USDJPY', 'GBPJPY']
    symbols += ['HK50', 'NGAS', 'EURJPY', 'AUDJPY', 'EURUSD']
    download(symbols, '../MarketData/Axiory/')
    
def dl2():
    symbols = ['SP', 'HK50', 'DAX', 'FTSE',  'XAGUSD', 'EURJPY', 'AUDJPY']
    symbols = ['NIKKEI', 'USDJPY']
    download(symbols, '../MarketData/Axiory/')
    
def save(filepath, obj):
    import pickle
    with open(filepath, mode='wb') as f:
        pickle.dump(obj, f)
    
def save_data():
    year_from = 2020
    month_from = 1
    year_to = 2024
    month_to = 8
    loader = DataLoader()
    for symbol in ['NIKKEI', 'DOW', 'NSDQ', 'USDJPY']:
        for tf in ['M1', 'M5', 'M15', 'M30']:
            n, data = loader.load_data(symbol, tf, year_from, month_from, year_to, month_to)
            os.makedirs('./data', exist_ok=True)
            save('./data/' + symbol + '_' + tf + ".pkl", data)
    
def main():
    dl1()
    save_data()
    #download_tick(['NIKKEI', 'DOW', 'USDJPY'], '../data/tick')
    
    
if __name__ == '__main__':
    main()
