import os
import shutil
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from dateutil import tz
JST = tz.gettz('Asia/Tokyo')
UTC = tz.gettz('utc') 
from common import Columns

class DataLoader:
    def server_time_str_2_datetime(self, server_time_str_list, server_timezone, format='%Y-%m-%d %H:%M:%S'):
        t_utc = []
        t_jst = []
        for time_str in server_time_str_list:
            i = time_str.find('+')
            if i > 0:
                time_str = time_str[:i]
            t = datetime.strptime(time_str, format)
            t = t.replace(tzinfo=server_timezone)
            utc = t.astimezone(UTC)
            t_utc.append(utc)
            jst = t.astimezone(JST)        
            t_jst.append(jst)
        return t_utc, t_jst

    def data_filepath(self, symbol, timeframe, year, month):
        path = '../MarketData/Axiory/'
        dir_path = os.path.join(path, symbol, timeframe)
        name = symbol + '_' + timeframe + '_' + str(year) + '_' + str(month).zfill(2) + '.csv'
        filepath = os.path.join(dir_path, name)
        if os.path.isfile(filepath):
            return filepath 
        else:
            return None
        
    def load_data(self, symbol, timeframe, from_year, from_month, to_year, to_month):
        dfs = []
        year = from_year
        month = from_month
        while True:
            filepath = self.data_filepath(symbol, timeframe, year, month)
            if filepath is not None:
                df = pd.read_csv(filepath)
                dfs.append(df)
            if year == to_year and month == to_month:
                break
            month += 1
            if month > 12:
                year += 1
                month = 1
                
        if len(dfs) == 0:
            return 0
        df = pd.concat(dfs, ignore_index=True)
        n = len(df)
        dic = {}
        for column in df.columns:
            dic[column] = list(df[column].values)
        tzone = timezone(timedelta(hours=2))
        #if timeframe.upper() == 'D1'or timeframe.upper() == 'W1':
        #    format='%Y-%m-%d'
        #else:
        format='%Y-%m-%d %H:%M:%S'
        utc, jst = self.server_time_str_2_datetime(dic[Columns.TIME], tzone, format=format)
        dic[Columns.TIME] = utc
        dic[Columns.JST] = jst
        print(symbol, timeframe, 'Data size:', len(jst), jst[0], '-', jst[-1])
        self.size = n
        return n, dic
    
    def data(self):
        return self.dic
    
    
def save(filepath, obj):
    import pickle
    with open(filepath, mode='wb') as f:
        pickle.dump(obj, f)
        
        
def arrange(data):
    time = data['time']
    jst = []
    utc = []
    for t in time:
        tj = t + timedelta(hours=14) 
        jst.append(tj.astimezone(JST))
        tu = t + timedelta(hours=2)
        utc.append(tu.astimezone(UTC))
    data['jst'] = jst
    data['utc'] = utc   
    
    
def load_backtest_market():
    file = './data/BacktestMarket/csv/spi500_h1.csv'
    df = pd.read_csv(file)
    date = df['date']
    time0 = df['time']
    
    time = []
    for d, t in zip(date, time0):
        tim = datetime.strptime(d + ' ' + t, '%d/%m/%Y %H:%M:%S')
        time.append(tim)
    
    data = {}    
    data['time'] = time
    data['open'] = df['open'].to_list()
    data['high'] = df['high'].to_list()
    data['low'] = df['low'].to_list()
    data['close']  = df['close'].to_list()
    arrange(data)
    save('./data/BacktestMarket/BM_spi500_H1.pkl', data)
    
if __name__ == '__main__':
    load_backtest_market()
        
        
        
        
        
        
        