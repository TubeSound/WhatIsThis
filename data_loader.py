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