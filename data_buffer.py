import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dateutil import tz

JST = tz.gettz('Asia/Tokyo')
UTC = tz.gettz('utc') 


def nans(length:int):
    out = [np.nan for _ in range(length)]
    return out

def time_utc(year: int, month: int, day: int, hour: int, minute: int):
    t = datetime(year, month, day, hour, minute)
    return t.replace(tzinfo=UTC)    


class DataBuffer:
    def __init__(self, time_column: str):
        self.time_column = time_column
        self.size = 0
        
    def initilize(self, arrays: dict):
        self.arrays = arrays
        self.keys = list(self.arrays.keys())
        self.size = len(arrays[self.keys[0]])    
        
    def time_array(self):
        return self.arrays[self.time_column]
        
    def time_last(self):
        time = self.time_array()
        return time[-1]
        
    def add_empty(self, keys: [str]):
        for key in keys:
            self.arrays[key] = nans(self.size)
        self.keys = list(self.arrays.keys())
        
    def shift(self, length=1):
        for key, array in self.arrays.items():
            new_array = array[length:] + nans(length)
            self.arrays[key] = new_array
            
    def slice_dic(self, data: dict, begin: int, end: int):
        dic = {}
        for key, array in data.items():
            dic[key] = array[begin: end + 1]
        return dic, (end - begin + 1)
            
    def split_data(self, data: dict):
        t_last = self.time_last()
        time = data[self.time_column]
        n = len(time)
        index = None
        for i, t in enumerate(time):
            if t > t_last:
                index = i
                break
        if index is None:
            return (data, None, 0)
        
        replace_data, length = self.slice_dic(data, 0, index - 1)
        new_data, new_length = self.slice_dic(data, index, n - 1)                 
        return (replace_data, new_data, new_length)
        
    def update(self, data: dict):
        length = None
        for key, value in data.items():
            if length is None:
                length = len(value)
            else:
                if length != len(value):
                    raise Exception('Dimension error')
        replace_data, new_data, new_length = self.split_data(data)
        self.replace(replace_data)
        if new_length > 0:
            self.add_data(new_data, new_length)
            return new_length
        else:
            return 0
                
    
    def replace(self, data: dict):
        time = self.time_array()
        t_list = data[self.time_column]
        for i, t in enumerate(t_list):
            for j, t0 in enumerate(time):
                if t == t0:
                    for key, array in data.items():
                        self.arrays[key][j] = array[i]
                    break
                
                
    def add_data(self, data: dict, length: int):
        for key in self.keys:
            if key in data.keys():
                new_array =  self.arrays[key][length:] + data[key]                
            else:
                new_array = self.arrays[key][length:] + nans(length)
            self.arrays[key] = new_array

    def get_data(self, key: str):
        return self.arrays[key]
    
    def data_last(self, key: str, length: int):
        array = self.arrays[key]
        return array[-length:]

    def update_data(self, key, data):
        length = len(data)
        array = self.arrays[key]
        begin = self.size - length
        for i, d in enumerate(data):
            array[begin + i] = d 
            
    
def test():
    import time
    from mt5_api import Mt5Api

    symbol = 'DOW'
    timeframe = 'M1'
    interval = 20
    bars = 200
    
    api = Mt5Api()
    api.connect()
    buffer = DataBuffer('time')
    for i in range(100):
        if i == 0:
            data = api.get_rates(symbol, timeframe, bars)
        else:
            data = api.get_rates(symbol, timeframe, 2)
        if buffer.size == 0:
            buffer.initilize(data)
        else:
            n = buffer.update(data)
            print('#', i, 'Update data size', n)
        time.sleep(interval)
    df = pd.DataFrame(buffer.arrays)
    df.to_csv('./bufferd_data.xlsx', index=False)
    
    data = api.get_rates(symbol, timeframe, bars)
    df = pd.DataFrame(data)
    df.to_csv('./refference.xlsx', index=False)
    
def test2():
    import time
    from mt5_api import Mt5Api
    from technical import VWAP
    
    symbol = 'NIKKEI'
    timeframe = 'M1'
    interval = 20
    bars = 800
    
    api = Mt5Api()
    api.connect()
    data = api.get_rates(symbol, timeframe, bars)
    VWAP(data, 1.8, [8, 16, 20])
    df = pd.DataFrame(data)
    df.to_csv('./nikkei_debug.csv', index=False)    
    
    
   
if __name__ == '__main__':
    test2()
    
    
    