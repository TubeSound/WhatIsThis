import os
import shutil
import sys
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from dateutil import tz
JST = tz.gettz('Asia/Tokyo')
UTC = tz.gettz('utc')

from technical import VWAP, RCI, ATR_TRAIL, SUPERTREND
from strategy import Simulation
from time_utils import TimeFilter
from data_loader import DataLoader

class GeneticCode:
    DataType = int
    GeneInt: DataType = 1
    GeneFloat: DataType = 2
    GeneList: DataType = 3
    def __init__(self, gene_space):
        self.gene_space = gene_space
        
    def gen_number(self, gene_space):
        typ = gene_space[0]
        if typ == self.GeneList:
            lis = gene_space[1]
            n = len(lis)
            i = random.randint(0, n - 1)
            return lis[i]
        begin = gene_space[1]
        end = gene_space[2]
        step = gene_space[3]
        num = int((end - begin) / step) + 1
        i = random.randint(0, num - 1)
        out = begin + step * i
        if typ == self.GeneInt:
            return int(out)
        elif typ == self.GeneFloat:
            return float(out)

    def create_code(self):
        n = len(self.gene_space)
        code = []
        for i in range(n):
            space = self.gene_space[i]
            value = self.gen_number(space)
            code.append(value)
        return code    

class Parameters:
    def __init__(self, symbol, strategy):
        self.symbol = symbol
        self.strategy = strategy
        self.trade_space = self.trade_gene_space()
        self.generator_trade = GeneticCode(self.trade_space)
        self.technical_space = self.technical_gene_space()
        self.generator_technical = GeneticCode(self.technical_space)

    def generate(self):    
        code1 = self.generator_trade.create_code()
        param1 = self.code_to_trade_param(code1)
        while param1 is None:
            code1 = self.generator_trade.create_code()
            param1 = self.code_to_trade_param(code1)
        code2 = self.generator_technical.create_code()
        param2 = self.code_to_technical_param(code2)
        return param1, param2
    
    def code_to_technical_param(self, code):
        strategy = self.strategy
        if strategy.find('VWAP') >= 0:
            return self.code_to_vwap_param(code)
        elif strategy == 'RCI':
            return self.code_to_rci_param(code)
        elif strategy == 'ATR_TRAIL':
            return self.code_to_atr_param(code)
        elif strategy == 'SUPERTREND':
            return self.code_to_supertrend_param(code)
        else:
            raise Exception('Bad strategy', strategy)
        
        
    def code_to_supertrend_param(self, code):
        param = {  'window': code[0], 
                    'multiply': code[1],
                    'break_count': code[2]
                }
        return {'SUPERTREND': param}    
        
    def code_to_vwap_param(self, code):
        vwap_ma_window = code[0]
        vwap_median_window = code[1]
        vwap_threshold = code[2]
        param = {  'begin_hour_list': [7, 19], 
                    'pivot_threshold': vwap_threshold, 
                    'pivot_left_len':5,
                    'pivot_center_len':7,
                    'pivot_right_len':5,
                    'median_window': vwap_median_window,
                    'ma_window': vwap_ma_window
                }
        return {'VWAP': param}
    
    def code_to_rci_param(self, code):
        param = {   'window': code[0],
                    'pivot_threshold': code[1],
                    'pivot_len': code[2]}
        return {'RCI': param}
    
    def code_to_atr_param(self, code):
        param = {   'window': code[0],
                    'multiply': code[1],
                    'peak_hold': code[2],
                    'horizon': code[3]}
        return {'ATR_TRAIL': param}
    
    def code_to_trade_param(self, code):
        begin_hour = code[0]
        begin_minute = code[1]
        hours = code[2]
        sl = code[3]
        target_profit = code[4]
        trailing_stop = code[5]
        only = code[5]
        if trailing_stop == 0 or target_profit == 0:
            trailing_stop = target_profit = 0
        elif trailing_stop < target_profit:
            return None   
        param =  {
                    'strategy': self.strategy,
                    'begin_hour': begin_hour,
                    'begin_minute': begin_minute,
                    'hours': hours,
                    'sl': sl,
                    'target_profit': target_profit,
                    'trailing_stop': trailing_stop, 
                    'volume': 0.1, 
                    'position_max': 5, 
                    'timelimit': 0,
                    'only': only}
        return param
        
    def technical_gene_space(self):
        strategy = self.strategy        
        if strategy == 'ATR_TRAIL':
            space = [
                        [GeneticCode.GeneInt,   10, 100, 10],  # window
                        [GeneticCode.GeneFloat, 0.6, 5.0, 0.2],  # multiply
                        [GeneticCode.GeneFloat, 5, 50, 5],   # peak_hold
                        [GeneticCode.GeneInt, 0, 5, 1]      # horizon
                    ]
        elif strategy.find('VWAP') >= 0:
            space = [
                        [GeneticCode.GeneInt,   10, 100, 10],  # vwap_ma_windo
                        [GeneticCode.GeneInt, 10, 100, 10],  # vwap_median_window
                        [GeneticCode.GeneFloat, 10, 50, 10],   # vwap_threshold
                    ]
        
        elif strategy == 'RCI':
            space = [
                        [GeneticCode.GeneInt,   10, 100, 10],  # rci_window
                        [GeneticCode.GeneFloat, 60, 100, 10],  # rci_pivot_threshold
                        [GeneticCode.GeneInt,   10, 50, 10]    # rci_pivot_len    
            ]
        elif strategy == 'SUPERTREND':
            space = [
                        [GeneticCode.GeneInt,   10, 100, 10],    # window
                        [GeneticCode.GeneFloat, 0.6, 5.0, 0.2],  # multiply
                        [GeneticCode.GeneInt,   0, 10, 1]        # break_count    
            ]
        return space

    def trade_gene_space(self):
        symbol = self.symbol
        gene_space = None
        if symbol == 'NIKKEI' or symbol == 'DOW':
            r =  [GeneticCode.GeneFloat, 50, 500, 50]    
        elif symbol == 'NSDQ': #16000
            r = [GeneticCode.GeneFloat, 20, 200, 20]
        elif symbol == 'HK50':    
            r = [GeneticCode.GeneFloat, 50, 400, 50]
        elif symbol == 'USDJPY' or symbol == 'EURJPY':
            r = [GeneticCode.GeneFloat, 0.05, 0.5, 0.05]
        elif symbol == 'EURUSD': #1.0
            r = [GeneticCode.GeneFloat, 0.0005, 0.005, 0.0005]
        elif symbol == 'GBPJPY':
            r = [GeneticCode.GeneFloat, 0.05, 0.5, 0.05]
        elif symbol == 'AUDJPY': # 100
            r = [GeneticCode.GeneFloat, 0.025, 0.5, 0.025]
        elif symbol == 'XAUUSD': #2000
            r = [GeneticCode.GeneFloat, 0.5, 5.0, 0.5] 
        elif symbol == 'CL': # 70
            r = [GeneticCode.GeneFloat, 0.02, 0.2, 0.02] 
        else:
            raise Exception('Bad symbol')

        d = [0.0] + list(np.arange(r[1], r[2], r[3]))
        sl = r
        trailing_stop =  [GeneticCode.GeneList, d] 
        target = r
        space = [ 
                    [GeneticCode.GeneInt, 7, 23, 1], # begin_hour
                    [GeneticCode.GeneList, [0, 30]], # begin_minute
                    [GeneticCode.GeneInt, 1, 20, 1], # hours
                    sl,                              # stoploss
                    target,                          # target_profit
                    trailing_stop,                    # trailing_stop    
                    [GeneticCode.GeneInt, -1, 1, 1] #only
                ] 
        return space
    
class BackTest:
    def __init__(self, name, symbol, timeframe, strategy, data):
        self.name = name
        self.symbol = symbol
        self.timeframe = timeframe
        self.strategy = strategy.upper()
        self.data = data
        
    def indicators(self, data, param):
        if self.strategy.find('VWAP') >= 0:
            p = param['VWAP']
            VWAP(data,
                p['begin_hour_list'],
                p['pivot_threshold'],
                p['pivot_left_len'],
                p['pivot_center_len'],
                p['pivot_right_len'],
                p['median_window'],
                p['ma_window']
            )
        elif self.strategy == 'RCI':
            p = param['RCI']
            RCI(data, p['window'], p['pivot_threshold'], p['pivot_len'])
        elif self.strategy == 'ATR_TRAIL':
            p = param['ATR_TRAIL']
            ATR_TRAIL(data, p['window'], p['multiply'], p['peak_hold'], p['horizon'])
        elif self.strategy == 'SUPERTREND':
            p = param['SUPERTREND']
            SUPERTREND(data, p['window'], p['multiply'], p['break_count'])
        
    def trade(self, trade_param, technical_param):
        data = self.data.copy()
        self.indicators(data, technical_param)
        sim = Simulation(trade_param)        
        df, summary, profit_curve = sim.run(data)
        trade_num, profit, win_rate = summary
        return (df, summary, profit_curve)

def expand(name: str, dic: dict):
    data = []
    columns = []
    for key, value in dic.items():
        if name == '':
            column = key
        else:
            column = name + '.' + key
        if type(value) == dict:
            d, c = expand(column, value)                    
            data += d
            columns += c
        else:
            data.append(value)
            columns.append(column)
    return data, columns 
  
class Optimizer:
    def __init__(self, number : int, symbol: str, timeframe: float, strategy: float, repeat: int):
        self.number = number
        self.symbol = symbol
        self.timeframe = timeframe
        
        strategy = strategy.upper()
        if strategy.find('ATR') >= 0:
            strategy = 'ATR_TRAIL'
        elif strategy.find('SUPER') >= 0:
            strategy = 'SUPERTREND'
        self.strategy = strategy
        
        self.repeat = repeat
        
    def get_path(self, number: int, year=None, month=None):
        dir_path = './result'
        os.makedirs(dir_path, exist_ok=True)
        if year is None:
            filename = str(number).zfill(2) + '_' +  self.strategy + '_' + self.symbol + '_' + self.timeframe + '.xlsx'
        else:
            filename = str(number).zfill(2) + '_' +  self.strategy + '_' + self.symbol + '_' + self.timeframe + '_' + str(year) + '_' + str(month).zfill(2) + '.xlsx'
        path = os.path.join(dir_path, filename)      
        return path
          
    def run(self):
        year_from = 2018
        month_from = 10
        year_to = 2020
        month_to = 7
        loader = DataLoader()
        timeframe = self.timeframe
        n, data = loader.load_data(self.symbol, self.timeframe, year_from, month_from, year_to, month_to)
        if n < 100:
            print('Data size small')
            return
        self.trade(data)
        
    def trade(self, data, year=None, month=None):
        symbol = self.symbol
        param = Parameters(symbol, self.strategy)
        result = []
        for i in range(self.repeat):
            trade_param, technical_param = param.generate()                
            test = BackTest('', symbol, self.timeframe, self.strategy, data)
            df, summary, profit_curve = test.trade(trade_param, technical_param)
            d, columns= self.arrange_data(i, symbol, self.timeframe, technical_param, trade_param, summary)
            result.append(d)
            _, profit, _ = summary
            print(i, 'Profit', summary)
            if profit > 500:
                self.save_profit_curve(symbol, self.timeframe, i, profit_curve, year, month)                    
            try:
                df = pd.DataFrame(data=result, columns=columns)
                df = df.sort_values('profit', ascending=False)
                path = self.get_path(self.number, year=year, month=month)
                df.to_excel(path, index=False)
            except:
                pass
            
    def arrange_data(self, i, symbol, timeframe, technical_param, trade_param, summary):
        data = []
        columns = ['No', 'symbol', 'timeframe']
        data += [i, symbol, timeframe]
        for param in [technical_param, trade_param]:
            d, c = expand('', param)
            data += d
            columns += c
        columns += ['trade_num', 'profit', 'win_rate']    
        data += list(summary)
        return data, columns
    
    def save_profit_curve(self, symbol, timeframe, i, profit_curve, year, month):
        dir_path = os.path.join('./profit_curve', self.strategy,  symbol, timeframe, str(self.number).zfill(2))
        os.makedirs(dir_path, exist_ok=True)
        if year is None:
            filename = '#' + str(i).zfill(4) + '_' + self.strategy + '_' + symbol + '_' + timeframe + '.png'
        else:
            filename = '#' + str(i).zfill(4) + '_' + self.strategy + '_' + symbol + '_' + timeframe + str(year) + '_' + str(month).zfill(2) + '.png'
        path = os.path.join(dir_path, filename)
        fig, ax = plt.subplots(1, 1, tight_layout=True)
        if year is None:
            title = '#' + str(i).zfill(4) +  ' ' + self.strategy + ' ' + symbol + ' ' + timeframe
        else:
            title = '#' + str(i).zfill(4) +  ' ' + self.strategy + ' ' + symbol + ' ' + timeframe + '(' + str(year) + '-' + str(month).zfill(2) + ')'
            
        ax.plot(range(len(profit_curve)), profit_curve, color='blue')
        ax.set_title(title)
        try:
            plt.savefig(path)   
            plt.close()               
        except:
            pass      
def main():
    args = sys.argv
    if len(args) == 5:
        symbol = args[1]
        timeframe = args[2]
        strategy = args[3]
        number = args[4]
    elif len(args) == 1:
        symbol = 'DOW'
        timeframe = 'M15'
        strategy = 'SUPER'
        number = 0
    else:
        raise Exception('Bad parameter')
    repeat = 1000
    opt = Optimizer(number, symbol, timeframe, strategy, repeat)
    opt.run()
    
def test():
    dic = {'a': 2.0, 'b': [3, 4], 'x': {'x0': 10, 'x1': {'y': 666, 'z':{'aho': True}}}}
    ret = expand('', dic)
    print(ret)
                
if __name__ == '__main__':
    main()
    #test()