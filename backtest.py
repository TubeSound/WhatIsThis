import os
import shutil
import sys
from datetime import datetime, timedelta
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from dateutil import tz
JST = tz.gettz('Asia/Tokyo')
UTC = tz.gettz('utc')

from technical import VWAP, RCI
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
    def __init__(self, symbol):
        self.symbol = symbol
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
        vwap_ma_window = code[0]
        vwap_median_window = code[1]
        vwap_threshold = code[2]
        rci_window = code[3]
        rci_pivot_threshold = code[4]
        rci_pivot_len = code[5]
        param = {'vwap': {  'begin_hour_list': [7, 19], 
                            'pivot_threshold': vwap_threshold, 
                            'pivot_left_len':5,
                            'pivot_center_len':7,
                            'pivot_right_len':5,
                            'median_window': vwap_median_window,
                            'ma_window': vwap_ma_window},
                    'rci': {'window': rci_window,
                            'pivot_threshold': rci_pivot_threshold,
                            'pivot_len': rci_pivot_len}
                    }
        return param

    def code_to_trade_param(self, code):
        mode = code[0]
        begin_hour = code[1]
        begin_minute = code[2]
        hours = code[3]
        sl = code[4]
        target_profit = code[5]
        trailing_stop = code[6]
        if trailing_stop == 0 or target_profit == 0:
            trailing_stop = target_profit = 0
        elif trailing_stop < target_profit:
            return None   
        param =  {
                    'mode': mode,
                    'begin_hour': begin_hour,
                    'begin_minute': begin_minute,
                    'hours': hours,
                    'sl': sl,
                    'target_profit': target_profit,
                    'trailing_stop': trailing_stop, 
                    'volume': 0.1, 
                    'position_max': 5, 
                    'timelimit': 0}
        return param
        
    def technical_gene_space(self):
        space = [
                    [GeneticCode.GeneInt,   10, 100, 10],  # vwap_ma_windo
                    [GeneticCode.GeneInt, 10, 100, 10],  # vwap_median_window
                    [GeneticCode.GeneFloat, 10, 50, 10],   # vwap_threshold
                    [GeneticCode.GeneInt,   10, 100, 10],  # rci_window
                    [GeneticCode.GeneFloat, 60, 100, 10],  # rci_pivot_threshold
                    [GeneticCode.GeneInt,   10, 50, 10]    # rci_pivot_len    
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
                    [GeneticCode.GeneInt, 1, 3, 1],  # mode
                    [GeneticCode.GeneInt, 7, 23, 1], # begin_hour
                    [GeneticCode.GeneList, [0, 30]], # begin_minute
                    [GeneticCode.GeneInt, 1, 20, 1], # hours
                    sl,                              # stoploss
                    target,                          # target_profit
                    trailing_stop                    # trailing_stop    
                ] 
        return space
    
class BackTest:
    def __init__(self, name, symbol, timeframe, data):
        self.name = name
        self.symbol = symbol
        self.timeframe = timeframe
        self.data = data
        
    def indicators(self, data, param):
        vwap_param = param['vwap']
        hours = vwap_param['begin_hour_list']
        VWAP(data,
            hours,
            vwap_param['pivot_threshold'],
            vwap_param['pivot_left_len'],
            vwap_param['pivot_center_len'],
            vwap_param['pivot_right_len'],
            vwap_param['median_window'],
            vwap_param['ma_window']
            )
        RCI(data, param['rci']['window'], param['rci']['pivot_threshold'], param['rci']['pivot_len'])
        
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
    def __init__(self, number : int, symbol, timeframe, repeat):
        self.number = number
        self.symbol = symbol
        self.timeframe = timeframe
        self.repeat = repeat
        
    def get_path(self, number: int, symbol: str):
        dir_path = './result'
        os.makedirs(dir_path, exist_ok=True)
        filename = str(number).zfill(2) + '_trade_summary_' + symbol + '_' + self.timeframe + '.xlsx'
        path = os.path.join(dir_path, filename)      
        return path
          
    def run(self):
        year_from = 2020
        month_from = 1
        year_to = 2024
        month_to = 5
        loader = DataLoader()
        symbol = self.symbol
        param = Parameters(symbol)
        timeframe = self.timeframe
        n, data = loader.load_data(symbol, timeframe, year_from, month_from, year_to, month_to)
        if n < 100:
            print('Data size small')
            return
        result = []
        for i in range(self.repeat):
            trade_param, technical_param = param.generate()                
            test = BackTest('', symbol, timeframe, data)
            df, summary, profit_curve = test.trade(trade_param, technical_param)
            d, columns= self.arrange_data(i, symbol, timeframe, technical_param, trade_param, summary)
            result.append(d)
            _, profit, _ = summary
            print(i, 'Profit', summary)
            if profit > 0:
                self.save_profit(symbol, timeframe, i, profit_curve)                    
            try:
                df = pd.DataFrame(data=result, columns=columns)
                df = df.sort_values('profit', ascending=False)
                path = self.get_path(self.number, symbol)
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
    
    def save_profit(self, symbol, timeframe, i, profit_curve):
        dir_path = os.path.join('./profit_curve', symbol, timeframe, str(self.number).zfill(2))
        os.makedirs(dir_path, exist_ok=True)
        filename = '#' + str(i).zfill(4) + '_profitcurve_' + symbol + '_' + timeframe + '.png'
        path = os.path.join(dir_path, filename)
        fig, ax = plt.subplots(1, 1, tight_layout=True)
        title = '#' + str(i).zfill(4) + '_profitcurve_' + symbol + '_' + timeframe
        ax.plot(range(len(profit_curve)), profit_curve, color='blue')
        ax.set_title(title)
        plt.savefig(path)   
        plt.close()               
              
def main():
    args = sys.argv
    if len(args) == 4:
        symbol = args[1]
        timeframe = args[2]
        number = args[3]
    else:
        symbol = 'NIKKEI'
        timeframe = 'M5'
        number = 0
        #raise Exception('Bad parameter')
    repeat = 1000
    opt = Optimizer(number, symbol, timeframe, repeat)
    opt.run()
    
def test():
    dic = {'a': 2.0, 'b': [3, 4], 'x': {'x0': 10, 'x1': {'y': 666, 'z':{'aho': True}}}}
    ret = expand('', dic)
    print(ret)
                
if __name__ == '__main__':
    main()
    #test()