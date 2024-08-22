import os
import shutil
import sys
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import pandas as pd
from dateutil import tz
JST = tz.gettz('Asia/Tokyo')
UTC = tz.gettz('utc')

from common import Indicators
from technical import SUPERTREND, SUPERTREND_SIGNAL, ATRP, MA, FILTER_MA_ATRP
from strategy import Simulation
from time_utils import TimeFilter, TimeUtils
from data_loader import DataLoader

def makeFig(rows, cols, size):
    fig, ax = plt.subplots(rows, cols, figsize=(size[0], size[1]))
    return (fig, ax)

def gridFig(row_rate, size):
    rows = sum(row_rate)
    fig = plt.figure(figsize=size)
    gs = gridspec.GridSpec(rows, 1, hspace=0.6)
    axes = []
    begin = 0
    for rate in row_rate:
        end = begin + rate
        ax = plt.subplot(gs[begin: end, 0])
        axes.append(ax)
        begin = end
    return (fig, axes)


def from_pickle(symbol, timeframe):
    import pickle
    if symbol == 'DOW' and timeframe == 'M15':
        filepath = './data/BacktestMarket/BM_dji_15min.pkl'
    elif symbol == 'NIKKEI' and timeframe == 'M15':
        filepath = './data/BacktestMarket/BM_nikkei_15min.pkl'
    else:
        filepath = './data/Axiory/' + symbol + '_' + timeframe + '.pkl'
    with open(filepath, 'rb') as f:
        data0 = pickle.load(f)
    return data0

def timefilter(data, year_from, month_from, year_to, month_to):
    t0 = datetime(year_from, month_from, 1).astimezone(JST)
    t1 = datetime(year_to, month_to, 1).astimezone(JST)
    t1 += relativedelta(months=1)
    t1 -= timedelta(days=1)
    return TimeUtils.slice(data, data['jst'], t0, t1)
        
class BackTest:
    def __init__(self, name, symbol, timeframe, strategy, data):
        self.name = name
        self.symbol = symbol
        self.timeframe = timeframe
        self.strategy = strategy.upper()
        self.data = data
        
  
    def trade(self, trade_param, technical_param):
        data = self.data.copy()
        self.indicators(data, technical_param)
        sim = Simulation(trade_param)        
        df, summary, profit_curve = sim.run(data, 'MABAND_LONG', 'MABAND_SHORT')
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
    def __init__(self, data, number : int, symbol: str, timeframe: float, repeat: int):
        self.data = data
        self.number = number
        self.symbol = symbol
        self.timeframe = timeframe
        

        
        
    def get_dirpath(self):
        return './2018.10-2024.5/'
        
        
    def get_path(self, number: int, year=None, month=None):
        dirpath = os.path.join(self.get_dirpath(), 'result')
        os.makedirs(dirpath, exist_ok=True)
        if year is None:
            filename = str(number).zfill(2) + '_' +  self.strategy + '_' + self.symbol + '_' + self.timeframe + '.xlsx'
        else:
            filename = str(number).zfill(2) + '_' +  self.strategy + '_' + self.symbol + '_' + self.timeframe + '_' + str(year) + '_' + str(month).zfill(2) + '.xlsx'
        path = os.path.join(dirpath, filename)      
        return path
          

        
    
    def run(self):
        year_from = 2018
        month_from = 10
        year_to = 2024
        month_to = 7
        n, data = self.from_pickle(year_from, month_from, year_to, month_to)
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
            if profit > 10000:
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
        dirpath = os.path.join(self.get_dirpath(), 'profit_curve', self.strategy,  symbol, timeframe, str(self.number).zfill(2))
        os.makedirs(dirpath, exist_ok=True)
        if year is None:
            filename = '#' + str(i).zfill(4) + '_' + self.strategy + '_' + symbol + '_' + timeframe + '.png'
        else:
            filename = '#' + str(i).zfill(4) + '_' + self.strategy + '_' + symbol + '_' + timeframe + str(year) + '_' + str(month).zfill(2) + '.png'
        path = os.path.join(dirpath, filename)
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
def opt():
    args = sys.argv
    if len(args) == 5:
        symbol = args[1]
        timeframe = args[2]
        strategy = args[3]
        number = args[4]
    elif len(args) == 1:
        symbol = 'DOW'
        timeframe = 'M15'
        strategy = 'SUPERTREND'
        number = 0
    else:
        raise Exception('Bad parameter')
    repeat = 1000
    opt = Optimizer(number, symbol, timeframe, strategy, repeat)
    opt.run()
    
    
def get_trade_param():
    param =  {
                'strategy': 'supertrend',
                'begin_hour': 0,
                'begin_minute': 0,
                'hours': 0,
                'sl': {
                        'method': 'fix',
                        'value': 400
                    },
                'target_profit': None,
                'trailing_stop': None, 
                'volume': 0.1, 
                'position_max': 5, 
                'timelimit': 0}
    return param


def plot(data):
    jst = data['jst']
    op = data['open']
    hi = data['high']
    lo = data['low']
    cl = data['close']


    ma_short = data['MA_SHORT']
    limit_upper = data[Indicators.SUPERTREND_UPPER]
    limit_lower = data[Indicators.SUPERTREND_LOWER]
    atr_u = data[Indicators.ATR_UPPER]
    atr_l = data[Indicators.ATR_LOWER]
    signal = data[Indicators.SUPERTREND_SIGNAL]
    #profit, df, curve = calc_profit(data, long_event, short_event)
    #print(df)
    #drawdown = calc_drawdown(curve[1])
    #print(df)
    #print('Profit:', profit, drawdown)
    
    #axes[0].plot(jst, cl, color='blue')
    fig, axes = gridFig([4, 1], (15, 8))
    axes[0].plot(jst, ma_short, color='blue')
    axes[0].plot(jst, atr_u, color='gray', linewidth=1.0)
    axes[0].plot(jst, atr_l, color='gray', linewidth=1.0)
    axes[0].plot(jst, limit_upper, color='orange', linewidth=2.0)
    axes[0].plot(jst, limit_lower, color='green', linewidth=2.0)
    
    
    for i, s in enumerate(signal):
        if s == 1:
            axes[0].scatter(jst[i], ma_short[i], color='green', marker='o', s=200, alpha=0.5)
        if s == -1:
            axes[0].scatter(jst[i], ma_short[i], color='red', marker='o', s=200, alpha=0.8)

  
    #axes[2].plot(curve[0], curve[1])
    
    [ax.set_xlim(jst[0], jst[-1]) for ax in axes]    
    form = mdates.DateFormatter("%m/%d %H:%M")
    [ax.xaxis.set_major_formatter(form) for ax in axes]
    
    
def plot_profit(path, number, param, curve):
    fig, ax = plt.subplots(1, 1, figsize=(15, 4))
    ax.plot(curve[0], curve[1], color='blue')
    ax.set_title("#" + str(number) + ' ' + str(param))
    fig.savefig(path)
    plt.close()
    
def trade(symbol, timeframe, data, param):
    trade_param = get_trade_param()
    sim = Simulation(data, trade_param)        
    return sim.run_doten(Indicators.SUPERTREND_SIGNAL, entry_filter_column=Indicators.FILTER_MA_ATRP)
    
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
    

def optimize(symbol, timeframe, year):
    print(symbol, timeframe)
    months = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]    
    data0 = from_pickle(symbol, timeframe)
    root = './optimize'
    for m in range(len(months)):
        data = data0.copy()
        month = months[m]
        month_from = month[0]
        month_to = month[-1]
        print(symbol, timeframe, year, month)
        n, data = timefilter(data0, year, month_from, year, month_to)
        title = f'{symbol}_{timeframe}_{year}_{month_from}'
        sim(root, symbol, timeframe, title, data)
        
def optimize_fulltime(symbol, timeframe):
    print(symbol, timeframe)
    ma_window = 25
    limit = 20000
    data0 = from_pickle(symbol, timeframe)
    root = f'./optimize_full/MA{ma_window}'
    data = data0.copy()
    title = f'{symbol}_{timeframe}_MA{ma_window}'
    sim(root, symbol, timeframe, title, data, ma_window, limit)    
    
def sim(root, symbol, timeframe, title, data, ma_window, limit):
    dir_path = os.path.join(root, f'{symbol}/{timeframe}')
    os.makedirs(dir_path, exist_ok=True)
    number = 0
    out = []
    matrix = []
    cols = []

    for atr_window in range(5, 90, 5):
        line = [atr_window]
        cols = ['->multiply']
        for multiply in np.arange(0.5, 5, 0.5):
            cols.append(str(multiply))
            number += 1
            param = {'atr_window': atr_window, 'atr_multiply': multiply, 'ma_window': ma_window, 'filter_ma': 4 * 24 * 4, 'atrp_window': 40, 'atrp_ma': 40, 'atrp_threshold': 0.2}
            SUPERTREND(data, param['atr_window'], param['atr_multiply'], param['ma_window'])
            SUPERTREND_SIGNAL(data, 0)
            ATRP(data, param['atrp_window'], ma_window=param['atrp_ma'])
            MA(data, param['filter_ma'])
            FILTER_MA_ATRP(data, data[Indicators.MA], data[Indicators.ATRP], param['atrp_threshold'])
            r = trade(symbol, timeframe, data, param)
            if r is None:
                line.append(0)
                continue
            df , summary, curve = r
            p, columns = expand('supertrend', param)
            out.append([number] + p + list(summary))
            line.append(summary[1])
            if summary[1]> limit:
                print(summary)
                print(param)
                path = f'profit_curve_#{number}_{title}.png'
                plot_profit(os.path.join(dir_path, path), number, param, curve)
        matrix.append(line)
    df = pd.DataFrame(data=out, columns=['no', 'atr_window', 'atr_multiply', 'ma_wndow', 'n', 'profit', 'drawdown'])           
    path = os.path.join(dir_path, f'summary_{title}.xlsx')
    df.to_excel(path, index=False)           
    df = pd.DataFrame(data=matrix, columns=cols)
    path = os.path.join(dir_path, f'matrix_{title}.xlsx')
    df.to_excel(path, index=False)    

    
def optimize_crash(symbol, timeframe):
    print(symbol, timeframe)  
    data0 = from_pickle(symbol, timeframe)
    for ma_window in range(10, 60, 5):
        root = f'./optimize/Axiory/covid/MA{ma_window}'
   
        year = 2020
        month_from = 1
        month_to = 4
        n, data = timefilter(data0, year, month_from, year, month_to)
        title = f'{symbol}_{timeframe}_MA{ma_window}_{year}_{month_from}'
        sim(root, symbol, timeframe, title, data, ma_window, 5000)

def main1():
    args = sys.argv
    if len(args) == 4:
        symbol = args[1]
        timeframe = args[2]
        year = int(args[3])
    else:
        symbol = 'DOW'
        timeframe = 'M15'
        year = 2024
    optimize(symbol, timeframe, year)
    
def main2():
    args = sys.argv
    if len(args) == 3:
        symbol = args[1]
        timeframe = args[2]
    else:
        symbol = 'NIKKEI'
        timeframe = 'M5'
    optimize_fulltime(symbol, timeframe)
    
    
def main3():
    symbol = 'NIKKEI'
    timeframe = 'M5'
    optimize_crash(symbol, timeframe)
    
if __name__ == '__main__':
    main2()
    #test()