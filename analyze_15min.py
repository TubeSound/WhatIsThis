# -*- coding: utf-8 -*-
"""
Created on Sat Aug 10 09:54:33 2024

@author: docs9
"""

import os
import shutil
import sys
sys.path.append('../Libraries/trade')

from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
from dateutil import tz
JST = tz.gettz('Asia/Tokyo')
UTC = tz.gettz('utc')

from common import Columns, Indicators
from candle_chart import CandleChart, makeFig, gridFig
from utils import TimeUtils
from data_loader import from_pickle
from technical import sma, MAGAP, MAGAP_SIGNAL, SUPERTREND, SUPERTREND_SIGNAL, detect_gap_cross, detect_peaks


def round_time(time, interval_sec):
    seconds = time.hour * 60 * 60 + time.minute * 60 + time.second
    rounded = int(seconds / interval_sec) * interval_sec
    hour = int(rounded / 60 / 60)
    rounded -= hour * 60 * 60
    minute = int(rounded / 60)
    rounded -= minute * 60 
    t = datetime(time.year, time.month, time.day, hour, minute, rounded)
    return t

def sample(time, values, func, interval_sec):
    current = None
    tohlc = []
    for t, v in zip(time, values):
        tround = round_time(t, interval_sec)
        if current is None:
            current = tround
            data = [v]
        else:
            if tround > current:
                tohlc.append([tround] + func(data))
                data = [v]
                current = tround
            else:
                data.append(v)
    data.append(v)
    tohlc.append([tround] + func(data))

    arrays = [[], [], [], [], []]
    for t, o, h, l, c in tohlc:
        arrays[0].append(t)
        arrays[1].append(o)
        arrays[2].append(h)
        arrays[3].append(l)
        arrays[4].append(c)
    return tohlc, arrays


def ohlc(data):
    return [data[0], max(data),  min(data), data[-1]]

def save(filepath, obj):
    import pickle
    with open(filepath, mode='wb') as f:
        pickle.dump(obj, f)



def arrange():
    
    data0 = from_pickle()
    time = data0['time']
    jst = []
    utc = []
    for t in time:
        tj = t + timedelta(hours=14) 
        jst.append(tj.astimezone(JST))
        tu = t + timedelta(hours=2)
        utc.append(tu.astimezone(UTC))
   
    
    data0['jst'] = jst
    data0['utc'] = utc   
    
    save('./data/BacktestMarket/BM_dji_15min.pkl', data0)
    
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
    
def main():
    symbol = 'DOW'
    timeframe = 'M15'     
    data0 = from_pickle(symbol, timeframe)
    jst = data0['jst']
    
    technical_param = {'MAGAP': {'long_term': 4 * 24 * 4 ,
                                 'mid_term': 4 * 24 * 2 ,
                                 'short_term': 4 * 16,
                                 'tap': 16, 
                                 'level': 0.1, 
                                 'threshold': 0.1,
                                  'slope_threshold': 0.03,
                                  'delay_max': 16},
                       'SUPERTREND': { 'atr_window': 40,
                                      'atr_multiply': 3.0,
                                      'short_term': 7
                                      }
                       
                       }
    
    
    for year in range(2019, 2025):
        t0 = datetime(year, 1, 1).astimezone(JST)
        t1 = datetime(year, 12, 31).astimezone(JST)
        n, data1 = TimeUtils.slice(data0, jst, t0, t1)
        if n < 100:
            continue
        param = technical_param['MAGAP']
        MAGAP(data1, param['long_term'], param['mid_term'], param['short_term'], param['tap'], timeframe)
        param = technical_param['SUPERTREND']
        SUPERTREND(data1, param['atr_window'], param['atr_multiply'])
        SUPERTREND_SIGNAL(data1, param['short_term'])
        calc(data1, symbol, timeframe, year, technical_param)
    
def calc(data, symbol, timeframe, year, technical_param):
    for month in range(1, 13):
        t0 = datetime(year, month, 1).astimezone(JST)
        t1 = t0 + timedelta(days=31)
        n, data1 = TimeUtils.slice(data, data['jst'], t0, t1)
        if n > 50:
            plot(data1, symbol, timeframe, year, month, technical_param)
    
    
def plot(data, symbol, timeframe, year, month, technical_param):
    jst = data['jst']
    cl = data[Columns.CLOSE]
    trend = data[Indicators.SUPERTREND]
    atrp = data[Indicators.ATRP]
    gap = data[Indicators.MAGAP]
    slope = data[Indicators.MAGAP_SLOPE]
    
    param = technical_param['MAGAP']
    xup, xdown = MAGAP_SIGNAL(data, param['slope_threshold'], param['delay_max'])
    peaks = detect_peaks(gap)
    
    fig, axes = gridFig([4, 4, 2], (18, 12))
    axes[0].scatter(jst, cl, color='cyan', alpha=0.1, s=5)
    axes[0].plot(jst, data[Indicators.MA_LONG], color='purple', label='Long')
    axes[0].plot(jst, data[Indicators.MA_MID], color='blue', label='Mid')
    axes[0].plot(jst, data[Indicators.MA_SHORT], color='red', label='Short')
    axes[0].plot(jst, data[Indicators.SUPERTREND_U], color='green', linestyle='dotted', linewidth=2.0)
    axes[0].plot(jst, data[Indicators.SUPERTREND_L], color='red', linestyle='dotted', linewidth=2.0)
    axes[1].plot(jst, gap, color='blue')
    #ax = plt.twinx(axes[1])
    #ax.scatter(jst, cl, color='cyan', alpha=0.1, s=5)
    axes[2].plot(jst, atrp, color='orange')
    
    for i in xup:
        axes[1].scatter(jst[i], gap[i], marker='^', color='green', alpha=0.4, s=200)
        axes[1].text(jst[i - 50], gap[i] + 1.0, str(slope[i])[:5])
        
    for i in xdown:
        axes[1].scatter(jst[i], gap[i], marker='v', color='red', alpha=0.4, s=200)
        axes[1].text(jst[i - 30], gap[i] - 1.0, str(slope[i])[:5])
    
    t0 = datetime(year, month, 1)
    t1 = t0 + timedelta(days=31)
    for i in range(1, 2):
        axes[i].hlines(0.0, t0, t1, color='yellow')
    for ax in axes:
        #ax.legend()
        ax.set_xlim(t0, t1)
        ax.grid()
    axes[1].set_ylim(-4.0, 4.0)
    axes[2].set_ylim(0, 2.0)
    title = f'{symbol} {timeframe} {year}/{month} {technical_param}'
    axes[0].set_title(title)
    axes[1].set_title('MAGAP')
    axes[2].set_title('ATRP')
    dirpath = f'./chart/{symbol}/{timeframe}'
    os.makedirs(dirpath, exist_ok=True)
    name = f'magap_chart_{symbol}_{timeframe}_{year}_{month}.png'
    fig.savefig(os.path.join(dirpath, name))
    #plt.close(fig)
    
def test():
    
    t = datetime(2024, 8, 10, 12, 15, 16)
    print(t, '->', round_time(t, 60 * 15))
    
if __name__ == '__main__':
    main()
    

