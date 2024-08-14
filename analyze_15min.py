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

from candle_chart import CandleChart, makeFig, gridFig
from utils import TimeUtils
from technical import sma


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

def from_pickle():
    import pickle
    filepath = './data/BacktestMarket/BM_dji_15min.pkl'
    with open(filepath, 'rb') as f:
        data0 = pickle.load(f)
    return data0

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
    
    
def main():
     
    data0 = from_pickle()
    jst = data0['jst']
    
    t0 = datetime(2024, 8, 5, 15, 0).astimezone(JST)
    t1 = datetime(2024, 8, 6, 12, 0).astimezone(JST)
    n, data = TimeUtils.slice(data0, jst, t0, t1)

    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    chart = CandleChart(fig, ax, date_format='%d/%H:%M')
    chart.drawCandle(data['jst'], data['open'], data['high'], data['low'], data['close'], xlabel=True)
    #ma = sma(data['close'], 5)
    #chart.drawLine(data['jst'], ma)
    ax.legend()
    locator = mdates.AutoDateLocator(minticks=6, maxticks=10)
    ax.xaxis.set_major_locator(locator)



def test():
    
    t = datetime(2024, 8, 10, 12, 15, 16)
    print(t, '->', round_time(t, 60 * 15))
    
if __name__ == '__main__':
    main()
    

