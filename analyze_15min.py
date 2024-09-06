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
    
    
def main():
    symbol = 'NIKKEI'
    timeframe = 'M15'     
    data0 = from_pickle(symbol, timeframe)
    jst = data0['jst']
    
    for year in range(2019, 2025):
        t0 = datetime(year, 1, 1).astimezone(JST)
        t1 = datetime(year, 12, 31).astimezone(JST)
        n, data1 = TimeUtils.slice(data0, jst, t0, t1)
        if n < 100:
            continue
        MAGAP(data1, 4 * 24 * 2, 4 * 16, 16, timeframe)
        SUPERTREND(data1, 40, 3.0)
        SUPERTREND_SIGNAL(data1, 7)
        calc(data1, symbol, timeframe, year)
    
def calc(data, symbol, timeframe, year):
    for month in range(1, 13):
        t0 = datetime(year, month, 1).astimezone(JST)
        t1 = t0 + timedelta(days=31)
        n, data1 = TimeUtils.slice(data, data['jst'], t0, t1)
        if n > 50:
            plot(data1, symbol, timeframe, year, month)
    
    
def plot(data, symbol, timeframe, year, month):
    jst = data['jst']
    cl = data[Columns.CLOSE]
    trend = data[Indicators.SUPERTREND]
    atrp = data[Indicators.ATRP]
    gap = data[Indicators.MAGAP]
    up, down = MAGAP_SIGNAL(data, 0.1, 1.0)
    xup, xdown = detect_gap_cross(gap, data[Indicators.MAGAP_SLOPE], 0.05)
    peaks = detect_peaks(gap)
    
    fig, axes = gridFig([4, 4, 2], (20, 12))
    axes[0].scatter(jst, cl, color='cyan', alpha=0.1, s=5)
    axes[0].plot(jst, data[Indicators.MA_LONG], color='blue')
    axes[0].plot(jst, data[Indicators.MA_SHORT], color='red')
    axes[0].plot(jst, data[Indicators.SUPERTREND_U], color='green', linestyle='dotted', linewidth=2.0)
    axes[0].plot(jst, data[Indicators.SUPERTREND_L], color='red', linestyle='dotted', linewidth=2.0)
    axes[1].plot(jst, gap, color='blue')
    #ax = plt.twinx(axes[1])
    #ax.scatter(jst, cl, color='cyan', alpha=0.1, s=5)
    axes[2].plot(jst, atrp, color='orange')
    
    for i, value in xup:
        axes[1].scatter(jst[i], gap[i], marker='^', color='green', alpha=0.4, s=200)
        axes[1].text(jst[i - 50], gap[i] + 1.0, str(value)[:5])
        
    for i, value in xdown:
        axes[1].scatter(jst[i], gap[i], marker='v', color='red', alpha=0.4, s=200)
        axes[1].text(jst[i - 30], gap[i] - 1.0, str(value)[:5])
        
    for i in up:
        axes[1].scatter(jst[i], gap[i], marker='^', color='gray', alpha=0.8, s=100)
        
    for i in down:
        axes[1].scatter(jst[i], gap[i], marker='v', color='gray', alpha=0.8, s=100)
    
    for i in range(1, 2):
        axes[i].hlines(0.0, jst[0], jst[-1], color='yellow')

    for ax in axes:
        ax.legend()
        ax.set_xlim(jst[0], jst[-1])
        ax.grid()
    axes[1].set_ylim(-7.0, 7.0)
    axes[2].set_ylim(0, 2.0)
    title = f'{symbol} {timeframe} {year}/{month}'
    axes[0].set_title(title)
    axes[1].set_title('MAGAP')
    axes[2].set_title('ATR')
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
    

