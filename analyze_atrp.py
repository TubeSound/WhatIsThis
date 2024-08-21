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
from technical import sma, ATRP, is_nan


cmap = plt.get_cmap("tab10")

def from_pickle(symbol, timeframe, axiory=False):
    import pickle
    symbol = symbol.lower()
    timeframe = timeframe.upper()
    if axiory:
        filepath = f'./data/Axiory/{symbol}_{timeframe}.pkl'
    else:
        filepath = f'./data/BacktestMarket/BM_{symbol}_{timeframe}.pkl'
    with open(filepath, 'rb') as f:
        data0 = pickle.load(f)
    return data0

    
    
def main(): 
    symbol = 'DOW'
    timeframe = 'M15'
    number = 1
    data0 = from_pickle(symbol, timeframe)
    ATRP(data0, 24, ma_window=20)

    
    terms = []
    t0 = datetime(2008, 8, 1, 0, 0).astimezone(JST)
    t1 = datetime(2009, 3, 30, 0, 0).astimezone(JST)
    terms.append([t0, t1])
    t0 = datetime(2011, 6, 1, 0, 0).astimezone(JST)
    t1 = datetime(2011, 8, 30, 0, 0).astimezone(JST)
    terms.append([t0, t1])
    t0 = datetime(2015, 7, 1, 0, 0).astimezone(JST)
    t1 = datetime(2015, 9, 30, 0, 0).astimezone(JST)
    terms.append([t0, t1])
    t0 = datetime(2018, 1, 1, 0, 0).astimezone(JST)
    t1 = datetime(2018, 4, 30, 0, 0).astimezone(JST)
    terms.append([t0, t1])
    t0 = datetime(2020, 1, 1, 0, 0).astimezone(JST)
    t1 = datetime(2020, 6, 30, 0, 0).astimezone(JST)
    terms.append([t0, t1])
    t0 = datetime(2024, 7, 1, 0, 0).astimezone(JST)
    t1 = datetime(2024, 8, 30, 0, 0).astimezone(JST)
    terms.append([t0, t1])
    
    number = 1    
    for t0, t1 in terms:
        n, data = TimeUtils.slice(data0, data0['jst'], t0, t1)
    
        jst = data['jst']
        atrp = data['ATRP']
        cl = data['close']

        fig, axes = plt.subplots(2, 1, figsize=(12, 5))
        axes[0].plot(jst, cl, color='blue')
        axes[1].plot(jst, atrp, color='red')
        [ax.legend() for ax in axes]
        locator = mdates.AutoDateLocator(minticks=6, maxticks=10)
        #[ax.set_major_locator(locator) for ax in axes]
        axes[0].set_title(f'{symbol} {timeframe} ATRP #{number}')
        [ax.grid() for ax in axes]
        number += 1
    
    
def detect(data, threshold=0.5):
    atrp = data['ATRP']
    n = len(atrp)
    sig = [0 for _ in range(n)]
    for i in range(n):
        if is_nan(atrp[i]):
            continue
        if atrp[i] >= 1.0:
            sig[i] = 1 
            
    xup = []
    for i in range(1, n):
        if sig[i - 1] == 0 and sig[i] == 1:
            xup.append(i)
            
    break_points = []
    length1 = 24 * 40
    length2 = 24 * 5
    length = length1 + length2
    
    for i in range(length, n):
        d = atrp[i - length: i - length + length1]
        maxv = np.nanmax(d)
        if atrp[i] > maxv * 1.1 and atrp[i] > threshold:
            break_points.append(i)
            
    return sig, xup, break_points
        
def main2(): 
    symbol = 'NASDAQ'
    timeframe = 'H1'
    data0 = from_pickle(symbol, timeframe)
    ATRP(data0, 40, ma_window=40)
    for year in range(2008, 2025):
        t0 = datetime(year-1, 10, 1).astimezone(JST)
        t1 = datetime(year, 12, 31).astimezone(JST)
        plot(data0, year, symbol, timeframe,t0, t1)
        
def plot(data0, year, symbol, timeframe, t0, t1):
        n, data = TimeUtils.slice(data0, data0['jst'], t0, t1)
        signal, xup, break_points = detect(data)
        jst = data['jst']
        atrp = data['ATRP']
        cl = data['close']
    
        fig, axes = plt.subplots(2, 1, figsize=(20, 8))
        axes[0].plot(jst, cl, color='blue')
        axes[1].scatter(jst, atrp, color='green', alpha=0.1, s=1)
        [ax.legend() for ax in axes]
        locator = mdates.AutoDateLocator(minticks=12, maxticks=20)
        [ax.xaxis.set_major_locator(locator) for ax in axes]
        axes[0].set_title(f'{year} {symbol} {timeframe}  ATRP(40, 40) threshold: 0.5')
        for i in range(len(signal)):
            if signal[i] == 1:
                axes[1].scatter(jst[i], atrp[i], color='red', s=10, alpha=0.2)
                #end = n - 1 if (i + 480) >= n else i + 480
                #axes[1].hlines(0, jst[i], jst[end], color='red', linewidth=5.0)
        out = []
        for p in break_points:
            axes[0].scatter(jst[p], cl[p], color='orange', alpha=0.3, s=50)
            out.append([jst[p], atrp[i]])
        df = pd.DataFrame(data=out, columns=['jst', 'atrp'])
        os.makedirs('./debug', exist_ok=True)
        df.to_csv(f'./debug/{symbol}_{year}_atrp_breakout.csv', index=False)
            
            
        for x in xup:
            axes[0].scatter(jst[x], cl[x], color='red', alpha=0.9, marker='x', s= 400)
            
        [ax.grid() for ax in axes]
        [ax.set_xlim(t0, t1) for ax in axes]
        axes[1].set_ylim(0, 3)
        
def main4():
    symbol = 'SPI500'
    timeframe = 'H1'
    data0 = from_pickle(symbol, timeframe)
    ATRP(data0, 40, ma_window=40)
    t0 = datetime(2024, 5, 1).astimezone(JST)
    t1 = datetime(2024, 8, 10).astimezone(JST)
    plot(data0, 2024, symbol, timeframe, t0, t1)
    
def main5():
    symbols = ['NIKKEI', 'DOW', 'SP', 'NSDQ', 'USDJPY', 'XAUUSD']
    timeframe = 'H1'
    data = {}
    for symbol in symbols:
        data0 = from_pickle(symbol, timeframe, axiory=True)
        ATRP(data0, 40, ma_window=40)
        data[symbol] = data0

    dic = {}        
    for year in range(2020, 2025):
        for symbol, d in data.items():
            t0 = datetime(year, 1, 1).astimezone(JST)
            t1 = datetime(year, 8, 31).astimezone(JST)
            n, d1 = TimeUtils.slice(d, d['jst'], t0, t1)
            dic[symbol] = d1
        plot_atrp(dic, year, timeframe, t0, t1)
    
def plot_atrp(dic, year, timeframe, t0, t1):
    fig, axes = plt.subplots(2, 1, figsize=(18, 10))
    i = 0
    for symbol, data in dic.items():
        jst = data['jst']
        cl = data['close']
        atrp = data['ATRP']
        if symbol == 'NIKKEI' or symbol == 'DOW':
            axes[0].plot(jst, cl, label=symbol, color=cmap(i))
        axes[1].plot(jst, atrp, color=cmap(i), label=symbol, alpha=0.95)
        i += 1
    [ax.grid() for ax in axes]
    [ax.legend() for ax in axes]
    [ax.set_xlim(t0, t1) for ax in axes]
    axes[1].set_ylim(0, 3.0)
    axes[1].set_title('ATRP')
    axes[0].set_title(timeframe)
    
def strength(data, term):
    op = data['open']
    hi = data['high']
    lo = data['low']
    cl = data['close']

    posneg = []
    for o, c in zip(op, cl):
        if c > o:
            posneg.append(1.0)
        elif c < o:
            posneg.append(-1.0)
    n = len(cl)
    strng = np.full(n, 0.0)
    for i in range(term - 1, n):
        d = posneg[i - term + 1: i + 1]
        strng[i] = np.nanmean(d)   
    return strng
        
    
def main6():
    symbol = 'NIKKEI'
    timeframe = 'M5'
    data0 = from_pickle(symbol, timeframe, axiory=True)
    t0 = datetime(2024, 7, 25).astimezone(JST)
    t1 = datetime(2024, 8, 6).astimezone(JST)
    n, data = TimeUtils.slice(data0, data0['jst'], t0, t1)
    streng = strength(data, 60)
    jst = data['jst']
    cl = data['close']
    fig, axes = plt.subplots(2, 1, figsize=(15, 10))
    axes[0].plot(jst, cl)
    axes[1].plot(jst, streng)
    axes[1].set_ylim(-1, 1)    

        
    
if __name__ == '__main__':
    main6()
    

