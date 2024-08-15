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




def from_pickle(symbol, timeframe):
    import pickle
    symbol = symbol.lower()
    timeframe = timeframe.upper()
    filepath = f'./data/BacktestMarket/BM_{symbol}_{timeframe}.pkl'
    with open(filepath, 'rb') as f:
        data0 = pickle.load(f)
    return data0

    
    
def main(): 
    symbol = 'DOW'
    timeframe = 'H1'
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
    
    
def detect(data, threshold):
    atrp = data['ATRP']
    sig = [0 for _ in range(len(atrp))]
    for i in range(len(atrp)):
        if is_nan(atrp[i]):
            continue
        if atrp[i] >= threshold:
            sig[i] = 1 
    return sig
        
def main2(): 
    symbol = 'NIKKEI'
    timeframe = 'H1'
    number = 1
    data = from_pickle(symbol, timeframe)
    ATRP(data, 24, ma_window=20)
    threshold = 1.0
    signal = detect(data, threshold)
    
    jst = data['jst']
    atrp = data['ATRP']
    cl = data['close']

    fig, axes = plt.subplots(2, 1, figsize=(20, 10))
    axes[0].plot(jst, cl, color='blue')
    axes[1].scatter(jst, atrp, color='green', alpha=0.1, s=1)
    [ax.legend() for ax in axes]
    locator = mdates.AutoDateLocator(minticks=12, maxticks=20)
    [ax.xaxis.set_major_locator(locator) for ax in axes]
    axes[0].set_title(f'{symbol} {timeframe} ATRP threshold: {threshold}')
    n = len(atrp)
    for i in range(len(signal)):
        if signal[i] == 1:
            axes[1].scatter(jst[i], atrp[i], color='red', s=10, alpha=0.2)
            end = n - 1 if (i + 480) >= n else i + 480
            axes[1].hlines(0, jst[i], jst[end], color='red', linewidth=5.0)
    [ax.grid() for ax in axes]

    
    

    
    
if __name__ == '__main__':
    main()
    

