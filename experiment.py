import os
import shutil
import sys
sys.path.append('../Libraries/trade')

import math
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

from technical import MABAND, MABAND_SIGNAL, EMABREAK
from strategy import Simulation
from time_utils import TimeFilter, TimeUtils
from data_loader import DataLoader
from candle_chart import CandleChart, makeFig, gridFig


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


          
def load():
    year_from = 2020
    month_from = 1
    year_to = 2024
    month_to = 7
    loader = DataLoader()
    timeframe = 'M15'
    symbol = 'NIKKEI'
    return loader.load_data(symbol, timeframe, year_from, month_from, year_to, month_to)

def save(filepath, obj):
    import pickle
    with open(filepath, mode='wb') as f:
        pickle.dump(obj, f)
      

def indicator(data):
    MABAND(data, 15, 40, 200, 15, 15, 25)
     
def plot_bar(ax, data, indices, color):
    jst = data['jst']
    cl = data['close']
    for begin, end in indices:
        x = [jst[begin], jst[end]]
        y = [cl[begin], cl[end]]
        ax.plot(x, y, color=color, linewidth=5.0)    
        
def calc_profit(data, long, short, ls=300):
    jst = data['jst']
    cl = data['close']
    hi = data['high']
    lo = data['low']
    profits = []
    s = 0
    time = []
    acc = []
    for i0, i1 in long:
        p0 = cl[i0]
        profit = cl[i1] - cl[i0]
        s += profit
        acc.append([jst[i1], s])
        profits.append([1, jst[i0], jst[i1], cl[i0], cl[i1],profit])
    for i0, i1 in short:
        p0 = cl[i0]
        profit = cl[i0] - cl[i1]
        s += profit
        acc.append([jst[i1], s])
        profits.append([-1, jst[i0], jst[i1], cl[i0], cl[i1], profit])    
    df = pd.DataFrame(data=profits, columns=['L/S', 'TimeOpen', 'TimeClose', 'PriceOpen', 'PriceClose', 'Profit'])
    df = df.sort_values('TimeOpen')
    acc = sorted(acc, key=lambda x: x[0])
    t = [v[0] for v in acc]
    prf = [v[1] for v in acc]
    return s, df, (t, prf)

def calc_drawdown(curve, term=20):
    n = len(curve)
    dd =[]
    if n < term:
        term = n - 1
    for i in range(term,  n):
        d = curve[i - term + 1: i + 1]
        dd.append(min(d) - max(d))
    return min(dd)
    
def main():
    n, data0 = load()
    save('./nk_m15.pkl', data0)
    fig, axes = gridFig([4, 2, 2], (20, 12))

    indicator(data0)
    t0 = datetime(2020, 3, 9).astimezone(JST)
    t1 = t0 + timedelta(days=2)
    n, data = TimeUtils.slice(data0, data0['time'], t0, t1)
    long_event, short_event = MABAND_SIGNAL(data)
    
    jst = data['jst']
    op = data['open']
    hi = data['high']
    lo = data['low']
    cl = data['close']


    ma_short = data['MA_SHORT']
    band = data['MABAND']
    #profit, df, curve = calc_profit(data, long_event, short_event)
    #print(df)
    #drawdown = calc_drawdown(curve[1])
    #print(df)
    #print('Profit:', profit, drawdown)
    
    axes[0].plot(jst, cl, color='gray')
    axes[0].plot(jst, data['MA_LONG'], color='purple', linewidth=2.0)
    axes[0].plot(jst, data['MA_MID'], color='blue', linewidth=2.0)
    axes[0].plot(jst, data['MA_SHORT'], color='red', linewidth=2.0)
    for i0, i1 in long_event:
        axes[0].scatter(jst[i0], cl[i0], color='green', marker='o', s=200, alpha=0.5)
        axes[0].scatter(jst[i1], cl[i1], color='green', marker='x', s=200, alpha=0.8)

    for i0, i1 in short_event:
        axes[0].scatter(jst[i0], cl[i0], color='red', marker='o', s=200, alpha=0.5)
        axes[0].scatter(jst[i1], cl[i1], color='red', marker='x', s=200, alpha=0.8)
    
    
    axes[1].plot(jst, data['MABAND'])
    axes[1].hlines(0, jst[0], jst[-1])
    
    
    #axes[2].plot(curve[0], curve[1])
    
    [ax.set_xlim(jst[0], jst[-1]) for ax in axes]    
    form = mdates.DateFormatter("%m/%d %H:%M")
    [ax.xaxis.set_major_formatter(form) for ax in axes]

        
        


    
    #df.to_csv('./Profits.csv', index=False)

if __name__ == '__main__':
    main()