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
import pandas as pd
from dateutil import tz
JST = tz.gettz('Asia/Tokyo')
UTC = tz.gettz('utc')

from technical import MABAND, MABAND_SIGNAL, EMABREAK
from strategy import Simulation
from time_utils import TimeFilter, TimeUtils
from data_loader import DataLoader
from candle_chart import CandleChart, makeFig, gridFig



          
def load():
    year_from = 2020
    month_from = 1
    year_to = 20212
    month_to = 3
    loader = DataLoader()
    timeframe = 'M15'
    symbol = 'DOW'
    return loader.load_data(symbol, timeframe, year_from, month_from, year_to, month_to)

def save(filepath, obj):
    import pickle
    with open(filepath, mode='wb') as f:
        pickle.dump(obj, f)
      

def indicator(data):
    MABAND(data, 7, 15, 15, 15, 25)
     
def plot_line(chart, data, indices, color):
    jst = data['jst']
    cl = data['close']
    for begin, end in indices:
        x = [jst[begin], jst[end]]
        y = [cl[begin], cl[end]]
        chart.plotLine(x, y, color=color, linewidth=5.0)    
        
def calc_profit(data, long, short, ls=300):
    jst = data['jst']
    cl = data['close']
    hi = data['high']
    lo = data['low']
    profits = []
    s = 0
    for i0, i1 in long:
        p0 = cl[i0]
        cut = False
        for i in range(i0 + 1, i1 + 1):
            profit = lo[i] - p0
            if profit < -ls:
                s += profit
                
                profits.append([1, jst[i0], jst[i], p0, cl[i], profit])
                cut = True
        if not cut:
            profit = cl[i1] - cl[i0]
            s += profit
            profits.append([1, jst[i0], jst[i1], cl[i0], cl[i1],profit])
    for i0, i1 in short:
        p0 = cl[i0]
        cut = False
        for i in range(i0 + 1, i1 + 1):
            profit = p0 - hi[i]
            if profit < -ls:
                s += profit
                profits.append([-1, jst[i0], jst[i], p0, cl[i], profit])
                cut = True        
        if not cut:
            profit = cl[i0] - cl[i1]
            s += profit
            profits.append([-1, jst[i0], jst[i1], cl[i0], cl[i1], profit])    
    df = pd.DataFrame(data=profits, columns=['L/S', 'TimeOpen', 'TimeClose', 'PriceOpen', 'PriceClose', 'Profit'])
    return s, df        
    
def main():
    n, data0 = load()
    save('./dw_m15_2020.pkl', data0)
    fig, axes = gridFig([4, 2, 2], (20, 12))

    indicator(data0)
    t0 = datetime(2020, 3, 1).astimezone(JST)
    t1 = t0 + timedelta(days=5)
    n, data = TimeUtils.slice(data0, data0['time'], t0, t1)
    up_event, down_event = MABAND_SIGNAL(data)
    
    jst = data['jst']
    op = data['open']
    hi = data['high']
    lo = data['low']
    cl = data['close']
    chart1 = CandleChart(fig, axes[0])
    chart1.drawCandle(jst, op, hi, lo, cl)
    chart1.drawLine(jst, data['MA_LONG'], color='blue', linewidth=2.0)
    chart1.drawLine(jst, data['MA_SHORT'], color='red', linewidth=2.0)
    ma_short = data['MA_SHORT']
    band = data['MABAND']
    
    chart3 = CandleChart(fig, axes[2])
    chart3.drawLine(jst, data['ADX'])
    
    chart2 = CandleChart(fig, axes[1])
    chart2.xlimit([jst[0], jst[-1]])
    for i in range(len(band)):
        x = [jst[i], jst[i]]
        y = [0, band[i]]
        if band[i] > 0:
            color = 'green'
        else:
            color = 'red'
        chart2.plotLine(x, y, color=color)
        
        
    for i0, i1 in up_event:
        chart2.drawMarker(jst[i0], band[i0], color='red', marker='o', markersize=10)
        chart2.drawMarker(jst[i1], band[i1], color='red', marker='x', markersize=10)

    for i0, i1 in down_event:
        chart2.drawMarker(jst[i0], band[i0], color='green', marker='o', markersize=10)
        chart2.drawMarker(jst[i1], band[i1], color='green', marker='x', markersize=10)

    profit, df = calc_profit(data, up_event, down_event)
    print(df)
    print('Profit:', profit)
    #df.to_csv('./Profits.csv', index=False)

if __name__ == '__main__':
    main()
