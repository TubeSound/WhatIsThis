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
import pandas as pd
from dateutil import tz
JST = tz.gettz('Asia/Tokyo')
UTC = tz.gettz('utc')

from mt5_api import Mt5Api


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


def main():
    symbol = 'NIKKEI'
    t0 = datetime(2024, 8, 5, 9, 0).astimezone(JST)
    t1 = datetime(2024, 8, 5, 9, 10).astimezone(JST)
    mt5 = Mt5Api()
    df = mt5.get_ticks(symbol, t0, t1)

    df.to_csv('./debug/ticks.csv', index=False)


    time = list(df['jst'])
    bid = list(df['bid'])
    #print(time[:10], bid[:10])
    
    tohlc, arrays = sample(time, bid, ohlc, 10)
    df = pd.DataFrame(data=tohlc, columns=['jst', 'open', 'high', 'low', 'close'])
    df.to_csv('./debug/tohlc.csv', index=False)

    #fig, ax = plt.subplots(1, 1, figsize=(20, 5))
    #ax.plot(time, bid, color='blue')
    #plt.grid(color='b', linestyle=':', linewidth=0.3)


def test():
    
    t = datetime(2024, 8, 10, 12, 15, 16)
    print(t, '->', round_time(t, 60 * 15))
    
if __name__ == '__main__':
    main()
    

