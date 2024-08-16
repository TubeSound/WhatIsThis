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
import numpy as np

import pandas as pd
from dateutil import tz
JST = tz.gettz('Asia/Tokyo')
UTC = tz.gettz('utc')
import sched

from mt5_trade import Mt5Trade, Columns, PositionInfo
from utils import TimeUtils
from technical import sma, ATRP, is_nan


SYMBOLS = ['NIKKEI', 'DOW']
TIMEFRAME = 'H1'

scheduler = sched.scheduler()

    
class CrashAlert:
    def __init__(self, symbol, timeframe):
        self.symbol = symbol
        self.timeframe = timeframe
        mt5 = Mt5Trade(symbol)
        self.mt5 = mt5
    
    def detect_crash_signal(self, data, threshold=1.0):
        atrp = data['ATRP']
        n = len(atrp)
        sig = [0 for _ in range(n)]
        for i in range(n):
            if is_nan(atrp[i]):
                continue
            if atrp[i] >= threshold:
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
            if atrp[i] > maxv * 1.1 and atrp[i] > 0.5:
                break_points.append(i)
        return sig, xup, break_points

    def detect_crash(self):
        self.mt5.get_rates(self.timeframe, 200)
    
    
    
        return False


    def detect(self):
        if self.detect_crash():
            self.alert()
    
    
    def alert(self):
        pass
    
def main(): 
    alerts = []
    for symbol in SYMBOLS:
        alert = CrashAlert(symbol, TIMEFRAME)
        alerts.append(alert)
    
    while True:
        for i, alert in enumerate(alerts):
            scheduler.enter(10, i + 1, alert.detect)
            
        scheduler.run()

    
if __name__ == '__main__':
    main()
    

