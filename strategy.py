import numpy as np 
import pandas as pd
import math
import statistics as stat
from common import Indicators, Signal, Columns, UP, DOWN, HIGH, LOW, HOLD
from datetime import datetime, timedelta
from dateutil import tz

JST = tz.gettz('Asia/Tokyo')
UTC = tz.gettz('utc') 

    
def nans(length):
    return [np.nan for _ in range(length)]

def full(value, length):
    return [value for _ in range(length)]

class Position:
    def __init__(self, trade_param, signal: Signal, index: int, time: datetime, price, volume):
        self.id = None
        self.sl = trade_param['sl']
        try:
            self.target = trade_param['target']
            self.trail_stop = trade_param['trail_stop']
        except:
            self.target = 0
            self.trail_stop = 0
        self.signal = signal
        self.entry_index = index
        self.entry_time = time
        self.entry_price = price
        self.volume = volume
        self.exit_index = None
        self.exit_time = None       
        self.exit_price = None
        self.profit = None
        self.fired = False
        self.profit_max = None
        self.closed = False
        self.losscutted = False
        self.trail_stopped = False
        self.timelimit = False
        
    # return  True: Closed,  False: Not Closed
    def update(self, index, time, o, h, l, c):
        if self.sl == 0:
            return False
        
        # check stoploss
        if self.signal == Signal.LONG:
            profit = l - self.entry_price
            if profit <= -1 * self.sl:
                 self.exit(index, time, l)
                 self.losscutted = True
                 return True
            profit = c - self.entry_price
        else:
            profit = self.entry_price - h
            if profit <= -1 * self.sl:
                self.exit(index, time, h)
                self.losscutted = True
                return True
            profit = c - self.entry_price            
        
        if self.target == 0:
            return False
        
        if self.fired:
            if profit > self.profit_max:
                self.profit_max = profit
            else:
                if self.profit_max - profit < self.trail_stop:
                    self.exit(index, time, c)
                    self.trail_stopped = True         
                    return True
        else:
            if profit >= self.target:
                self.profit_max = profit
                self.fired = True        
        return False
    
    def exit(self, index, time, price):
        self.exit_index = index
        self.exit_time = time
        self.exit_price = price
        self.closed = True
        self.profit = price - self.entry_price
        if self.signal == Signal.SHORT:
            self.profit *= -1

class Positions:
    
    def __init__(self):
        self.positions = []
        self.closed_positions = []
        self.current_id = 0
                
    def num(self):
        return len(self.positions) 
    
    def total_volume(self):
        v = 0
        for position in self.positions:
            v += position.volume
        return v
    
    def add(self, position: Position):
        position.id = self.current_id
        self.positions.append(position)
        self.current_id += 1
        
    def update(self, index, time, op, hl, lo, cl):
        closed = []
        for position in self.positions:
            if position.update(index, time, op, hl, lo, cl):
                closed.append(position.id)
        for id in closed:
            for i, position in enumerate(self.positions):
                if position.id == id:
                    pos = self.positions.pop(i)        
                    self.closed_positions.append(pos)
                    
    def exit_all(self, index, time, price):
        for position in self.positions:
            position.exit(index, time, price)            
        self.closed_positions += self.positions
        self.positions = []
    
    def summary(self):
        s = 0
        win = 0
        profits = []
        acc = []
        time = []
        for position in self.closed_positions:
            profits.append(position.profit)
            s += position.profit
            time.append(position.entry_time)
            acc.append(s)
            if position.profit > 0:
                win += 1
        if self.num() > 0:
            win_rate = float(win) / float(self.num())
        else:
            win_rate = 0
        return s, (time, acc), win_rate
    
    def to_dataFrame(self):
        data = []
        for i, position in enumerate(self.closed_positions):
            d = [i, position.signal, position.entry_index, str(position.entry_time), position.entry_price]
            d += [position.exit_index, str(position.exit_time), position.exit_price, position.profit]
            d += [position.closed, position.losscutted,  position.trail_stopped, position.doten, position.timelimit]
            data.append(d)
        columns = ['No', 'signal', 'entry_index', 'entry_time', 'entry_price']
        columns += ['exit_index', 'exit_time', 'exit_price', 'profit']
        columns += ['closed', 'losscuted', 'trail_stopped', 'doten', 'timelimit']
        df = pd.DataFrame(data=data, columns=columns)
        return df 
    
class Simulation:
    def __init__(self, trade_param):
        self.trade_param = trade_param
        self.volume = trade_param['volume']
        self.position_num_max = trade_param['position_num_max']
        self.positions = Positions()
        
    def run_doten(self, data):
        self.data = data
        time = data[Columns.TIME]
        op = data[Columns.OPEN]
        hi = data[Columns.HIGH]
        lo = data[Columns.LOW]
        cl = data[Columns.CLOSE]
        signal = data[Indicators.VWAP_RATE_SIGNAL]
        n = len(time)
        state = None
        for i in range(1, n):
            if i == n - 1:
                self.positions.exit_all(i, time[i], cl[i])
                break
            
            self.positions.update(i, time[i], op[i], hi[i], lo[i], cl[i])
            sig = signal[i]
            if sig == Signal.LONG:
                if state == Signal.SHORT:
                    self.doten(sig, i, time[i], op[i], hi[i], lo[i], cl[i])
                else:
                    self.entry(sig, i, time[i], op[i], hi[i], lo[i], cl[i])
            elif sig == Signal.SHORT:               
                if state == Signal.LONG:
                    self.doten(sig, i, time[i], op[i], hi[i], lo[i], cl[i])
                else:
                    self.entry(sig, i, time[i], op[i], hi[i], lo[i], cl[i])
            return self.positions.to_dataFrame()
                    
    def doten(self, signal, index, time, op, hi, lo, cl):
        self.positions.exit_all(index, time, cl)
        self.entry(signal, index, time, op, hi, lo, cl)
        pass
    
    def entry(self, signal, index, time, op, hl, lo, cl):
        if self.positions.total_num() < self.position_num_max:
            position = Position(self.trade_param, signal, index, time, cl, self.volume)
            self.positions.add(position)
        
    
        
                        
                        
            
        
        
        