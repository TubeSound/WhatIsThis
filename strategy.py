import numpy as np 
import pandas as pd
import math
import statistics as stat
from common import Indicators, Signal, Columns, UP, DOWN, HIGH, LOW, HOLD
from datetime import datetime, timedelta
from dateutil import tz
JST = tz.gettz('Asia/Tokyo')
UTC = tz.gettz('utc') 
from time_utils import TimeFilter
    
def nans(length):
    return [np.nan for _ in range(length)]

def full(value, length):
    return [value for _ in range(length)]

class Position:
    def __init__(self, trade_param, signal: Signal, index: int, time: datetime, price, volume, sl):
        self.id = None
        self.sl = sl
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
        self.doten = False
        
    # return  True: Closed,  False: Not Closed
    def update(self, index, time, o, h, l, c, timefilter: TimeFilter):
        if self.sl == 0:
            return False
        
        if timefilter is not None:
            if timefilter.over(time):
                self.exit(index, time, c, timelimit=True)
                return True
        
        # check stoploss
        if self.signal == Signal.LONG:
            profit = l - self.entry_price
            if profit <= -1 * self.sl:
                 self.exit(index, time, self.entry_price - self.sl)
                 self.losscutted = True
                 return True
            profit = c - self.entry_price
        else:
            profit = self.entry_price - h
            if profit <= -1 * self.sl:
                self.exit(index, time, self.entry_price + self.sl)
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
    
    def exit(self, index, time, price, doten=False, timelimit=False):
        self.exit_index = index
        self.exit_time = time
        self.exit_price = price
        self.closed = True
        self.profit = price - self.entry_price
        if self.signal == Signal.SHORT:
            self.profit *= -1
        self.doten=doten
        self.timelimit = timelimit

class Positions:
    
    def __init__(self, timefilter: TimeFilter):
        self.timefilter = timefilter
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
            if position.update(index, time, op, hl, lo, cl, self.timefilter):
                closed.append(position.id)
        for id in closed:
            for i, position in enumerate(self.positions):
                if position.id == id:
                    pos = self.positions.pop(i)        
                    self.closed_positions.append(pos)
                    
    def exit_all(self, index, time, price, doten=False):
        for position in self.positions:
            position.exit(index, time, price, doten=doten)            
        self.closed_positions += self.positions
        self.positions = []
    
    def exit_all_signal(self, signal, index, time, price, doten=False):
        pos = []
        for i, position in enumerate(self.positions):
            if position.signal == signal:
                position.exit(index, time, price, doten=doten)         
                pos.append(i)     
        for i in pos:           
            position  = self.positions.pop(i)
            self.closed_positions.append(position)
        
        
    def summary(self):
        profit_sum = 0
        win = 0
        profits = []
        tacc = []
        acc = []
        time = []
        for position in self.closed_positions:
            profits.append(position.profit)
            prf = position.profit
            if prf is not None:
                if not np.isnan(prf):
                    profit_sum += prf
            time.append(position.entry_time)
            tacc.append(position.exit_time)
            acc.append(profit_sum)
            if position.profit > 0:
                win += 1
        n = len(self.closed_positions)
        if n > 0:
            win_rate = float(win) / float(n)
        else:
            win_rate = 0
        return (n, profit_sum, win_rate), (tacc, acc)
    
    def to_dataFrame(self, strategy: str):
        def bool2str(v):
            s = 'true' if v else 'false'
            return s
            
        data = []
        for i, position in enumerate(self.closed_positions):
            d = [strategy, position.signal, position.entry_index, str(position.entry_time), position.entry_price]
            d += [position.exit_index, str(position.exit_time), position.exit_price, position.profit]
            d += [bool2str(position.closed), bool2str(position.losscutted),  bool2str(position.trail_stopped)]
            d += [bool2str(position.doten), bool2str(position.timelimit)]
            data.append(d)
        columns = ['Mode', 'signal', 'entry_index', 'entry_time', 'entry_price']
        columns += ['exit_index', 'exit_time', 'exit_price', 'profit']
        columns += ['closed', 'losscuted', 'trail_stopped', 'doten', 'timelimit']
        df = pd.DataFrame(data=data, columns=columns)
        return df 
    
class Simulation:
    SL_NONE = 0
    SL_FIX = 1
    SL_HIGH_LOW = 2
    
    def __init__(self, data, trade_param:dict):
        self.data = data
        self.trade_param = trade_param
        self.strategy = self.trade_param['strategy'].upper()
        self.volume = trade_param['volume']
        self.position_num_max = trade_param['position_max']
        try :
            begin_hour = self.trade_param['begin_hour']
            begin_minute = self.trade_param['begin_minute']
            hours = self.trade_param['hours']
            if hours == 0:
                self.timefilter = None
            else:
                self.timefilter = TimeFilter(JST, begin_hour, begin_minute, hours)
        except:
            self.timefilter = None
                
        self.positions = Positions(self.timefilter)
        
    def calc_sl(self, index, signal, price):
        method = self.trade_param['sl']['method']
        value = self.trade_param['sl']['value']
        if method == self.SL_NONE:
            return 0
        elif method == self.SL_FIX:
            # fix
            return value
        elif method == self.SL_HIGH_LOW:
            # high_low
            if signal == Signal.LONG:
                low = self.data['low']
                d = low[index - value: index + 1]
                return price - np.nanmin(d)
            else:
                high  = self.data['high']
                d = high[index - value: index + 1]
                return np.nanmax(d) - price
        else:
            return 0
            
        
    def run(self, data, long_signal, short_signal):
        self.data = data
        time = data[Columns.JST]
        op =data[Columns.OPEN]
        hi = data[Columns.HIGH]
        lo = data[Columns.LOW]
        cl = data[Columns.CLOSE]
        n = len(time)
        long = data[long_signal]
        short = data[short_signal]
        for i in range(1, n):
            t = time[i]
            if i == n - 1:
                self.positions.exit_all(i, time[i], cl[i])
                break
            self.positions.update(i, time[i], op[i], hi[i], lo[i], cl[i])
            if long[i] == 1:
                sl = self.calc_sl(data, i, Signal.LONG, cl[i])
                self.entry(Signal.LONG, i, time[i], cl[i], sl)
            elif long[i] == -1:
                self.positions.exit_all_signal(Signal.LONG, i, time[i], cl[i])
            if short[i] == 1:
                sl = self.calc_sl(data, i, Signal.SHORT, cl[i])
                self.entry(Signal.SHORT, i, time[i], cl[i], sl)
            elif short[i] == -1:
                self.positions.exit_all_signal(Signal.SHORT, i, time[i], cl[i])
        summary, profit_curve = self.positions.summary()
        return self.positions.to_dataFrame(self.strategy), summary, profit_curve
        
        
    
    def run_doten(self, signal_column):        
        data = self.data
        signal = data[signal_column]
        time = data[Columns.JST]
        op = data[Columns.OPEN]
        hi = data[Columns.HIGH]
        lo = data[Columns.LOW]
        cl = data[Columns.CLOSE]
        n = len(time)
        state = None
        for i in range(1, n):
            t = time[i]
            if i == n - 1:
                self.positions.exit_all(i, time[i], cl[i])
                break
            self.positions.update(i, time[i], op[i], hi[i], lo[i], cl[i])
            sig = signal[i]
            if sig == Signal.LONG:
                if state == Signal.SHORT:
                    self.doten(sig, i, time[i], cl[i])
                else:
                    self.entry(sig, i, time[i], cl[i])
                state = Signal.LONG
            elif sig == Signal.SHORT:               
                if state == Signal.LONG:
                    self.doten(sig, i, time[i], cl[i])
                else:
                    self.entry(sig, i, time[i], cl[i])
                state = Signal.SHORT
        summary, profit_curve = self.positions.summary()
        return self.positions.to_dataFrame(self.strategy), summary, profit_curve
    
    def run_doten2(self, time, entry_signal, exit_signal, op, hi, lo, cl):
        n = len(time)
        state = None
        for i in range(1, n):
            if i == n - 1:
                self.positions.exit_all(i, time[i], cl[i])
                break
            self.positions.update(i, time[i], op[i], hi[i], lo[i], cl[i])
    
            if state is None:
                if entry_signal[i] == Signal.LONG or entry_signal[i] == Signal.SHORT:
                    self.entry(entry_signal[i], i, time[i], cl[i])
            elif state == Signal.LONG:
                if exit_signal[i] == Signal.SHORT:
                    self.doten(exit_signal[i], i, time[i], op[i], hi[i], lo[i], cl[i])
            elif state == Signal.SHORT:
                 if exit_signal[i] == Signal.LONG:
                    self.doten(exit_signal[i], i, time[i], cl[i])
        return self.positions.to_dataFrame()
                    
    def doten(self, signal, index, time, price):
        self.positions.exit_all(index, time, price, doten=True)
        self.entry(signal, index, time, price)
        pass
    
    def entry(self, signal, index, time, price):
        sl = self.calc_sl(index, signal, price)
        if self.timefilter is not None:
            if self.timefilter.on(time) == False:
                return
        
        if self.positions.num() < self.position_num_max:
            position = Position(self.trade_param, signal, index, time, price, self.volume, sl)
            self.positions.add(position)
        