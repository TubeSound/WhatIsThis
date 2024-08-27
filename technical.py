import numpy as np 
import math
import statistics as stat
from scipy.stats import rankdata
from sklearn.cluster import KMeans 
from common import Indicators, Signal, Columns, UP, DOWN, HIGH, LOW, HOLD
from datetime import datetime, timedelta
from dateutil import tz

JST = tz.gettz('Asia/Tokyo')
UTC = tz.gettz('utc') 

    
def nans(length):
    return [np.nan for _ in range(length)]

def full(length, value):
    return [value for _ in range(length)]

def is_nan(value):
    if value is None:
        return True
    return np.isnan(value)

def is_nans(values):
    if len(values) == 0:
        return True
    for value in values:
        if is_nan(value):
            return True
    return False

def sma(vector, window):
    window = int(window)
    n = len(vector)
    out = full(n, np.nan)
    ivalid = window- 1
    if ivalid < 0:
        return out
    for i in range(ivalid, n):
        d = vector[i - window + 1: i + 1]
        out[i] = stat.mean(d)
    return out

def ema(vector, window):
    window = int(window)
    weights = np.exp(np.linspace(-1., 0., window))
    weights /= weights.sum()
    n = len(vector)
    out = full(n, np.nan)
    ivalid = window- 1
    if ivalid < 0:
        return out
    for i in range(ivalid, n):
        d = vector[i - window + 1: i + 1]
        out[i] = np.sum(d * weights)
    return out

def slopes(signal: list, window: int, minutes: int, tolerance=0.0):
    n = len(signal)
    out = full(n, np.nan)
    for i in range(window - 1, n):
        d = signal[i - window + 1: i + 1]
        if np.min(d) == 0:
            continue        
        m, offset = np.polyfit(range(window), d, 1)
        if abs(m) > tolerance:
            out[i] = m / np.mean(d[:3]) * 100.0 / (window * minutes)  * 60 * 24
    return out


def slope(vector):
    n = len(vector)
    d = np.array(vector)
    m, offset = np.polyfit(range(n), d, 1)
    return m, offset    
    
def subtract(signal1: list, signal2:list):
    n = len(signal1)
    if len(signal2) != n:
        raise Exception('dont match list size')
    out = nans(n)
    for i in range(n):
        if is_nan(signal1[i]) or is_nan(signal2[i]):
            continue
        out[i] = signal1[i] - signal2[i]
    return out


def linearity(signal: list, window: int):
    n = len(signal)
    out = nans(n)
    for i in range(window, n):
        data = signal[i - window + 1: i + 1]
        if is_nans(data):
            continue
        m, offset = np.polyfit(range(window), data, 1)
        e = 0
        for j, d in enumerate(data):
            estimate = m * j + offset
            e += pow(estimate - d, 2)
        error = np.sqrt(e) / window / data[0] * 100.0
        if error == 0:
            out[i] = 100.0
        else:
            out[i] = 1 / error
    return out
            
            
def true_range(high, low, cl):
    n = len(high)
    out = nans(n)
    ivalid = 1
    for i in range(ivalid, n):
        d = [ high[i] - low[i],
              abs(high[i] - cl[i - 1]),
              abs(low[i] - cl[i - 1])]
        out[i] = max(d)
    return out


def rci(vector, window):
    n = len(vector)
    out = nans(n)
    for i in range(window - 1, n):
        d = vector[i - window + 1: i + 1]
        if is_nans(d):
            continue
        r = rankdata(d, method='ordinal')
        s = 0
        for j in range(window):
            x = window - j
            y = window - r[j] + 1
            s += pow(x - y, 2)
        out[i] = (1 - 6 * s / ((pow(window, 3) - window))) * 100
    return out

def roi(vector:list):
    n = len(vector)
    out = nans(n)
    for i in range(1, n):
        if is_nan(vector[i - 1]) or is_nan(vector[i]):
            continue
        if vector[i - 1] == 0:
            out[i] = 0.0
        else:
            out[i] = (vector[i] - vector[i - 1]) / vector[i - 1] * 100.0
    return out

def pivot(up: list, down: list, threshold: float=7 , left_length: int=5, right_length: int=5):
    n = len(up)
    state = full(n, 0)
    for i in range(left_length + right_length, n):
        if up[i] + down[i] < 90:
            continue
        left = up[i - left_length - right_length: i - right_length]
        right = up[i - right_length: i + 1]        
        range_left = max(left) - min(left)
        if range_left < 2:
            if np.mean(left) < 20:
                if (np.mean(right) - np.mean(left)) > threshold:
                    state[i] = HIGH
            if np.mean(left) > 80:
                if (np.mean(right) - np.mean(left)) < -threshold:
                    state[i] = LOW
    return state

def cross_value(vector: list, value):
    n = len(vector)
    up = nans(n)
    down = nans(n)
    cross = full(n, HOLD)
    for i in range(1, n):
        if vector[i - 1] < value and vector[i] >= value:
            up[i] = 1
            cross[i] = UP
        elif vector[i - 1] > value and vector[i] <= value:
            down[i] = 1
            cross[i] = DOWN
    return up, down, cross

def median(vector, window):
    n = len(vector)
    out = nans(n)
    for i in range (window, n):
        d = vector[i - window: i + 1]
        if is_nans(d):
            continue
        med = np.median(d)
        out[i] = med
    return out
        
def band_position(data, lower, center, upper):
    n = len(data)
    pos = full(n, 0)
    for i in range(n):
        if is_nan(data[i]):
            continue 
        if data[i] > upper[i]:
            pos[i] = 2
        else:
            if data[i] > center[i]:
                pos[i] = 1
        if data[i] < lower[i]:
            pos[i] = -2
        else:
            if data[i] < center[i]:
                pos[i] = -1
    return pos

def probability(position, states, window):
    n = len(position)
    prob = full(n, 0)
    for i in range(window - 1, n):
        s = 0
        for j in range(i - window + 1, i + 1):
            if is_nan(position[j]):
                continue
            for st in states:
                if position[j] == st:
                    s += 1
                    break
        prob[i] = float(s) / float(window) * 100.0 
    return prob      

def cross(long, short):
    p0, p1 = long
    q0, q1 = short
    
    if q0 == q1 :
        return 0
    if p0 == p1:
        return 0
    if p0 >= q0 and p1 <= q1:
        return 1
    elif p0 <= q0 and p1 >= q1:
        return -1
    return 0       


def wakeup(long, short, range_signal, is_up):
    n = len(long)
    direction = 1 if is_up else -1
    x = []
    for i in range(1, n):
        if cross(long[ i - 1: i +1], short[i - 1: i +1]) == direction:
            x.append(i)
    x.append(n - 1)
    trend = full(n, 0)   
    for i in range(len(x) - 1):
        x0 = x[i]
        x1 = x[i + 1]
        for j in range(x0 + 3, x1):
            before = j - 3
            l0 = long[x0]
            s0 = short[x0]
            l1 = long[before]
            s1 = short[before]
            d1 = s1 - s0
            l2 = long[j]            
            s2 = short[j]
            d2 = s2 - s1
            w2 = range_signal[j]
            if is_up:
                if d1 > 0 and d2 > 0 and (s2 - l2) > w2:
                    trend[j] = 1
            else:
                if d1 < 0 and d2 < 0 and (l2 - s2) < -w2:
                    trend[j] = -1 
    return trend, x

def wakeup2(long, short, range_signal, is_up):
    n = len(long)
    trend = full(n, 0)   
    for j in range(5, n):
        before = j - 5
        l1 = long[before]
        s1 = short[before]
        l2 = long[j]            
        s2 = short[j]
        d2 = s2 - s1
        w2 = range_signal[j]
        if is_up:
            if d2 > 0 and (s2 - l2) > w2:
                trend[j] = 1
        else:
            if d2 < 0 and (l2 - s2) < -w2:
                trend[j] = -1 
    return trend

def wakeup3(long, mid, short, width, is_up, is_slow=True):
    n = len(long)
    direction = 1 if is_up else -1
    x = []
    for i in range(1, n):
        if is_slow:
            if cross(long[ i - 1: i +1], mid[i - 1: i +1]) == direction:
                x.append(i)
        else:
            if cross(mid[ i - 1: i +1], short[i - 1: i +1]) == direction:
                x.append(i)
            
    x.append(n - 1)
    trend = full(n, 0)   
    for i in range(len(x) - 1):
        x0 = x[i]
        x1 = x[i + 1]
        for j in range(x0 + 5, x1):
            half = int((j - x0 ) / 2 + x0)
            l0 = long[half]
            m0 = mid[half]
            s0 = short[half]
            l1 = long[j]            
            m1 = mid[j]
            s1 = short[j]
            w1 = width[j]
            if is_up:
                if l1 > l0 and m1 > m0 and s1 > s0 and (s1 - m1) > w1 and (m1 - l1) > w1:
                    trend[j] = 1
            else:
                if l1 < l0 and m1 < m0 and s1 < s0 and (m1 - s1) > w1 and (l1 - m1) > w1:
                    trend[j] = -1 
    return trend, x
    
def ascend(vector, range_signal, count=3):
    n = len(vector)
    asc = full(n, 0)
    for i in range(count, n):
        if vector[i] > range_signal[i] and vector[i - count] < vector[i]:
            asc[i] = 1
        if vector[i] < - range_signal[i] and vector[i - count] > vector[i]:
            asc[i] = -1
    return asc
        
def adx_filter(signal, adx, threshold):
    n = len(signal)
    trend = full(n, 0)
    for i in range(3, n):
        if adx[i] > threshold and adx[i] > adx[i - 3]:
            if signal[i] == 1:
                trend[i] = 1
            elif signal[i] == -1:
                trend[i] = -1
    return trend    

def MA(dic: dict, long_term, short_term):
    ma_long = sma(dic[Columns.CLOSE], long_term)
    dic[Indicators.MA_LONG] = ma_long
    ma_short = sma(dic[Columns.CLOSE], short_term)
    dic[Indicators.MA_SHORT] = ma_short
        
def MABAND( dic: dict, short: int, mid: int, long: int, di_window: int, adx_window: int, adx_threshold: float):
    cl = dic[Columns.CLOSE]
    hi = dic[Columns.HIGH]
    lo  = dic[Columns.LOW]
    
    ema_short = ema(cl, short)
    sma_mid = sma(cl, mid)
    sma_long = sma(cl, long)
    band = ema_short - sma_mid
    adx, _, _ = ADX(hi, lo, cl, di_window, adx_window)
    atr = calc_atr(dic, 4 * 24)

    dic[Indicators.MA_SHORT] = ema_short
    dic[Indicators.MA_MID] = sma_mid
    dic[Indicators.MA_LONG] = sma_long
    dic[Indicators.ADX] = adx
    dic[Indicators.ATR] = atr
    dic[Indicators.MABAND] = band

def MABAND_SIGNAL(dic: dict):
    atr = dic[Indicators.ATR]
    band = dic[Indicators.MABAND]
    ma = dic[Indicators.MA_MID]
    threshold = math.sqrt(atr[-1]) * 5
    print('ATR', atr[-1], threshold)
    
    up, down, signal = detect_cross(band, atr / 2)   
    print(up)
    print(down)
    n = len(band)
    delay = 1
    rate = 0.8
    long = full(n, 0)
    
    """
    for x0, x1 in up:
        long[x0 + delay] = 1
        for x in range(x0 + 1, x1):
            d = ma[x0 + 1: x + 1]
            if max(d) == ma[x]:
                long[x] = -1
            else:
                if (ma[x + 1] - ma[x0]) / (max(d) - ma[x0]) < rate:S
                    long[x] = -1                
    short = full(n, 0)
    for x0, x1 in down:
        short[x0 + delay] = 1
        for x in range(x0 + 1, x1):
            d = ma[x0 + 1: x + 1]
            if min(d) == ma[x0]:
                short[x] = -1 
            else:
                if (ma[x + 1] - ma[x0]) / (min(d) - ma[x0]) < rate:
                    short[x] = -1   
    dic[Indicators.MABAND_LONG] = long
    dic[Indicators.MABAND_SHORT] = short
    """
    return up, down


def detect_cross(band, range_signal):
    n = len(band)

    sig = full(n, 0)
    for i in range(1, n):
        threshold = range_signal[i]
        p0 = band[i - 1]
        p1 = band[i]
        if p0 <= threshold and p1 > threshold:
            sig[i] = 1
        elif p0 >= -threshold and p1 < -threshold:
            sig[i] = -1
            
    up = []
    down = []
    signal = full(n, 0)
    state = 0
    begin = None
    for i, s in enumerate(sig):
        if state == 0:
            state = s
            signal[i] = s
            begin = i
        elif state == 1:
            if s == -1:
                up.append([begin, i - 1])
                begin = i
                state = s
        elif state == -1:
            if s == 1:
                down.append([begin, i - 1])
                begin = i
                state = s
    return up, down, signal

    
def EMABREAK( dic: dict, short: int, long: int, di_window: int, adx_window: int, adx_threshold: float):
    cl = dic[Columns.CLOSE]
    hi = dic[Columns.HIGH]
    lo  = dic[Columns.LOW]
    
    ema_short = ema(cl, short)
    sma_long = sma(cl, long)

    
    dic[Indicators.MA_SHORT] = ema_short
    dic[Indicators.MA_LONG] = sma_long
    adx, _, _ = ADX(hi, lo, cl, di_window, adx_window)
    dic['ADX'] = adx
    
    n = len(cl)
    trend = full(n, 0)
    for i in range(3, n):
        if adx[i] > adx_threshold and adx[i] > adx[i - 3]:
            if ema_short[i] > sma_long[i] and lo[i] > ema_short[i]:
                trend[i] = 1 
            if ema_short[i]  < sma_long[i] and lo[i] < ema_short[i]:
                trend[i] = -1 
    
    up, down, up_event, down_event = detect_signal(trend)
    dic[Indicators.EMABREAK] = trend
    dic[Indicators.EMABREAK_LONG] = up
    dic[Indicators.EMABREAK_SHORT] = down
    

def detect_signal(data):
    up_event = []
    down_event = []
    n = len(data)    
    up = full(n, 0)
    down = full(n, 0)
    active = 0
    for i in range(n):
        if active == 1:
            if data[i] <= 0:
                up_event.append([begin, i])
                up[i] = -1
                active = 0
        elif active == -1:
            if data[i] >= 0:
                down[i] = -1
                down_event.append([begin, i])
                active = 0
        else:
            if data[i] == 1:
                up[i] = 1
                begin = i
                active = 1
            elif data[i] == -1:
                down[i] = 1
                begin = i
                active = -1
    return up, down, up_event, down_event
                
                
def calc_atr(dic, window):
    hi = dic[Columns.HIGH]
    lo = dic[Columns.LOW]
    cl = dic[Columns.CLOSE]
    tr = true_range(hi, lo, cl)
    atr = sma(tr, window)
    return atr

def ATR(dic: dict, term: int, term_long:int):
    hi = dic[Columns.HIGH]
    lo = dic[Columns.LOW]
    cl = dic[Columns.CLOSE]
    term = int(term)
    tr = true_range(hi, lo, cl)
    dic[Indicators.TR] = tr
    atr = sma(tr, term)
    dic[Indicators.ATR] = atr
    if term_long is not None:
        atr_long = sma(tr, term_long)
        dic[Indicators.ATR_LONG] = atr_long
        
        
        
def ATRP(dic: dict, window, ma_window=0):
    hi = dic[Columns.HIGH]
    lo = dic[Columns.LOW]
    cl = dic[Columns.CLOSE]
    window = int(window)
    tr = true_range(hi, lo, cl)
    dic[Indicators.TR] = tr
    atr = sma(tr, window)

    n = len(cl)
    atrp = nans(n)
    for i in range(n):
        a = atr[i]
        c = cl[i]
        if is_nans([a, c]):
            continue
        atrp[i] = a / c * 100.0 
        
    if ma_window > 0:
        atrp = sma(atrp, ma_window)        
    dic[Indicators.ATRP] = atrp


def ADX(hi, lo, cl, di_window: int, adx_term: int):
    tr = true_range(hi, lo, cl)
    n = len(hi)
    dmp = nans(n)     
    dmm = nans(n)     
    for i in range(1, n):
        p = hi[i]- hi[i - 1]
        m = lo[i - 1] - lo[i]
        dp = dn = 0
        if p >= 0 or n >= 0:
            if p > m:
                dp = p
            if p < m:
                dn = m
        dmp[i] = dp
        dmm[i] = dn
    dip = nans(n)
    dim = nans(n)
    dx = nans(n)
    for i in range(di_window - 1, n):
        s_tr = sum(tr[i - di_window + 1: i + 1])
        s_dmp = sum(dmp[i - di_window + 1: i + 1])
        s_dmm = sum(dmm[i - di_window + 1: i + 1])
        dip[i] = s_dmp / s_tr * 100 
        dim[i] = s_dmm / s_tr * 100
        if (dip[i] + dim[i]) == 0:
            dx[i] = 0.0
        else:
            dx[i] = abs(dip[i] - dim[i]) / (dip[i] + dim[i]) * 100
            if dx[i] < 0:
                dx[i] = 0.0
    adx = sma(dx, adx_term)
    return adx, dip, dim
        
        
    
def POLARITY(data: dict, window: int):
    hi = data[Columns.HIGH]
    lo = data[Columns.LOW]
    tr = data[Indicators.TR]
    n = len(hi)
    dmp = nans(n)     
    dmm = nans(n)     
    for i in range(1, n):
        p = hi[i]- hi[i - 1]
        m = lo[i - 1] - lo[i]
        dp = dn = 0
        if p >= 0 or n >= 0:
            if p > m:
                dp = p
            if p < m:
                dn = m
        dmp[i] = dp
        dmm[i] = dn
    dip = nans(n)
    dim = nans(n)
    for i in range(window - 1, n):
        s_tr = sum(tr[i - window + 1: i + 1])
        s_dmp = sum(dmp[i - window + 1: i + 1])
        s_dmm = sum(dmm[i - window + 1: i + 1])
        dip[i] = s_dmp / s_tr * 100 
        dim[i] = s_dmm / s_tr * 100
    
    di = subtract(dip, dim)
    pol = nans(n)
    for i in range(n):
        if is_nan(di[i]):
            continue
        if di[i] > 0:
            pol[i] = UP
        elif di[i] < 0:
            pol[i] = DOWN
    data[Indicators.POLARITY] = pol  
    
def BBRATE(data: dict, window: int, ma_window):
    cl = data[Columns.CLOSE]
    n = len(cl)
    std = nans(n)     
    for i in range(window - 1, n):
        d = cl[i - window + 1: i + 1]    
        std[i] = np.std(d)   
    ma = sma(cl, ma_window)     
    rate = nans(n)
    for i in range(n):
        c = cl[i]
        m = ma[i]
        s = std[i]
        if is_nans([c, m, s]):
            continue
        rate[i] = (cl[i] - ma[i]) / s * 100.0
    data[Indicators.BBRATE] = rate

def BB(data: dict, window: int, ma_window:int, band_multiply):
    cl = data[Columns.CLOSE]
    n = len(cl)
    #ro = roi(cl)
    std = nans(n)     
    for i in range(window - 1, n):
        d = cl[i - window + 1: i + 1]    
        std[i] = np.std(d)   
    ma = sma(cl, ma_window)     
        
    upper, lower = band(ma, std, band_multiply)    
    data[Indicators.BB] = std
    data[Indicators.BB_UPPER] = upper
    data[Indicators.BB_LOWER] = lower
    data[Indicators.BB_MA] = ma
    
    pos = band_position(cl, lower, ma, upper)
    up = probability(pos, [1, 2], 50)
    down = probability(pos, [-1, -2], 50)
    data[Indicators.BB_UP] = up
    data[Indicators.BB_DOWN] = down
    
    cross_up, cross_down, cross = cross_value(up, 50)
    data[Indicators.BB_CROSS] = cross
    data[Indicators.BB_CROSS_UP] = cross_up
    data[Indicators.BB_CROSS_DOWN] = cross_down

def time_jst(year, month, day, hour=0):
    t0 = datetime(year, month, day, hour)
    t = t0.replace(tzinfo=JST)
    return t

def pivot2(signal, threshold, left_length=2, right_length=2):
    n = len(signal)
    out = full(n, np.nan) 
    out_mid = full(n, np.nan)
    for i in range(left_length + right_length, n):
        if is_nans(signal[i - right_length - right_length: i + 1]):
            continue
        center = signal[i - right_length]
        left = signal[i - left_length - right_length: i - right_length]
        range_left = abs(max(left) - min(left))
        right = signal[i - right_length + 1: i + 1]
        d_right = np.mean(right) - center
        
        if range_left < 5:
            if center >= 90 and d_right < -threshold:
                if np.nanmin(out[i - 10: i]) != Signal.SHORT:
                    out[i] = Signal.SHORT
            elif center <= 10 and d_right > threshold:
                if np.nanmax(out[i - 10: i]) != Signal.LONG:
                    out[i] = Signal.LONG
                                
            if center >= 40 and center <= 60:
                if d_right < -threshold:
                    if np.nanmin(out_mid[i - 10: i]) != Signal.SHORT:
                        out_mid[i] = Signal.SHORT 
                elif d_right > threshold:
                    if np.nanmax(out_mid[i - 10: i]) != Signal.LONG:
                        out_mid[i] = Signal.LONG 
    return out, out_mid

def vwap_rate(price, vwap, std, median_window, ma_window):
    n = len(price)
    rate = nans(n)
    i = -1
    for p, v, s in zip(price, vwap, std):
        i += 1
        if is_nans(([p, v, s])):
            continue
        if s != 0.0:
            r = (p - v) / s * 100.0
            rate[i] = r #20 * int(r / 20)        
    med = median(rate, median_window)        
    ma = sma(med, ma_window)
    return ma

def vwap_pivot(signal, threshold, left_length, center_length, right_length):
    n = len(signal)
    out = full(n, np.nan) 
    for i in range(left_length + center_length + right_length, n):
        if is_nans(signal[i - right_length - center_length - right_length: i + 1]):
            continue
        l = i - left_length - center_length - right_length + 1
        c = i - right_length - center_length + 1
        r = i - right_length + 1
        left = signal[l: c]
        center = np.mean(signal[c: r])
        right = signal[r: i + 1]
        
        polarity = 0
        # V peak
        d_left = np.nanmax(left) - center
        d_right = np.nanmax(right) - center
        if d_left > 0 or d_right > 0:
            if d_left >= threshold and d_right >= threshold:
                polarity = 1
        # ^ Peak
        d_left = center - np.nanmin(left)
        d_right = center - np.nanmin(right)
        if d_left > 0 and d_right > 0:
            if d_left >= threshold and d_right >= threshold:
                polarity = -1
        
        if polarity == 0:      
            sig = np.nan
        elif polarity > 0:
            sig = Signal.LONG
        elif polarity < 0:
            sig = Signal.SHORT

        """
        if center >= 200:
            if sig == Signal.LONG:
                sig = np.nan
                    
        if center > -50 and center < 50:
            sig = np.nan
    
        if center <= -200:
            if sig == Signal.SHORT:
                sig = np.nan            
        """
        
        if sig == Signal.SHORT:
            if np.nanmin(out[i - 10: i]) == Signal.SHORT:
                sig = np.nan
        elif sig == Signal.LONG:
            if np.nanmax(out[i - 10: i]) == Signal.LONG:
                sig = np.nan
                           
        out[i] = sig
    return out

def VWAP(data: dict, begin_hour_list, pivot_threshold, pivot_left_len, pivot_center_len, pivot_right_len, median_window, ma_window):
    jst = data[Columns.JST]
    n = len(jst)
    MID(data)
    mid = data[Columns.MID]
    volume = data[Columns.VOLUME]
    
    vwap = full(n, np.nan)
    power_acc = full(n, np.nan)
    volume_acc = full(n, np.nan)
    std = full(n, 0)
    valid = False
    for i in range(n):
        t = jst[i]
        if t.hour in begin_hour_list:
            if t.minute == 0 and t.second == 0:
                power_sum = 0
                vwap_sum = 0
                volume_sum = 0
                valid = True
        if valid:
            vwap_sum += volume[i] * mid[i]
            volume_sum += volume[i]  
            volume_acc[i] = volume_sum
            power_sum += volume[i] * mid[i] * mid[i]  
            if volume_sum > 0:
                vwap[i] = vwap_sum / volume_sum
                power_acc[i] = power_sum
                deviation = power_sum / volume_sum - vwap[i] * vwap[i]
                if deviation > 0:
                    std[i] = np.sqrt(deviation)
                else:
                    std[i] = 0
    data[Indicators.VWAP] = vwap
    rate = vwap_rate(mid, vwap, std, median_window, ma_window)
    data[Indicators.VWAP_RATE] = rate
    
    dt = jst[1] - jst[0]
    data[Indicators.VWAP_SLOPE] = slope(vwap, 10, dt.total_seconds() / 60)
    
    for i in range(1, 5):
        upper, lower = band(vwap, std, float(i))
        data[Indicators.VWAP_UPPER + str(i)] = upper
        data[Indicators.VWAP_LOWER + str(i)] = lower
    
    signal1 = vwap_pivot(rate, pivot_threshold, pivot_left_len, pivot_center_len, pivot_right_len)
    data[Indicators.VWAP_RATE_SIGNAL] = signal1    
    pos = band_position(mid, lower, vwap, upper)
    up = probability(pos, [1, 2], 40)
    down = probability(pos, [-1, -2], 40)
    data[Indicators.VWAP_PROB] = up
    data[Indicators.VWAP_DOWN] = down
    
    signal2 = slice(up, 90, 10, 10)
    data[Indicators.VWAP_PROB_SIGNAL] = signal2
    
    
def slice(vector, threshold_upper: float, threshold_lower: float, length: int):
    n = len(vector)
    states = nans(n)
    begin = None
    state = 0
    for i in range(n):
        if state == 0:
            if vector[i] >= threshold_upper:
                state = 1
                begin = i
            elif vector[i] <= threshold_lower:
                state = -1
                begin = i
        elif state == 1:
            if vector[i] < threshold_upper:
                state = 0
                if (i - begin + 1) >= length:
                    states[i] = Signal.SHORT
        elif state == -1:
            if vector[i] > threshold_lower:
                state = 0
                if (i - begin + 1) >= length:
                    states[i] = Signal.LONG
    return states            
    
def RCI(data: dict, window: int, pivot_threshold: float, pivot_length: int):
    cl = data[Columns.CLOSE]
    rc = rci(cl, window)
    data[Indicators.RCI] = rc
    signal = slice(rc, pivot_threshold, -pivot_threshold, pivot_length)
    data[Indicators.RCI_SIGNAL] = signal
    
       
def band(vector, signal, multiply):
    n = len(vector)
    upper = nans(n)
    lower = nans(n)
    for i in range(n):
        upper[i] = vector[i] + multiply * signal[i]
        lower[i] = vector[i] - multiply * signal[i]
    return upper, lower



def volatility(data: dict, window: int):
    time = data[Columns.TIME]
    op = data[Columns.OPEN]
    hi = data[Columns.HIGH]
    lo = data[Columns.LOW]
    cl = data[Columns.CLOSE]
    n = len(cl)
    volatile = nans(n)
    for i in range(window, n):
        d = []
        for j in range(i - window + 1, i + 1):
            d.append(cl[j - 1] - op[j])
            if cl[j] > op[j]:
                # positive
                d.append(lo[j] - op[j])
                d.append(hi[j] - lo[j])
                d.append(cl[j] - hi[j])
            else:
                d.append(hi[j] - op[j])
                d.append(lo[j] - hi[j])
                d.append(cl[j] - lo[j])
        sd = stat.stdev(d)
        volatile[i] = sd / float(window) / op[i] * 100.0
    return               
            
def TREND_ADX_DI(data: dict, adx_threshold: float):
    adx = data[Indicators.ADX]
    adx_slope = slope(adx, 5)
    di_p = data[Indicators.DI_PLUS]
    di_m = data[Indicators.DI_MINUS]
    n = len(adx)
    trend = full(n, 0)
    for i in range(n):
        if adx[i] > adx_threshold and adx_slope[i] > 0: 
            delta = di_p[i] - di_m[i]
            if delta > 0:
                trend[i] = UP
            elif delta < 0:
                trend[i] = DOWN
    data[Indicators.TREND_ADX_DI] = trend

def MID(data: dict):
    cl = data[Columns.CLOSE]
    op = data[Columns.OPEN]
    n = len(cl)
    md = nans(n)
    for i in range(n):
        o = op[i]
        c = cl[i]
        if is_nans([o, c]):
            continue
        md[i] = (o + c) / 2
    data[Columns.MID] = md
    
def ATR_TRAIL(data: dict, atr_window: int, atr_multiply: float, peak_hold_term: int, horizon: int):
    atr_window = int(atr_window)
    atr_multiply = int(atr_multiply)
    peak_hold_term = int(peak_hold_term)
    time = data[Columns.TIME]
    op = data[Columns.OPEN]
    hi = data[Columns.HIGH]
    lo = data[Columns.LOW]
    cl = data[Columns.CLOSE]
    n = len(cl)
    ATR(data, atr_window, None)
    atr = data[Indicators.ATR]
    stop = nans(n)
    for i in range(n):
        h = hi[i]
        a = atr[i]
        if is_nans([h, a]):
            continue
        stop[i] = h - a * atr_multiply
        
    trail_stop = nans(n)
    for i in range(n):
        d = stop[i - peak_hold_term + 1: i + 1]
        if is_nans(d):
            continue
        trail_stop[i] = max(d)
        
    trend = full(n, np.nan)
    up = full(n, np.nan)
    down = full(n, np.nan)
    for i in range(n):
        c = cl[i]
        s = trail_stop[i]
        if is_nans([c, s]):
            continue
        if c > s:
            trend[i] = UP
            up[i] = s
        else:
            trend[i] = DOWN
            down[i] = s
            
    data[Indicators.ATR_TRAIL_UP] = up
    data[Indicators.ATR_TRAIL_DOWN] = down
            
    break_signal = full(n, np.nan)
    for  i in range(1, n):
        if trend[i - 1] == UP and trend[i] == DOWN:
            break_signal[i] = DOWN
        if trend[i - 1] == DOWN and trend[i] == UP:
            break_signal[i] = UP

    signal = full(n, np.nan)
    for i in range(horizon, n):
        brk = break_signal[i - horizon]
        if brk == DOWN and trail_stop[i] > cl[i]:
            signal[i] = Signal.SHORT
        elif brk == UP and trail_stop[i] < cl[i]:
            signal[i] = Signal.LONG        
            
    data[Indicators.ATR_TRAIL] = trail_stop
    data[Indicators.ATR_TRAIL_SIGNAL] = signal

             
def SUPERTREND(data: dict,  atr_window: int, multiply, column=Columns.MID):
    time = data[Columns.TIME]
    if column == Columns.MID:
        MID(data)
    price = data[column]
    n = len(time)
    atr = calc_atr(data, atr_window)
    atr_u, atr_l = band(data[column], atr, multiply)
    data[Indicators.ATR_UPPER] = atr_u
    data[Indicators.ATR_LOWER] = atr_l

    
def SUPERTREND_SIGNAL(data: dict, short_term):
    time = data[Columns.TIME]
    n = len(time)
    cl = data[Columns.CLOSE]
    atr_u = data[Indicators.ATR_UPPER]
    atr_l = data[Indicators.ATR_LOWER]
    price = sma(cl, short_term)
    
    trend = nans(n)
    sig = full(n, 0)
    stop_price = nans(n)
    upper = nans(n)
    lower = nans(n)
    is_valid = False
    for i in range(1, n):
        if is_valid == False:
            if is_nans([atr_l[i - 1], atr_u[i - 1]]):
                continue
            else:
                lower[i - 1] = atr_l[i - 1]
                trend[i - 1] = UP
                is_valid = True            
        if trend[i - 1] == UP:
            # up trend
            if np.isnan(lower[i - 1]):
                lower[i] = atr_l[i -1]
            else:
                if atr_l[i] > lower[i - 1]:
                    lower[i] = atr_l[i]
                else:
                    lower[i] = lower[i - 1]
            if price[i] < lower[i]:
                 # up->down trend 
                trend[i] = DOWN
                sig[i] = Signal.SHORT
                stop_price[i] = lower[i]
            else:
                trend[i] = UP
        else:
            # down trend
            if np.isnan(upper[i - 1]):
                upper[i] = atr_u[i]
            else:
                if atr_u[i] < upper[i - 1]:
                    upper[i] = atr_u[i]
                else:
                    upper[i] = upper[i - 1]
                    
            if price[i] > upper[i]:
                # donw -> up trend
                trend[i] = UP
                sig[i] = Signal.LONG
                stop_price[i] = upper[i]
            else:
                trend[i] = DOWN
           
    data[Indicators.SUPERTREND] = trend  
    data[Indicators.SUPERTREND_SIGNAL] = sig      
    data[Indicators.SUPERTREND_U] = upper  
    data[Indicators.SUPERTREND_L] = lower  
    return 

def TREND_MA(data: dict, long_term, short_term, term=10):
    op = data[Columns.OPEN]
    hi = data[Columns.HIGH]
    lo = data[Columns.LOW]
    cl = data[Columns.CLOSE]
    n = len(op)
    ma = sma(cl, long_term)
    ma_short = sma(cl, short_term)
    data[Indicators.MA_LONG] = ma
    data[Indicators.MA_SHORT] = ma_short
    gap = [0 for _ in range(n)]
    for i in range(n):
        gap[i] = (ma_short[i] - ma[i]) / ma[i] * 100.0
    data[Indicators.MA_GAP] = gap
    
    trend = [ 0 for _ in range(n)]
    for i in range(n):
        if ma[i] < lo[i]:
            trend[i] = 1 
        elif ma[i] > hi[i]:
            trend[i] = -1
    data[Indicators.MA_TREND] = trend
    

def diff(data: dict, column: str):
    signal = data[column]
    time = data[Columns.TIME]
    n = len(signal)
    out = nans(n)
    for i in range(1, n):
        dt = time[i] - time[i - 1]
        out[i] = (signal[i] - signal[i - 1]) / signal[i - 1] / (dt.seconds / 60) * 100.0
    return out

def test():
    sig = [29301.79, 29332.16, 28487.87, 28478.56, 28222.48,
           28765.66, 28489.13, 28124.28, 28333.52]
    ma = full(-1, len(sig))
    
    x = rci(sig, 9)
    print(x)
    
if __name__ == '__main__':
    test()
    

