
import MetaTrader5 as mt5
import pandas as pd
from dateutil import tz
from datetime import datetime, timedelta, timezone
from time_utils import TimeUtils
JST = tz.gettz('Asia/Tokyo')
UTC = tz.gettz('utc')  
        
def server_time(begin_month, begin_sunday, end_month, end_sunday, delta_hour_from_gmt_in_summer):
    now = datetime.now(JST)
    dt, tz = TimeUtils.delta_hour_from_gmt(now, begin_month, begin_sunday, end_month, end_sunday, delta_hour_from_gmt_in_summer)
    #delta_hour_from_gmt  = dt
    #server_timezone = tz
    #print('SeverTime GMT+', dt, tz)
    return dt, tz  
  
def server_time_to_utc(time: datetime):
    dt, tz = server_time(3, 2, 11, 1, 3.0)
    return time - dt

def utc_to_server_time(utc: datetime): 
    dt, tz = server_time(3, 2, 11, 1, 3.0)
    return utc + dt

def adjust(time):
    utc = []
    jst = []
    for ts in time:
        t0 = pd.to_datetime(ts)
        t0 = t0.replace(tzinfo=UTC)
        t = server_time_to_utc(t0)
        utc.append(t)
        tj = t.astimezone(JST)
        jst.append(tj)  
    return utc, jst
            
class TimeFrame:
    TICK = 'TICK'
    M1 = 'M1'
    M5 = 'M5'
    M15 = 'M15'
    M30 = 'M30'
    H1 = 'H1'
    H4 = 'H4'
    D1 = 'D1'
    W1 = 'W1'
    timeframes = {  M1: mt5.TIMEFRAME_M1, 
                    M5: mt5.TIMEFRAME_M5,
                    M15: mt5.TIMEFRAME_M15,
                    M30: mt5.TIMEFRAME_M30,
                    H1: mt5.TIMEFRAME_H1,
                    H4: mt5.TIMEFRAME_H4,
                    D1: mt5.TIMEFRAME_D1,
                    W1: mt5.TIMEFRAME_W1}
            
    @staticmethod 
    def const(timeframe_str: str):
        return TimeFrame.timeframes[timeframe_str]            
            
class Mt5Api:
    def __init__(self):
        self.connect()
        
    def connect(self):
        if mt5.initialize():
            print('Connected to MT5 Version', mt5.version())
        else:
            print('initialize() failed, error code = ', mt5.last_error())

    def get_rates(self, symbol: str, timeframe: str, length: int):
        #print(symbol, timeframe)
        
        rates = mt5.copy_rates_from_pos(symbol,  TimeFrame.const(timeframe), 0, length)
        if rates is None:
            raise Exception('get_rates error')
        return self.parse_rates(rates)

    def get_rates_jst(self, symbol: str, timeframe: str, jst_from: datetime, jst_to: datetime ):
        #print(symbol, timeframe)
        utc_from = jst_from.astimezone(tz=UTC)
        utc_to = jst_to.astimezone(tz=UTC)
        t_from = utc_to_server_time(utc_from)
        t_to = utc_to_server_time(utc_to)
        rates = mt5.copy_rates_range(symbol, TimeFrame.const(timeframe), t_from, t_to)
        if rates is None:
            raise Exception('get_rates error')
        return self.parse_rates(rates)

    def parse_rates(self, rates):
        df = pd.DataFrame(rates)
        t0 = pd.to_datetime(df['time'], unit='s') 
        utc, jst = adjust(t0)
        
        dic = {}
        dic['time'] = utc
        dic['jst'] = jst
        dic['open'] = df['open'].to_list()
        dic['high'] = df['high'].to_list()
        dic['low'] = df['low'].to_list()
        dic['close'] = df['close'].to_list()
        dic['volume'] = df['tick_volume'].to_list()        
        return dic
        


def test1():
    symbol = 'NIKKEI'
    mt5api = Mt5Api()
    dic = mt5api.get_rates(symbol, 'M1', 100)
    jst = dic['jst']
    print(jst[10:])

    pass


if __name__ == '__main__':
    test1()
