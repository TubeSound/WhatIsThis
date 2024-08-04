# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 10:23:54 2023

@author: IKU-Trader
"""

import os
import shutil

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import time

import dash
import dash_bootstrap_components as dbc
from dash import Dash, dcc, html, dash_table
from dash.dependencies import Input, Output, State

import plotly
import plotly.graph_objs as go
from plotly.figure_factory import create_candlestick

from dateutil import tz

JST = tz.gettz('Asia/Tokyo')
UTC = tz.gettz('utc')
from technical import MABAND, VWAP, BB, ATR_TRAIL, ADX, SUPERTREND, SUPERTREND_SIGNAL, detect_signal

from utils import Utils
from mt5_api import Mt5Api
from common import Indicators, Columns

from strategy import Simulation

CHART_WIDTH = 1400
CHART_HEIGHT = 1000

trade_param = {'begin_hour':9, 
               'begin_minute':30,
               'hours': 20,
               'sl': {'method': 1, 'value': 100},
               'volume': 0.1,
               'position_max':5,
               'target':0, 
               'trail_stop': 0,
               'timelimit':0}

STRATEGY = ['SUPERTREND', 'EMABREAK']

TICKERS = ['NIKKEI', 'DOW', 'NSDQ', 'USDJPY']
TIMEFRAMES = ['M1', 'M5', 'M15', 'M30', 'H1', 'H4', 'D1']
BARSIZE = ['100', '200', '300', '400', '600', '800', '1500', '2000', '3000']
HOURS = list(range(0, 24))
MINUTES = list(range(0, 60))

INTERVAL_MSEC = 30 * 1000

technical_param = { 'MABAND': 
                                {'short_term': 7,
                                'long_term': 15,
                                'adx_window': 30,
                                'di_window': 20,
                                'adx_threshold': 20
                           },
                    'VWAP': {'begin_hour_list': [7, 19], 
                            'pivot_threshold':10, 
                            'pivot_left_len':5,
                            'pivot_center_len':7,
                            'pivot_right_len':5,
                            'median_window': 5,
                            'ma_window': 15},
                    'ADX': {'window': 30,
                            'window_long': 70,
                            'di_window': 20},
                    'SUPERTREND': {'atr_window': 30,
                                  'atr_multiply':2.2,
                                  'ma_short': 20,
                                  'break_count': 2
                                  }
                    }

VWAP_BEGIN_HOUR_FX = [8]

MODE = ['Live', 'Fix', 'Pause']

api = Mt5Api()
app = Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])


old_time = None
old_symbol = None
old_timeframe = None

# ----

symbol_dropdown = dcc.Dropdown( id='symbol_dropdown',
                                    multi=False,
                                    value=TICKERS[1],
                                    options=[{'label': x, 'value': x} for x in TICKERS],
                                    style={'width': '140px'})

symbol = html.Div([ html.P('Ticker Symbol', style={'margin-top': '16px', 'margin-bottom': '4px'}, className='font-weight-bold'), symbol_dropdown])
timeframe_dropdown = dcc.Dropdown(  id='timeframe_dropdown', 
                                        multi=False, 
                                        value=TIMEFRAMES[2], 
                                        options=[{'label': x, 'value': x} for x in TIMEFRAMES],
                                        style={'width': '120px'})                
timeframe =  html.Div(  [   html.P('Time Frame',
                                style={'margin-top': '16px', 'margin-bottom': '4px'},
                                className='font-weight-bold'),
                                timeframe_dropdown])

barsize_dropdown = dcc.Dropdown(id='barsize_dropdown', 
                                    multi=False, 
                                    value=BARSIZE[2],
                                    options=[{'label': x, 'value': x} for x in BARSIZE],
                                    style={'width': '120px'})

barsize = html.Div([    html.P('Display Bar Size',
                               style={'margin-top': '16px', 'margin-bottom': '4px'},
                               className='font-weight-bold'),
                                barsize_dropdown])

dtp = dcc.DatePickerSingle(   id='start_date', 
                                                        min_date_allowed = datetime(2018,1,1), 
                                                        max_date_allowed = datetime.today(), 
                                                        date= datetime.today(),
                                                        month_format='YYYY-MM-DD',
                                                        placeholder='YYYY-MM-DD'
                                                    )
btn1 = dbc.Button("-", id='back_day', n_clicks=0)
btn2 = dbc.Button("+", id='next_day', n_clicks=0)

date_picker = html.Div(     [   
                                html.P('Start Date', style={'margin-top': '16px', 'margin-bottom': '4px'}, className='font-weight-bold'),
                                dbc.Row([dbc.Col(dtp, width=5), dbc.Col(btn1, width=1), dbc.Col(btn2, width=1)])
                            ])

header = html.Div(  [  dbc.Row([                                              
                                    dbc.Col(symbol, width=2),
                                    dbc.Col(timeframe, width=2),
                                    dbc.Col(barsize, width=2),
                                    dbc.Col(date_picker, width=5)
                                ])
                    ]
                ) 

mode_select = html.Div(     [   
                        html.P('Mode', style={'margin-top': '16px', 'margin-bottom': '4px'}, className='font-weight-bold'),
                       dcc.Dropdown(id='mode_select', 
                                    multi=False, 
                                    value=MODE[0],
                                    options=[{'label': x, 'value': x} for x in MODE],
                                    style={'width': '80px'})
                            ]
                    )

strategy_select = html.Div(     [   
                        html.P('Strategy', style={'margin-top': '16px', 'margin-bottom': '4px'}, className='font-weight-bold'),
                        dcc.Dropdown(id='strategy_select', 
                                    multi=False, 
                                    value=STRATEGY[0],
                                    options=[{'label': x, 'value': x} for x in STRATEGY],
                                    style={'width': '120px'})
                            ]
                    )



ma_short = html.Div([    html.P('MA Short'),
                        dcc.Input(id='ma_short',type="number", min=5, max=50, step=1, value=technical_param['MABAND']['short_term'])
                   ])
ma_long = html.Div([    html.P('MA Long'),
                        dcc.Input(id='ma_long',type="number", min=5, max=400, step=1, value=technical_param['MABAND']['long_term'])
                   ])
supertrend_window = dcc.Input(id='supertrend_window',type="number", min=5, max=50, step=1, value=technical_param['SUPERTREND']['atr_window'])
supertrend_multiply = dcc.Input(id='supertrend_multiply',type="number", min=0.2, max=5, step=0.1, value=technical_param['SUPERTREND']['atr_multiply'])
supertrend_ma = dcc.Input(id='supertrend_ma',type="number", min=5, max=50, step=1, value=technical_param['SUPERTREND']['ma_short'])
supertrend_break = dcc.Input(id='supertrend_break_count',type="number", min=0, max=10, step=1, value=technical_param['SUPERTREND']['break_count'])
param1 = html.Div([html.P('SUPERTREND ATR window'), supertrend_window])
param2 = html.Div([html.P('multiply'), supertrend_multiply])
param3 = html.Div([html.P('MA window'), supertrend_ma])
param4 = html.Div([html.P('break count'), supertrend_break])

adx_window = dcc.Input(id='adx_window',type="number", min=10, max=100, step=1, value=technical_param['ADX']['window'])
adx_window_long = dcc.Input(id='adx_window_long',type="number", min=10, max=100, step=1, value=technical_param['ADX']['window_long'])
adx_di_window = dcc.Input(id='adx_di_window',type="number", min=1, max=100, step=1, value=technical_param['ADX']['di_window'])
param11 = html.Div([html.P('ADX window'), adx_window])
param12 = html.Div([html.P('window long'), adx_window_long])
param13 = html.Div([html.P('DI window'), adx_di_window])

pivot_threshold = dcc.Input(id='pivot_threshold',type="number", min=1, max=70, step=1, value=technical_param['VWAP']['pivot_threshold'])
pivot_left_len = dcc.Input(id='pivot_left_len',type="number", min=1, max=30, step=1, value=technical_param['VWAP']['pivot_left_len'])
pivot_center_len = dcc.Input(id='pivot_center_len',type="number", min=1, max=30, step=1, value=technical_param['VWAP']['pivot_center_len'])
pivot_right_len = dcc.Input(id='pivot_right_len',type="number", min=1, max=30, step=1, value=technical_param['VWAP']['pivot_right_len'])
median_window = dcc.Input(id='median_window',type="number", min=1, max=50, step=1, value=technical_param['VWAP']['median_window'])
ma_window = dcc.Input(id='ma_window',type="number", min=1, max=50, step=1, value=technical_param['VWAP']['ma_window'])
param21 = html.Div([html.P('Pivot threshold'), pivot_threshold])
param22 = html.Div([html.P('Pivot left len'), pivot_left_len])
param23 = html.Div([html.P('Pivot center len'), pivot_center_len])
param24 = html.Div([html.P('Pivot right len'), pivot_right_len])
param25 = html.Div([html.P('VWAP median window'), median_window])
param26 = html.Div([html.P('VWAP ma window'), ma_window])




sidebar =  html.Div([   html.Div([
                                    mode_select,
                                    html.Hr(),
                                    strategy_select,
                                    html.Hr(),
                                    ma_short,
                                    ma_long,
                                    html.Hr(),
                                    param1,
                                    param2,
                                    param3,
                                    param4,
                                    html.Hr(),
                                    param11,
                                    param12,
                                    param13,
                                    html.Hr(),
                                    param21,
                                    param22,
                                    param23,
                                    param24,
                                    param25,
                                    param26],
                        style={'height': '50vh', 'margin': '8px'})
                    ])

contents = html.Div([   
                        #dbc.Row([html.H5('MetaTrader', style={'margin-top': '2px', 'margin-left': '24px'})], style={"height": "3vh"}, className='bg-primary text-white'),
                        dbc.Row([header], style={"height": "10vh"}, className='bg-light text-dark'),
                        dbc.Row([html.Div(id='chart')], className='bg-white'),
                        dbc.Row([html.Div(id='table')], className='bg-white'),
                        dcc.Interval(id='timer', interval=INTERVAL_MSEC, n_intervals=0)
                    ])

app.layout = dbc.Container( [dbc.Row(   [
                                            dbc.Col(sidebar, width=1, className='bg-info'),
                                            dbc.Col(contents, width=9)
                                        ],
                                        style={"height": "150vh"}),
                            ],
                            fluid=True)

# -----
def str2date(s):
    s = s[:10]
    values = s.split('-')
    t = datetime(int(values[0]), int(values[1]), int(values[2]))
    t = t.replace(tzinfo=JST)
    return t
    
def date2str(t):
    s =  t.strftime("%Y-%m-%d")   
    return s

@app.callback(
    Output('start_date', 'date'),
    [Input('back_day', 'n_clicks'),
     Input('next_day', 'n_clicks'),
     Input('start_date', 'date')]
)
def update_output(n_clicks1, n_clicks2, date):
    if n_clicks1 == 0 and n_clicks2 == 0:
        return date
    
    ctx = dash.callback_context
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    #print('input', date)
    try:
        t = str2date(date)
    except:
        return date
    t2 = datetime(t.year, t.month, t.day, 7)
    if button_id == 'back_day':    
        t2 -= timedelta(days=1)
        s = date2str(t2)
        print('output', s)
        return s  
    elif button_id == 'next_day':
        t2 += timedelta(days=1)
        return date2str(t2)
    else:
        return date

@app.callback(
    [Output('chart', 'children'), Output('table', 'children')],
    Input('timer', 'n_intervals'),
    State('mode_select', 'value'),
    State('strategy_select', 'value'),
    State('symbol_dropdown', 'value'), 
    State('timeframe_dropdown', 'value'), 
    State('barsize_dropdown', 'value'),
    State('start_date', 'date'),
    State('pivot_threshold', 'value'),
    State('pivot_left_len', 'value'),
    State('pivot_center_len', 'value'),
    State('pivot_right_len', 'value'),
    State('median_window', 'value'),
    State('ma_window', 'value'),
    State('adx_window', 'value'),
    State('adx_window_long', 'value'),
    State('adx_di_window', 'value'),
    State('supertrend_window', 'value'),
    State('supertrend_multiply', 'value'),
    State('supertrend_ma', 'value'),
    State('supertrend_break_count', 'value'),
    State('ma_short', 'value'),
    State('ma_long', 'value')
)
def update_chart(interval,
                 mode_select,
                 strategy_select,
                 symbol,
                 timeframe,
                 num_bars,
                 date,
                 pivot_threshold,
                 pivot_left_len,
                 pivot_center_len,
                 pivot_right_len,
                 median_window,
                 ma_window,
                 adx_window,
                 adx_window_long,
                 adx_di_window,
                 supertrend_window,
                 supertrend_multiply,
                 supertrend_ma,
                 supertrend_break_count,
                 ma_short,
                 ma_long
                 ):
    global graph
    global trade_table

    num_bars = int(num_bars)
    #print('Mode', mode_select)
    if mode_select == 'Live':
        data = api.get_rates(symbol, timeframe, num_bars + 60 * 8)
    elif mode_select == 'Fix':
        old_time = date
        old_symbol = symbol
        old_timeframe = timeframe    
        jst = calc_date(date, timeframe, num_bars)
        data = api.get_rates_jst(symbol, timeframe, jst[0], jst[1])
    elif mode_select == 'Pause':
        return graph, trade_table
                
    size = len(data['time'])
    if size < 50:
        return
    #print('Data... time ', data['time'][0], size)

    technical_param['MABAND']['short'] = ma_short
    technical_param['MABAND']['long'] = ma_long
    technical_param['VWAP']['pivot_threshold'] = pivot_threshold
    technical_param['VWAP']['pivot_left_len'] = pivot_left_len
    technical_param['VWAP']['pivot_center_len'] = pivot_center_len
    technical_param['VWAP']['pivot_right_len'] = pivot_right_len
    technical_param['VWAP']['median_window'] =  median_window
    technical_param['VWAP']['ma_window'] =  ma_window
    technical_param['ADX']['window'] = adx_window
    technical_param['ADX']['window_long'] = adx_window_long
    technical_param['ADX']['di_window'] = adx_di_window
    technical_param['SUPERTREND']['atr_window'] = supertrend_window
    technical_param['SUPERTREND']['atr_multiply'] = supertrend_multiply
    technical_param['SUPERTREND']['ma_short'] = supertrend_ma
    technical_param['SUPERTREND']['break_count'] = supertrend_break_count

    indicators1(symbol, data, technical_param)
    data = Utils.sliceDictLast(data, num_bars)
    trade_param['strategy'] = strategy_select    
    sim = Simulation(trade_param)


    #df, summary, profit_curve = sim.run(data)
    #trade_table = dbc.Table.from_dataframe(df, striped=True, bordered=True, hover=True)
    trade_table = None
    graph = create_graph(symbol, timeframe, data)
    return graph, trade_table

def calc_date(date, timeframe, barsize):
    def dt_bar(timeframe):
        if timeframe[0] == 'M':
            return timedelta(minutes=int(timeframe[1:]))
        elif timeframe[0] == 'H':
            return timedelta(hours=int(timeframe[1:]))
        elif timeframe[0] == 'D':
            return timedelta(days=int(timeframe[1:]))
        
    values = date.split('T')
    values = values[0].split('-')
    year = int(values[0])
    month = int(values[1])
    day = int(values[2])
    tfrom = datetime(year, month, day, 7, tzinfo=JST)
    dt = dt_bar(timeframe)
    tto = tfrom + dt * barsize
    tfrom -= dt * barsize
    return tfrom, tto
    
def indicators1(symbol, data, technical_param):
    param = technical_param['VWAP']
    if symbol.lower() == 'usdjpy':
        hours = VWAP_BEGIN_HOUR_FX
    else:
        hours = param['begin_hour_list']
    
    VWAP(data,
         hours,
         param['pivot_threshold'],
         param['pivot_left_len'],
         param['pivot_center_len'],
         param['pivot_right_len'],
         param['median_window'],
         param['ma_window']
         )    
    param =technical_param['SUPERTREND']
    SUPERTREND(data, param['atr_window'], param['atr_multiply'],  param['ma_short'])
    SUPERTREND_SIGNAL(data, param['break_count'])
    
    '''
    param = technical_param['MABAND']
    ma_short = param['short_term']
    ma_long = param['long_term']
    MABAND(data, ma_short, ma_long, param['di_window'], param['adx_window'], param['adx_threshold'])
    '''
    
def add_markers(fig, time, signal, data, value, symbol, color, row=1, col=1):
    if len(signal) == 0:
        return 
    x = []
    y = []
    
    if value > 0:
        offset = 50
    else:
        offset = -50
    for t, s, d in zip(time, signal, data) :
        try:
            if np.isnan(s):
                continue
        except:
            continue  
        if s == value:
            x.append(t)
            y.append(d + offset)
    #print('Marker ', symbol, x, y)
    markers = go.Scatter(
                            mode='markers',
                            x=x,
                            y=y,
                            opacity=1.0,
                            marker_symbol=symbol,
                            marker=dict(color=color, size=20, line=dict(color='White', width=2)),
                            showlegend=False
                        )
    fig.add_trace(markers, row=row, col=col)
    
def add_marker(fig, x, y, symbol, color, row=1, col=1):
    markers = go.Scatter(
                            mode='markers',
                            x=[x],
                            y=[y],
                            opacity=1.0,
                            marker_symbol=symbol,
                            marker=dict(color=color, size=20, line=dict(color='White', width=2)),
                            showlegend=False
                        )
    fig.add_trace(markers, row=row, col=col)

def create_fig(heights):
    rows = len(heights)
    fig=go.Figure()
    fig = plotly.subplots.make_subplots(rows=rows,
                                        cols=1,
                                        shared_xaxes=True,
                                        vertical_spacing=0.01, 
                                        row_heights=heights)
    return fig

def add_candle_chart(fig, data, row):
    t0 = time.time()
    jst = data['jst']
    n = len(jst)
    fig.add_trace(go.Candlestick(x=jst,
                    open=data['open'],
                    high=data['high'],
                    low=data['low'],
                    close=data['close'], 
                    name = 'market data'),
                  row=row,
                  col=1)
    
    colors = ['green' if data['open'][i] - data['close'][i] >= 0 else 'red' for i in range(n)]
    fig.add_trace(go.Bar(x=jst, y=data['volume'], marker_color=colors), row=row + 1, col=1)
    
def add_trend_bar(fig, data, row):
    colors = []
    value = []
    for trend in data['MABAND']:
        if trend == 1: 
            color = 'green'
            d = 1
        elif trend == -1:
            color = 'red'
            d = 1
        else:
            color = 'white'
            d = 0
        value.append(d)
        colors.append(color)
    jst = data['jst']    
    cl = data['close']
    fig.add_trace(go.Bar(x=jst, y=value, marker_color=colors), row=row, col=1)
    _, _, up, down = detect_signal(data['MABAND'])
    for begin, end in up:
        add_marker(fig, jst[begin], cl[begin], 'circle', 'Blue')
        add_marker(fig, jst[end], cl[end], 'cross', 'Blue')
    for begin, end in down:
        add_marker(fig, jst[begin], cl[begin],  'circle', 'Red')
        add_marker(fig, jst[end], cl[end], 'cross', 'Red')
    
def add_vwap_line(fig, data, row):
    jst = data['jst']
    n = len(jst)
    colors1 = ['Cyan', 'Lime', 'Blue']
    colors2 = ['Yellow', 'Orange', 'Red']
    for i in range(1, 4):
        fig.add_trace(go.Scatter(x=jst, 
                         y=data['VWAP_UPPER' + str(i)], 
                         opacity=0.7, 
                         line=dict(color=colors1[i - 1], width=2), 
                         name='VWAP Upper'),
                         row = row,
                         col=1)
 
        fig.add_trace(go.Scatter(x=jst, 
                         y=data['VWAP_LOWER' + str(i)], 
                         opacity=0.7, 
                         line=dict(color=colors2[i - 1], width=2), 
                         name='VWAP lower'))


def add_ma_line(fig, data, row):
    jst = data['jst']
    r = row
    for name, c in zip(['MA_SHORT', 'MA_MID', 'MA_LONG'], ['red', 'blue', 'green']):
        try:
            fig.add_trace(go.Scatter(x=jst, y=data[name], line=dict(color=c, width=2)), row=r, col=1)
        except:
            pass
    
def add_adx_chart(fig, data, row):
    jst = data['jst']
    r = row
    fig.add_trace(go.Scatter(x=jst, y=data['ADX'], line=dict(color='red', width=2)), row=r, col=1)
    #fig.add_trace(go.Scatter(x=jst, y=data['ADX_LONG'], line=dict(color='blue', width=2)), row=r, col=1)
    
def add_di_chart(fig, data, row):
    jst = data['jst']
    r = row
    fig.add_trace(go.Scatter(x=jst, y=data['DI_PLUS'], line=dict(color='green', width=2)), row=r, col=1)
    fig.add_trace(go.Scatter(x=jst, y=data['DI_MINUS'], line=dict(color='red', width=2)), row=r, col=1)
       
'''    
def add_cross_chart(fig, data, row):
    jst = data['jst']
    cl = data['close']
    up, down = CROSS(data, 7, 15)
    fig.add_trace(go.Scatter(x=jst, y=cl, line=dict(color='gray', width=1)), row=row, col=1)
    for begin, end in up:   
        x = [jst[begin], jst[end]]
        y = [cl[begin], cl[end]]
        color = 'green'
        fig.add_trace(go.Scatter(x=x, y=y, line=dict(color=color, width=2)), row=row, col=1)
    for begin, end in down:   
        x = [jst[begin], jst[end]]
        y = [cl[begin], cl[end]]
        color = 'red'
        fig.add_trace(go.Scatter(x=x, y=y, line=dict(color=color, width=2)), row=row, col=1)
'''       
       
def add_vwap_chart(fig, data, row):
    jst = data['jst']
    #fig.add_trace(go.Scatter(x=jst, y=data['VWAP_SLOPE'], line=dict(color='Green', width=2)), row=row, col=1)
    r = row
    fig.add_trace(go.Scatter(x=jst, y=data['VWAP_RATE'], line=dict(color='blue', width=2)), row=r, col=1)
    add_markers(fig, jst, data['VWAP_RATE_SIGNAL'], data['VWAP_RATE'], 1, 'triangle-up', 'Green', row=r, col=1)
    add_markers(fig, jst, data['VWAP_RATE_SIGNAL'], data['VWAP_RATE'], -1, 'triangle-down', 'Red', row=r, col=1)
    r += 1
    fig.add_trace(go.Scatter(x=jst, y=data['VWAP_PROB'], line=dict(color='blue', width=2)), row=r, col=1)
    fig.add_trace(go.Scatter(x=jst, y=data['VWAP_DOWN'], line=dict(color='red', width=2)), row=r, col=1)
    add_markers(fig, jst, data['VWAP_PROB_SIGNAL'], data['VWAP_PROB'], 1, 'triangle-up', 'Green', row=r, col=1)
    add_markers(fig, jst, data['VWAP_PROB_SIGNAL'], data['VWAP_PROB'], -1, 'triangle-down', 'Red', row=r, col=1)
    
def add_atr_stop_line(fig, data, row):
    jst = data['jst']
    #fig.add_trace(go.Scatter(x=jst, y=data['VWAP_SLOPE'], line=dict(color='Green', width=2)), row=row, col=1)
    r = row
    fig.add_trace(go.Scatter(x=jst, y=data[Indicators.ATR_TRAIL_UP], line=dict(color='blue', width=2)), row=r, col=1)
    fig.add_trace(go.Scatter(x=jst, y=data[Indicators.ATR_TRAIL_DOWN], line=dict(color='Orange', width=2)), row=r, col=1)
    add_markers(fig, jst, data[Indicators.ATR_TRAIL_SIGNAL], data[Indicators.ATR_TRAIL], 1, 'triangle-up', 'Green', row=r, col=1)
    add_markers(fig, jst, data[Indicators.ATR_TRAIL_SIGNAL], data[Indicators.ATR_TRAIL], -1, 'triangle-down', 'Red', row=r, col=1)
    
def add_supertrend_line(fig, data, row):
    jst = data['jst']
    #fig.add_trace(go.Scatter(x=jst, y=data['VWAP_SLOPE'], line=dict(color='Green', width=2)), row=row, col=1)
    r = row
    fig.add_trace(go.Scatter(x=jst, y=data[Indicators.MA_SHORT], line=dict(color='gray', width=1)), row=r, col=1)
    fig.add_trace(go.Scatter(x=jst, y=data[Indicators.SUPERTREND_UPPER], line=dict(color='blue', width=2)), row=r, col=1)
    fig.add_trace(go.Scatter(x=jst, y=data[Indicators.SUPERTREND_LOWER], line=dict(color='Orange', width=2)), row=r, col=1)
    add_markers(fig, jst, data[Indicators.SUPERTREND_SIGNAL], data[Columns.CLOSE], 1, 'triangle-up', 'Green', row=r, col=1)
    add_markers(fig, jst, data[Indicators.SUPERTREND_SIGNAL], data[Columns.CLOSE], -1, 'triangle-down', 'Red', row=r, col=1)    

def create_graph(symbol, timeframe, data):    
    jst = data['jst']
    xtick = (5 - jst[0].weekday()) % 5
    tfrom = jst[0]
    tto = jst[-1]
    if timeframe == 'D1' or timeframe == 'H1':
        form = '%m-%d'
    else:
        form = '%d/%H:%M'
    fig = create_fig([5.0, 1.0, 1.0])
    add_candle_chart(fig, data, 1)
    #add_ma_line(fig, data, 1)
    #add_vwap_line(fig, data, 2)
    #add_trend_bar(fig, data, 3)
    #add_cross_chart(fig, data, 4)
    #add_adx_chart(fig, data, 5)

    #add_atr_stop_line(fig, data, 1)
    add_supertrend_line(fig, data, 1)
    fig.update_layout(height=CHART_HEIGHT, width=CHART_WIDTH, showlegend=False, xaxis_rangeslider_visible=False)
    fig.update_layout({  'title': symbol + '  ' + timeframe + '  ('  +  str(tfrom) + ')  ...  (' + str(tto) + ')'})
    fig.update_xaxes(   {'title': 'Time',
                                        'showgrid': True,
                                        'ticktext': [x.strftime(form) for x in jst][xtick::5],
                                        'tickvals': np.arange(xtick, len(jst), 5)
                        })

    #fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    return dcc.Graph(id='stock-graph', figure=fig)

if __name__ == '__main__':    
    app.run_server(debug=True, port=3333)

