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
from technical import MAGAP, MAGAP_SIGNAL, SUPERTREND, SUPERTREND_SIGNAL, detect_signal, detect_gap_cross

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
TIMEFRAMES = ['M5', 'M15', 'M30', 'H1', 'H4', 'D1']
BARSIZE = ['100', '200', '300', '400', '600', '800', '1500', '2000', '3000']
HOURS = list(range(0, 24))
MINUTES = list(range(0, 60))

INTERVAL_MSEC = 30 * 1000

technical_param = { 'MAGAP': 
                                {'long_term': 192,
                                 'mid_term': 24 * 2 ,
                                 'short_term': 36,
                                 'tap': 16, 
                                 'level': 0.1, 
                                 'threshold': 0.1,
                                 'slope_threshold': 0.03,
                                 'delay_max': 16},
    
                    'SUPERTREND': {'atr_window': 40,
                                  'atr_multiply':2.5,
                                  'short_term': 20
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
                                    value=TICKERS[0],
                                    options=[{'label': x, 'value': x} for x in TICKERS],
                                    style={'width': '140px'})

symbol = html.Div([ html.P('Ticker Symbol', style={'margin-top': '16px', 'margin-bottom': '4px'}, className='font-weight-bold'), symbol_dropdown])
timeframe_dropdown = dcc.Dropdown(  id='timeframe_dropdown', 
                                        multi=False, 
                                        value=TIMEFRAMES[1], 
                                        options=[{'label': x, 'value': x} for x in TIMEFRAMES],
                                        style={'width': '120px'})                
timeframe =  html.Div(  [   html.P('Time Frame',
                                style={'margin-top': '16px', 'margin-bottom': '4px'},
                                className='font-weight-bold'),
                                timeframe_dropdown])

barsize_dropdown = dcc.Dropdown(id='barsize_dropdown', 
                                    multi=False, 
                                    value=BARSIZE[3],
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




supertrend_atr_window = dcc.Input(id='supertrend_atr_window',type="number", min=5, max=50, step=1, value=technical_param['SUPERTREND']['atr_window'])
supertrend_atr_multiply = dcc.Input(id='supertrend_atr_multiply',type="number", min=0.2, max=5, step=0.1, value=technical_param['SUPERTREND']['atr_multiply'])
supertrend_short_term = dcc.Input(id='supertrend_short_term',type="number", min=5, max=50, step=1, value=technical_param['SUPERTREND']['short_term'])
param1 = html.Div([html.P('SUPERTREND ATR window'), supertrend_atr_window])
param2 = html.Div([html.P('multiply'), supertrend_atr_multiply])
param3 = html.Div([html.P('MA window'), supertrend_short_term])

magap_long_term = dcc.Input(id='magap_long_term',type="number", min=1, max=1000, step=1, value=technical_param['MAGAP']['long_term'])
magap_mid_term = dcc.Input(id='magap_mid_term',type="number", min=1, max=200, step=1, value=technical_param['MAGAP']['mid_term'])
magap_short_term = dcc.Input(id='magap_short_term',type="number", min=1, max=100, step=1, value=technical_param['MAGAP']['short_term'])
magap_tap = dcc.Input(id='magap_tap',type="number", min=1, max=50, step=1, value=technical_param['MAGAP']['tap'])
magap_level = dcc.Input(id='magap_level',type="number", min=0, max=10, step=0.1, value=technical_param['MAGAP']['level'])
magap_threshold = dcc.Input(id='magap_threshold',type="number", min=0, max=10, step=0.1, value=technical_param['MAGAP']['threshold'])

param10 = html.Div([html.P('MAGAP long term'), magap_long_term])
param11 = html.Div([html.P('MAGAP mid term'), magap_mid_term])
param12 = html.Div([html.P('short term'), magap_short_term])
param13 = html.Div([html.P('tap'), magap_tap])
param14 = html.Div([html.P('level'), magap_level])
param15 = html.Div([html.P('threshold'), magap_threshold])


sidebar =  html.Div([   html.Div([
                                    mode_select,
                                    html.Hr(),
                                    strategy_select,
                                    html.Hr(),
                                    param1,
                                    param2,
                                    param3,
                                    html.Hr(),
                                    param10,
                                    param11,
                                    param12,
                                    param13,
                                    param14,
                                    param15
                                ],
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

    State('supertrend_atr_window', 'value'),
    State('supertrend_atr_multiply', 'value'),
    State('supertrend_short_term', 'value'),
    State('magap_long_term', 'value'),
    State('magap_mid_term', 'value'),
    State('magap_short_term', 'value'),
    State('magap_tap', 'value'),
    State('magap_level', 'value'),
    State('magap_threshold', 'value')
)
def update_chart(interval,
                 mode_select,
                 strategy_select,
                 symbol,
                 timeframe,
                 num_bars,
                 date,
                 supertrend_atr_window,
                 supertrend_atr_multiply,
                 supertrend_short_term,
                 magap_long_term,
                 magap_mid_term,
                 magap_short_term,
                 magap_tap,
                 magap_level,
                 magap_threshold
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
    technical_param['SUPERTREND']['atr_window'] = supertrend_atr_window
    technical_param['SUPERTREND']['atr_multiply'] = supertrend_atr_multiply
    technical_param['SUPERTREND']['short_term'] = supertrend_short_term
    technical_param['MAGAP']['long_term'] = magap_long_term
    technical_param['MAGAP']['mid_term'] = magap_mid_term
    technical_param['MAGAP']['short_term'] = magap_short_term
    technical_param['MAGAP']['tap'] = magap_tap
    technical_param['MAGAP']['level'] = magap_level
    technical_param['MAGAP']['threshold'] = magap_threshold

    indicators1(symbol, data, technical_param, timeframe)
    data = Utils.sliceDictLast(data, num_bars)
    trade_param['strategy'] = strategy_select    
    sim = Simulation(data, trade_param)


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
    
def indicators1(symbol, data, technical_param, timeframe):
    param = technical_param['SUPERTREND']
    SUPERTREND(data, param['atr_window'], param['atr_multiply'] )
    SUPERTREND_SIGNAL(data, param['short_term'])
    
    param = technical_param['MAGAP']
    MAGAP(timeframe, data, param['long_term'], param['mid_term'], param['short_term'], param['tap'])
    #MAGAP_SIGNAL(data, param['threshold'], param['delay_max'])
    
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
    for name, c in zip([ 'MA_SHORT', 'MA_MID', 'MA_LONG'], ['red', 'blue', 'purple']):
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
       

def add_magap_chart(fig, timeframe, data, row):
    jst = data['jst']
    fig.add_trace(go.Scatter(x=jst, y=data['MAGAP'], line=dict(color='green', width=2)), row=row, col=1)
    
    gap = data[Indicators.MAGAP]
    param = technical_param['MAGAP']
    xup, xdown = MAGAP_SIGNAL(timeframe, data, param['slope_threshold'], 10, param['delay_max'])

    for i in xup:
        add_marker(fig, jst[i], gap[i], 'triangle-up', 'green', row=row, col=1)
    for i in xdown:
        add_marker(fig, jst[i], gap[i], 'triangle-down', 'red', row=row, col=1)

       
        
       
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
    fig.add_trace(go.Scatter(x=jst, y=data[Indicators.SUPERTREND_U], line=dict(color='blue', width=2)), row=r, col=1)
    fig.add_trace(go.Scatter(x=jst, y=data[Indicators.SUPERTREND_L], line=dict(color='Orange', width=2)), row=r, col=1)
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
    fig = create_fig([5.0, 1.0, 2.0])
    add_candle_chart(fig, data, 1)
    add_supertrend_line(fig, data, 1)
    add_ma_line(fig, data, 1)
    add_magap_chart(fig, timeframe, data, 3)
    fig.update_layout(height=CHART_HEIGHT, width=CHART_WIDTH, showlegend=False, xaxis_rangeslider_visible=False)
    fig.update_layout({  'title': symbol + '  ' + timeframe + '  ('  +  str(tfrom) + ')  ...  (' + str(tto) + ')'})
    #fig.update_xaxes(   {'title': 'Time',
    #                                    'showgrid': True,
    ##                                    'ticktext': [x.strftime(form) for x in jst][xtick::5],
    #                                    'tickvals': np.arange(xtick, len(jst), 5)
    #                    })

    fig.update_xaxes(   {'title': 'Time',
                         'showgrid': True})
    
    #fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    fig.update_yaxes(title_text="MAGAP", range=(-4, 4), row=3, col=1)
    return dcc.Graph(id='stock-graph', figure=fig)

if __name__ == '__main__':    
    app.run_server(debug=True, port=3333)

