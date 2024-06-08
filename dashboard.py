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
from technical import VWAP, BB, ATR_TRAIL, ADX, RCI

from utils import Utils
from mt5_api import Mt5Api
from common import Indicators, Columns

from strategy import Simulation

trade_param = {'begin_hour':7, 
               'begin_minute':30,
               'hours': 8,
               'sl': 500,
               'volume': 0.1,
               'position_max':5,
               'target':250, 
               'trail_stop': 300,
               'timelimit':0}



TICKERS = ['NIKKEI', 'DOW', 'NSDQ', 'USDJPY']
TIMEFRAMES = ['M1', 'M5', 'M15', 'M30', 'H1', 'H4', 'D1']
BARSIZE = ['100', '200', '300', '400', '600', '800', '1500', '2000', '3000']
HOURS = list(range(0, 24))
MINUTES = list(range(0, 60))

INTERVAL_MSEC = 30 * 1000

technical_param = {'VWAP': {'begin_hour_list': [7, 19], 
                            'pivot_threshold':10, 
                            'pivot_left_len':5,
                            'pivot_center_len':7,
                            'pivot_right_len':5,
                            'median_window': 5,
                            'ma_window': 15},
                    'RCI': {'window': 30,
                            'pivot_threshold': 70,
                            'pivot_len': 10},
                    'ATR_TRAIL': {'window': 90,
                                  'multiply':2.8,
                                  'peak_hold': 25,
                                  'horizon': 1
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
                                    value='ATR_TRAIL',
                                    options=[{'label': x, 'value': x} for x in ['ATR_TRAIL', 'RCI', 'VWAP1', 'VWAP2']],
                                    style={'width': '120px'})
                            ]
                    )

atr_trail_window = dcc.Input(id='atr_trail_window',type="number", min=1, max=100, step=1, value=technical_param['ATR_TRAIL']['window'])
atr_trail_multiply = dcc.Input(id='atr_trail_multiply',type="number", min=1.0, max=5.0, step=0.2, value=technical_param['ATR_TRAIL']['multiply'])
atr_trail_hold = dcc.Input(id='atr_trail_hold',type="number", min=1, max=100, step=1, value=technical_param['ATR_TRAIL']['peak_hold'])
atr_trail_horizon = dcc.Input(id='atr_trail_horizon',type="number", min=0, max=20, step=1, value=technical_param['ATR_TRAIL']['horizon'])
param1 = html.Div([html.P('ATR_TRAIL window'), atr_trail_window])
param2 = html.Div([html.P('ATR_TRAIL multiply'), atr_trail_multiply])
param3 = html.Div([html.P('ATR_TRAIL hold'), atr_trail_hold])
param4 = html.Div([html.P('ATR_TRAIL horizon'), atr_trail_horizon])

rci_window = dcc.Input(id='rci_window',type="number", min=5, max=50, step=1, value=technical_param['RCI']['window'])
rci_threshold = dcc.Input(id='rci_threshold',type="number", min=5, max=100, step=5, value=technical_param['RCI']['pivot_threshold'])
rci_len = dcc.Input(id='rci_len',type="number", min=5, max=30, step=1, value=technical_param['RCI']['pivot_len'])
param5 = html.Div([html.P('RCI window'), rci_window])
param6 = html.Div([html.P('RCI threshold'), rci_threshold])
param7 = html.Div([html.P('RCI len'), rci_len])

pivot_threshold = dcc.Input(id='pivot_threshold',type="number", min=1, max=70, step=1, value=technical_param['VWAP']['pivot_threshold'])
pivot_left_len = dcc.Input(id='pivot_left_len',type="number", min=1, max=30, step=1, value=technical_param['VWAP']['pivot_left_len'])
pivot_center_len = dcc.Input(id='pivot_center_len',type="number", min=1, max=30, step=1, value=technical_param['VWAP']['pivot_center_len'])
pivot_right_len = dcc.Input(id='pivot_right_len',type="number", min=1, max=30, step=1, value=technical_param['VWAP']['pivot_right_len'])
median_window = dcc.Input(id='median_window',type="number", min=1, max=50, step=1, value=technical_param['VWAP']['median_window'])
ma_window = dcc.Input(id='ma_window',type="number", min=1, max=50, step=1, value=technical_param['VWAP']['ma_window'])
param8 = html.Div([html.P('Pivot threshold'), pivot_threshold])
param9 = html.Div([html.P('Pivot left len'), pivot_left_len])
param10 = html.Div([html.P('Pivot center len'), pivot_center_len])
param11 = html.Div([html.P('Pivot right len'), pivot_right_len])
param12 = html.Div([html.P('VWAP median window'), median_window])
param13 = html.Div([html.P('VWAP ma window'), ma_window])




sidebar =  html.Div([   html.Div([
                                    mode_select,
                                    html.Hr(),
                                    strategy_select,
                                    html.Hr(),
                                    param1,
                                    param2,
                                    param3,
                                    param4,
                                    html.Hr(),
                                    param5,
                                    param6,
                                    param7,
                                    html.Hr(),
                                    param8,
                                    param9,
                                    param10,
                                    param11,
                                    param12,
                                    param13],
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
    State('rci_window', 'value'),
    State('rci_threshold', 'value'),
    State('rci_len', 'value'),
    State('atr_trail_window', 'value'),
    State('atr_trail_multiply', 'value'),
    State('atr_trail_hold', 'value'),
    State('atr_trail_horizon', 'value')
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
                 rci_window,
                 rci_threshold,
                 rci_len,
                 atr_trail_window,
                 atr_trail_multiply,
                 atr_trail_hold,
                 atr_trail_horizon
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

    technical_param['VWAP']['pivot_threshold'] = pivot_threshold
    technical_param['VWAP']['pivot_left_len'] = pivot_left_len
    technical_param['VWAP']['pivot_center_len'] = pivot_center_len
    technical_param['VWAP']['pivot_right_len'] = pivot_right_len
    technical_param['VWAP']['median_window'] =  median_window
    technical_param['VWAP']['ma_window'] =  ma_window
    technical_param['RCI']['window'] = rci_window
    technical_param['RCI']['pivot_threshold'] = rci_threshold
    technical_param['RCI']['pivot_len'] = rci_len
    technical_param['ATR_TRAIL']['window'] = atr_trail_window
    technical_param['ATR_TRAIL']['multiply'] = atr_trail_multiply
    technical_param['ATR_TRAIL']['peak_hold'] = atr_trail_hold
    technical_param['ATR_TRAIL']['horizon'] = atr_trail_horizon

    indicators1(symbol, data, technical_param)
    data = Utils.sliceDictLast(data, num_bars)
    trade_param['strategy'] = strategy_select    
    sim = Simulation(trade_param)


    df, summary, profit_curve = sim.run(data)
    trade_table = dbc.Table.from_dataframe(df, striped=True, bordered=True, hover=True)
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
    param = technical_param['RCI']
    RCI(data, param['window'], param['pivot_threshold'], param['pivot_len'])
    param = technical_param['ATR_TRAIL']
    ATR_TRAIL(data, param['window'], param['multiply'], param['peak_hold'], param['horizon'])
    
def add_markers(fig, time, signal, data, value, symbol, color, row=0, col=0):
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
                            opacity=0.3,
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


    
def add_rci_chart(fig, data, row):
    jst = data['jst']
    r = row
    fig.add_trace(go.Scatter(x=jst, y=data['RCI'], line=dict(color='blue', width=2)), row=r, col=1)
    add_markers(fig, jst, data['RCI_SIGNAL'], data['RCI'], 1, 'triangle-up', 'Green', row=r, col=1)
    add_markers(fig, jst, data['RCI_SIGNAL'], data['RCI'], -1, 'triangle-down', 'Red', row=r, col=1)    
       
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
    

def create_graph(symbol, timeframe, data):    
    jst = data['jst']
    xtick = (5 - jst[0].weekday()) % 5
    tfrom = jst[0]
    tto = jst[-1]
    if timeframe == 'D1' or timeframe == 'H1':
        form = '%m-%d'
    else:
        form = '%d/%H:%M'
    fig = create_fig([7.0, 1.0, 1.0, 1.0, 1.0])
    add_candle_chart(fig, data, 1)
    #add_vwap_line(fig, data, 2)
    add_rci_chart(fig, data, 3)
    add_vwap_chart(fig, data, 4)
    add_atr_stop_line(fig, data, 1)
    fig.update_layout(height=900, width=1200, showlegend=False, xaxis_rangeslider_visible=False)
    fig.update_layout({  'title': symbol + '  ' + timeframe + '  ('  +  str(tfrom) + ')  ...  (' + str(tto) + ')'})
    """
    fig.update_xaxes(   {'title': 'Time',
                                        'showgrid': True,
                                        'ticktext': [x.strftime(form) for x in jst][xtick::5],
                                        'tickvals': np.arange(xtick, len(jst), 5)
                        })
    """
    #fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    fig.update_yaxes(title_text="RCI", range=[-150, 150], row=3, col=1)
    fig.update_yaxes(title_text="VWAP Rate", row=4, col=1)      
    fig.update_yaxes(title_text="Trail Stop", row=6, col=1)    
    return dcc.Graph(id='stock-graph', figure=fig)

if __name__ == '__main__':    
    app.run_server(debug=True, port=3333)

