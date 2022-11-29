import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pandas_ta as ta

#Download ticker data from yfinance

ticker = ['AMZN']
data = pd.DataFrame(yf.download(ticker, '2016-01-01', progress=False))

#Call function ta

x = pd.DataFrame()
x.ta.indicators()

#Check the columns with data.columns
data.columns

#Calculate Rate Of Change with 8 periods and drop the na numbers

roc = ta.roc(data['Close'], length=8).dropna()
roc

#Calculate indicator Average True Range using columns high, low, close and adding periods

atr = ta.atr(data['High'],data['Low'], data['Close'], length=14).dropna()

#Calculate Bollinger Bands

bb = ta.bbands(data['Close'], length=14).dropna()[['BBL_14_2.0','BBM_14_2.0', 'BBU_14_2.0']]

#Calculate ADX

adx = ta.adx(data['High'], data['Low'], data['Close']).dropna()

#Calculate Simple Moving Average

simple_ma = ta.sma(data['Close'], length=20).dropna()

simple_ma2 = data['Close'].rolling(20).mean().dropna()

#Calculate volatility:

vol = data['Close'].rolling(20).std().dropna()












