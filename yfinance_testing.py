import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
from datetime import datetime, timedelta
from pandas_datareader import data as pdr

tickers = ['AAPL', 'MSFT']

data = yf.download(tickers, period='ytd', group_by='tickers', auto_adjust=True)
data.head()

max_date = data.index.max().date()
min_date = data.index.min().date()

# Select a stock ticker symbol
symbol = "SPOT"

# Get historical market prices and range of dates
prices = yf.Ticker(symbol)
prices = prices.history(period="max")
max_date = prices.index.max().date()
min_date = prices.index.min().date()

# Get the company's full name and logo (if available)
stock_info = yf.Ticker(symbol).info
logo_url = stock_info["logo_url"]
company_name = stock_info['longName']

# Preview the data
print(f"Prices available from {min_date} to {max_date}.")
prices.head()

# Select a date range (by default all dates will be plotted)
start_date = min_date
end_date = max_date

# Slice prices based on the provided date range
prices = prices[start_date:end_date]

# Create a candlesitck chart
fig = go.Figure(
    data=[
        go.Candlestick(
            x=prices.index,
            open=prices["Open"],
            high=prices["High"],
            low=prices["Low"],
            close=prices["Close"],
            # Set line colors
            increasing_line_color="tomato",
            decreasing_line_color="forestgreen",
        )
    ]
)

# Add image (if available)
if not logo_url == "":
    fig.add_layout_image(
        dict(
            source=logo_url,
            xref="paper",
            yref="paper",
            x=1,
            y=1.05,
            sizex=0.2,
            sizey=0.2,
            xanchor="right",
            yanchor="bottom",
        )
    )

# Customize the layout
fig.update_layout(
    title=f"{company_name} ({symbol}) Stock Prices",  # Set title
    width=900,  # Set width
    height=500,  # Set height
    xaxis_rangeslider_visible=True,  # Set to False to remove Rangeslider
    template="ggplot2",  # Set a Plotly theme
)

fig.show()







