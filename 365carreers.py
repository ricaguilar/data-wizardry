from array import array
from cmath import log
from ossaudiodev import SNDCTL_DSP_SUBDIVIDE
import investpy
from matplotlib import ticker
import pandas as pd
from pandas_datareader import data as wb
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import mplfinance as mpf
import plotly.graph_objects as go
from scipy.stats import norm

#How to measure a security's risk

tickers=['DIS', 'COST']

sec_data=pd.DataFrame()

for t in tickers:
    sec_data[t]=wb.DataReader(t, data_source='yahoo', start='2012-1-1')['Adj Close']

sec_data.tail()

sec_returns=np.log(sec_data/sec_data.shift(1))
sec_returns

sec_returns['DIS'].mean()
sec_returns['DIS'].mean()*250
sec_returns['DIS'].std()
sec_returns['DIS'].std()*250**0.5

#Portfolio Diversification, calculating covariance and correlation

DIS_var=sec_returns['DIS'].var()
DIS_var
DIS_var_a=sec_returns['DIS'].var()*250
DIS_var_a

COST_var=sec_returns['COST'].var()
COST_var
COST_var_a=sec_returns['COST'].var()*250
COST_var_a

cov_matrix=sec_returns.cov()
cov_matrix

cov_matrix_a=sec_returns.cov()*250
cov_matrix_a

corr_matrix=sec_returns.corr()
corr_matrix

#Risk of multiple securities in a portfolio. Calculating Portfolio risk

 #Equal weighting scheme

weights=np.array([0.5, 0.5])

 #Portfolio variance

pfolio_var=np.dot(weights.T, np.dot(sec_returns.cov()*250, weights))
pfolio_var

 #Portfolio volatility
pfolio_vol=(np.dot(weights.T, np.dot(sec_returns.cov()*250, weights)))**0.5
pfolio_vol

print(str(round(pfolio_vol, 5)*100) + '%')

#Regression in Python

data = pd.read_excel('C:/Users/34644/Desktop/PythonforFinance/Clases/Housing.xlsx')
data

data[['House Price', 'House Size (sq.ft.)']]

x=data['House Size (sq.ft.)']
y=data['House Price']
x
y

plt.scatter(x,y)
plt.axis([0, 2500, 0, 1500000])
plt.ylabel('House Price')
plt.xlabel('House Size (sq.ft)')
plt.show()

#Efficient Frotier

assets=['DIS', '^GSPC']
pf_data=pd.DataFrame()

for a in assets:
    pf_data[a]=wb.DataReader(a, data_source='yahoo', start='2010-1-1')['Adj Close']
pf_data.tail()

(pf_data/pf_data.iloc[0]*100).plot(figsize=(10,5))

#To obtain log returns

log_returns=np.log(pf_data/pf_data.shift(1))
log_returns.mean()*250
log_returns.cov()*250
log_returns.corr()*250

#How to obtain dos pesos generados aleatoriamente

num_assets=len(assets)
num_assets

arr=np.random.random(2)
arr

weights=np.random.random(2)
weights/=np.sum(weights)
weights

#Expected portfolio return

np.sum(weights*log_returns.mean())*250

#Expected portfolio volatility

np.sqrt(np.dot(weights.T, np.dot(log_returns.cov()*250, weights)))


pfolio_returns=[]
pfolio_volatilities=[]

for x in range(1000):
    weights=np.random.random(num_assets)
    weights/=np.sum(weights)
    pfolio_returns.append(np.sum(weights*log_returns.mean())*250)
    pfolio_volatilities.append(np.sqrt(np.dot(weights.T, np.dot(log_returns.cov()*250, weights))))

pfolio_volatilities=np.array(pfolio_volatilities)
pfolio_returns=np.array(pfolio_returns)

pfolio_volatilities
pfolio_returns

portfolios=pd.DataFrame({'Return': pfolio_returns, 'Volatility': pfolio_volatilities})
portfolios.head()
portfolios.tail()

portfolios.plot(x='Volatility', y='Return', kind='scatter', figsize=(10,6))
plt.xlabel('Expected Volatility')
plt.ylabel('Expected Return')
plt.show()

#Capital Asset Pricing Model

tickers=['DIS', 'SPY']

data=pd.DataFrame()
for t in tickers:
    data[t]=yf.download(t, start="2020-01-01", end="2022-07-31")['Adj Close']

df = data.pct_change().dropna()
df.head()

sec_returns=np.log(data/data.shift(1))
sec_returns.head()

cov=sec_returns.cov()*250
cov

cov_with_market=cov.iloc[0,1]
cov_with_market

market_var=sec_returns['SPY'].var()*250
market_var

DIS_beta=cov_with_market/market_var
DIS_beta

DIS_er=0.025+DIS_beta*0.05
DIS_er

#Sharpe Ratio

Sharpe=(DIS_er-0.025)/(sec_returns['DIS'].std()*250**0.5)
Sharpe

#Montecarlo simulations

rev_m=170
rev_stdev=20
iterations=1000

rev=np.random.normal(rev_m, rev_stdev, iterations)
rev

plt.figure(figsize=(15,6))
plt.plot(rev)
plt.show()

COGS=-(rev*np.random.normal(0.6,0.1))
COGS

plt.figure(figsize=(15,6))
plt.plot(COGS)
plt.show()

COGS.mean()
COGS.std()

gross_profit=rev+COGS
gross_profit

plt.figure(figsize=(15,6))
plt.plot(gross_profit)
plt.show()

gross_profit.mean()
gross_profit.std()


plt.figure(figsize=(15,6))
plt.hist(gross_profit, bins=[40,50,60,70,80,90,100,110,120])
plt.show()


plt.figure(figsize=(15,6))
plt.hist(gross_profit, bins=20)
plt.show()

#Forecasting stock prices with Montercarlo simulation

ticker='DIS'
data=pd.DataFrame()
data[ticker]=wb.DataReader(ticker, data_source='yahoo', start=2012-1-1)['Adj Close']

log_returns=np.log(1+data.pct_change())   #pandas.pct_change() obtaion simple returns from a provided dataset
log_returns.tail()

data.plot(figsize=(10,6))
plt.show()

log_returns.plot(figsize=(10,6))
plt.show()

u=log_returns.mean()
u

var=log_returns.var()
var

drift=u-(0.5*var)
drift

stdev=log_returns.std() #Brownian motion r=drift+stdev*e^r
stdev

type(drift)

np.array(drift)

drift.values #object.values transfers the object into a numpy array
stdev.values

norm.ppf(0.95) #si un evento tiene una probabilidad de 95% de ocurrir, la distancia entre este evento y la media es 1.64 desviaciones

x=np.random.rand(10,2)
x

norm.ppf(x)

Z=norm.ppf(np.random.rand(10,2))
Z

t_intervals=1000
iterations=10

daily_returns=np.exp(drift.values+stdev.values*norm.ppf(np.random.rand(t_intervals,iterations)))
daily_returns

S0=data.iloc[-1]
S0

price_list=np.zeros_like(daily_returns)
price_list

price_list[0]=S0
price_list

for t in range(1, t_intervals):
    price_list[t]=price_list[t-1]*daily_returns[t] #St=St-1*daily_returns
price_list

plt.figure(figsize=(10,6))
plt.plot(price_list)
plt.show()

#Montecarlo - Black-Scholes-Merton

def d1(S,K,r,stdev,T):
    return(np.log(S/K)+(r+stdev**2/2)*T)/(stdev/np.sqrt(T))

def d2(S,K,r,stdev,T):
    return(np.log(S/K)+(r-stdev**2/2)*T)/(stdev/np.sqrt(T))

#Pandas Dataframe 

array_a=np.array([[3,2,1], [6,3,2]])
array_a

pd.DataFrame(array_a)
type(pd.DataFrame(array_a))

df=pd.DataFrame(array_a, columns=['Column 1', 'Column 2', 'Column 3'])
df

df=pd.DataFrame(array_a, columns=['Column 1', 'Column 2', 'Column 3'], index=['Row 1', 'Row 2'])
df

#Data Selection with Iloc and loc

data=pd.read_csv('Lending-company.csv', index_col='StringID')
lending_co_data=data.copy()
lending_co_data.head()

lending_co_data['Product']

lending_co_data.iloc[1] #the [1] is the row specifier
lending_co_data.iloc[1,3] #the second row and the 4th column

lending_co_data.iloc[1, :] #data from all columns
lending_co_data.iloc[:, 3] #data from all rows

lending_co_data.loc['LoanID_3']
lending_co_data.loc['LoanID_3', 'Region']
lending_co_data.loc[:, 'Location']
