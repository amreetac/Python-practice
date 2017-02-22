#Import important libraries. We will be performing statistics on Stock information

import pandas as pd
import numpy as np
import datetime 
from datetime import date
import matplotlib.pyplot as plt
import pandas_datareader.data as web

#Create a time range

start = date(2016, 9, 1)
end = date.today()

#Read data using data reader. Use yahoo or csv file from Yahoo Finance or similar resource. Assign to a variable stock and print the last 2 samples of data

data = pd.DataFrame()
data = web.DataReader("MSFT", 'yahoo', start, end)
data.tail(2)

#Sort Volume in Descending order
data.sort_values(['Volume'], ascending=False).head()

# minimum Low price during that period

data['Low'].min()

#Maximum high price
data['High'].max()

#Difference between Maximum Adj Close and the Minimum Adj Close

diff = data['Close'].max() - data['Close'].min()
diff

#Plot the 'Adj Close' value. 

data['Adj Close'].plot()
plt.show()

#Calculating log returns (use Adj Close)

returns = np.log(data['Adj Close']/data['Adj Close'].shift(1))
returns.tail()

#Calculate Mean-variance of returns

returns.mean() * 252

#Adding another stock

stock = web.DataReader('TWTR', 'yahoo', start, end)
stock.info()
stock.tail()

#Create a new column in stock

stock['Return']= 0.0
stock

#Calculating return
stock['Return'] = np.log(stock['Adj Close']/stock['Adj Close'].shift(1))

#Want to do some modern portfolio theory

portfolio = ["AAPL","MSFT","GE","BAC", "VZ"]
data = pd.DataFrame()
for co in portfolio:
    data[co] = web.DataReader(co, 'yahoo', start, end)["Adj Close"]

# want to data from datareader
(data/data.ix[0] * 100).plot()
plt.show()

#Calculating returns
returns = np.log(data/data.shift(1))
returns.tail()

#Mean-variance of returns
#Since we have significant differences in performance, 
#we have to use 252 trading days to annualize the daily returns 
returns.mean() * 252

#Covariance
returns.cov() * 252

#We assume that we do not open short position and we divide our money equally divided among 5 stocks
#So we generate 5 random numbers and then normalize them so that values would sum up 100% net oper assets
noa = len(portfolio)
weights = np.random.random(noa)
weights /= np.sum(weights)
weights

#Calculating Expected portfolio return based on the weights
expected_return = np.sum(returns.mean() * weights) *252
expected_return

#Now lets calculate Expected portfolio variance using our covariance matrix
#we use np.dot -  gets us a product of two matrices
expected_variance = np.dot(weights.T, np.dot(returns.cov() * 252, weights))
expected_variance

#Now we calculate expected standard deviation or volatility 

volatility = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
volatility

#Monte Carlo simulation to generate random portfolio weight vectors on larger scale
#For every simulated allocation we record the resulting portfolio return and variance
#We assume Risk free is 0
mrets = []
mvols = []
for i in range(2500):
    weights = np.random.random(noa)
    weights /= np.sum(weights)
    mrets.append(np.sum(returns.mean() * weights) * 252)
    mvols.append(np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights ))))

mrets = np.array(mrets)
mvols = np.array(mvols)

#Lets plot it
plt.figure()
plt.scatter(mvols, mrets, c=mrets / mvols, marker='o')
plt.grid(True)
plt.xlabel('Expected volatility')
plt.ylabel('Expected return')
plt.colorbar(label="Sharpe ratio")
plt.show()

#Lets plot it
plt.figure()
plt.scatter(mvols, mrets, c=mrets / mvols, marker='o')
plt.grid(True)
plt.xlabel('Expected volatility')
plt.ylabel('Expected return')
plt.colorbar(label="Sharpe ratio")
plt.show()

#Resampling

# creating object stock_price
stock_price = web.DataReader("TWTR", "yahoo", start, end)
# resampling data to 10days period
ten_days = stock_price['Adj Close'].resample('10D').ohlc()
stock_volume = stock_price['Volume'].resample('10D').sum()
stock_volume.tail()
# resetting index for OHLC
ten_days.reset_index(inplace=True)

# have to create new datetime objects for matplotlib
ten_days["Date"] = ten_days["Date"].map(mdates.date2num)

#Make sure to have the proper libraries

import pandas as pd 
import pandas_datareader.data as web
import matplotlib.pyplot as plt 
from matplotlib.finance import candlestick_ohlc
import matplotlib.dates as mdates

# resetting index for OHLC
ten_days.reset_index(inplace=True)
# have to create new datetime objects for matplotlib
ten_days["Date"] = ten_days["Date"].map(mdates.date2num)
# creating two ficures
price_fig = plt.subplot2grid((6,1),(0,0), rowspan=5, colspan=1)
volume_fig = plt.subplot2grid((6,1),(5,0), rowspan=1, colspan=1, sharex=price_fig)
# plotting data
price_fig.xaxis_date()
candlestick_ohlc(price_fig, ten_days.values, width=5)
volume_fig.fill_between(stock_volume.index.map(mdates.date2num), stock_volume.values, 0)

plt.show()


#Histogram

# creating new variable with data frame data structure
data = pd.DataFrame()

# defining start and end dates
start = date(2016, 1, 1)
end = date.today()

# using for loop to get historic prices for stocks
for ticker in symbols:
    data[ticker] = web.DataReader(ticker, 'yahoo', start, end)["Adj Close"]

# checking data
print(data.tail())

#We use 100 as a starting value
#Using ix as a primarily label-location based indexer
# (data/data.ix[0] * 100).plot()
# plt.show()

#Calculating log returns
log_returns = np.log(data/data.shift(1))
print(log_returns.tail())

# plotting histograms for log returns
log_returns.hist(bins=50)
plt.show()