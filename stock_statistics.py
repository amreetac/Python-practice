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