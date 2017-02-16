import pandas as pd
import numpy as np
import datetime
from datetime import date
import matplotlib.pyplot as plt
import pandas_datareader.data as web

start = date(2014, 1, 1)
end = date.today()

portfolio = ['AAPL', 'MSFT', 'GE', 'BAC', 'VZ']
data = pd.DataFrame()
for co in portfolio:
    data[co] = web.DataReader(co, 'yahoo', start, end)['Adj Close']

(data/data.ix[0] * 100).plot()
plt.show

#Calculating returns
returns = np.log(data/data.shift(1))
returns.tail()

#Mean returns for the period (Modern portfolio theory)
#Mean variance of returns
#Since we have significant differences in performance, we have to use 252 trading days to annualize the daily returns
#historic returns
returns.mean() * 252

#Covariance matrix
#How are they moving compared to each other
returns.cov() * 252

noa = len(portfolio)
#Portfolio is a list of stocks
#there is a random module from numpy
#Assign random weights for each stock in the porfolio. numpy decides what the weights will be and should sum to precisely 100%
weights = np.random.random(noa)
weights/= np.sum(weights)
weights

#Check above that it should equal about 100%
#Verizon is basically saysing the last stock Verizon should be the one to invest in

#Now we will calculate expected portfolio return based on the weights

expected_return = np.sum(returns.mean() * weights) * 252
expected_return

#See how our portfolio will be affected with different returns
# Now we do expected variance
# np.dot is a dot is a product of two arrays. 

expected_variance = np.dot(weights.T, np.dot(returns.cov() * 252, weights))
expected_variance

#Now we calculate volatility
#Monte Carlo simulation to generate random portfolio weight
#Speed this 2500 times, for different results. Some people do 7000 or 1000 times, etc.
#Every time a number comes up, we want to save the result. 
#monte carlo returns and monte carlo volatility
mrets = []
mvols = []
for i in range(2500):
    weights = np.random.random(noa)
    weights /= np.sum(weights)
    mrets.append(np.sum(returns.mean() * weights) * 252)
    mvols.append(np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights))))

mrets = np.array(mrets)
mvols = np.array(mvols)

#Let's plot it
#represents all the 5 stocks
plt.figure()
plt.scatter(mvols, mrets, c=mrets / mvols, marker='o')
plt.grid(True)
plt.xlabel('Expected volatility')
plt.ylabel('Expected return')
plt.colorbar(label='Sharpe ratio')
plt.show()