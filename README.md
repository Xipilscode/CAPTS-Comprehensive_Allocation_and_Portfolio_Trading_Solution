# CAPTS-Comprehensive_Allocation_and_Portfolio_Trading_Solution


Build a diversified optimal portfolio using k-means, PCA and  Monte Carlo Simulation on historical closing prices.

Option 1:  Max Sharpe Ratio
Option 2:  Minimum Volatility

Asset Class Constituents based on MCS

Use finta to generate trading signals
Fit a model to make buy/sell predictions
Use Alpaca to submit orders
Rebalance the portfolio on schedule

Simple Moving Average (SMA)
Exponential Moving Average (EMA)
Weighted Moving Average (WMA)

Get Current Portfolio Position
Get Signal (Buy or Sell) from model
if position == 0 and should_buy == True
Submit buy order
if position > 0 and should_buy == False
Submit sell order
Else Hold

Schedule Weekly Portfolio Rebalancing


