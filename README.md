# CAPTS-Comprehensive_Allocation_and_Portfolio_Trading_Solution


Build a diversified optimal portfolio using k-means, PCA and  Monte Carlo Simulation on historical closing prices.
<img width="1222" alt="image" src="https://user-images.githubusercontent.com/7315911/232337512-a9ed2381-8d6d-469b-b295-0e7238446190.png">

Option 1:  Max Sharpe Ratio
Option 2:  Minimum Volatility
<img width="270" alt="image" src="https://user-images.githubusercontent.com/7315911/232337534-06746fe5-3e55-4ae7-95cf-2c7ebc4d6eaa.png">

<img width="214" alt="image" src="https://user-images.githubusercontent.com/7315911/232337539-c8bbce20-b8ce-4932-ab67-58c8befdb61d.png">

<img width="214" alt="image" src="https://user-images.githubusercontent.com/7315911/232337545-15cd9363-dfa5-402e-a0ee-d5ad7671be68.png">

Asset Class Constituents based on MCS
<img width="384" alt="image" src="https://user-images.githubusercontent.com/7315911/232337551-3cadac86-cb94-46c1-81a6-157c3520c981.png">

Use finta to generate trading signals
Fit a model to make buy/sell predictions
Use Alpaca to submit orders
Rebalance the portfolio on schedule

<img width="392" alt="image" src="https://user-images.githubusercontent.com/7315911/232337579-53784b9b-dccf-41fe-8b6d-88110d08598c.png">

Simple Moving Average (SMA)
Exponential Moving Average (EMA)
Weighted Moving Average (WMA)
<img width="422" alt="image" src="https://user-images.githubusercontent.com/7315911/232337591-44f168e6-cd54-4a13-b192-05762ff4015c.png">

Trading Algorithm<img width="465" alt="image" src="https://user-images.githubusercontent.com/7315911/232337606-353e6712-10ec-4583-b31e-f92be83c8573.png">

Get Current Portfolio Position
Get Signal (Buy or Sell) from model
if position == 0 and should_buy == True
Submit buy order
if position > 0 and should_buy == False
Submit sell order
Else Hold
Schedule Weekly Portfolio Rebalancing
<img width="461" alt="image" src="https://user-images.githubusercontent.com/7315911/232337611-9b35c9fb-e4e6-4195-8abe-7a8d1ddcc4d5.png">

