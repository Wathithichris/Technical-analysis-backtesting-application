# Technical analysis backtesting app
## Overview
Technical analysis is the study of market action primarily through use of charts to predict future price trends. 
There are two popular approaches when backtesting a trading strategy using historical data:
- manual backtesting 
- automated backtesting.

Backtesting a strategy is simply analyzing historical performance of a security to try and 
develop a hypothesis of how a similar future event might play out.

Manual backtesting involves looking back into the historical data, and replaying the scenario as though it is the
first time the markets are unfolding before your eyes. This process is often time-consuming, 
clouded by subjectivity and personal bias and may lead to ultimately drawing the wrong conclusions. 

Automated backtesting seek to leverage the power of technology, to quantify technical analysis strategies.
This reduces the time taken and the biases associated with manual backtesting. 

This application uses the latter approach to arrive at the results.

![Alt text](top.png)

## Data
In backtesting the quality of the data plays a crucial role as having poor quality data will lead to biased results 
that may be misleading. For this particular project historical data is sourced using _**yfinance__* a python library 
that allows users to source data from Yahoo finance API. For more information about the library please visit
https://yfinance-python.org. 

The interval for the historical data is daily since daily closing price data is easily available,
as compared to intraday data that may have data limits on how far back the data can go.

*The api  has a limit request of 10,000 requests a day so be mindful of others while doing your analysis.*

### Historical data
The historical data for a particular security is obtained using a ticker symbol. 
This is a unique identifier of a security in the financial markets eg aapl is the ticker symbol for Apple Inc.

You can find ticker symbols of securities by searching for the name of the company in https://finance.yahoo.com.


## Date ranges
- The start date: This is when the backtest should begin.
- The end date: This is when the backtest should end.

*If the start date exceeds the earliest historical date, the earliest historical date is the one returned by default.*

*The maximum dates are set to the current day.*

## Strategies
### 1. Simple moving average crossover(SMA)
This is among the most popular technical indicators because of how easily the trading logic can be quantified.

The buy signal is initiated when the short moving average crosses above the long moving average and a sell signal is 
generated when the small moving average crosses below the long moving average.
 
The equation for the SMA is simply the average closing price of a security over the last 'n' periods:
$$ SMA=\frac{p1&plus;p2&plus;p3&plus;...pn}{n} $$

### 2. Donchian channel
This is a popular trend-following technical indicator.
The calculations are as follows:
- Upper band: The highest closing price in prior 'n' periods
- Lower band: The lowest closing price in prior 'n' periods
- Middle band: This is the average of the upper and lower band

The buy signal is initiated when the current closing price exceeds the previous upper band level and 
the sell signal is initiated when the current closing price is below the previous lower band.

### 3. Exponential moving average(EMA)
This moving average uses exponentially weighted closing prices to calculate the moving averages.

The EMA formula involves:
- current price(p): the price at the current time.
- Previous EMA($EMA~previous$):The EMA calculated at the previous price.
- Smoothing constant($\alpha$): a constant derived from the number of periods. it is calculated as $$\alpha= 
\frac{2}{N+1}$$

The formula is as follows:
$$ EMA~current = (p \times \alpha)+(EMA~previous\times (1 - \alpha)) $$

The EMA unlike the SMA is  more sensitive to most recent price changes. 

## Backtest results
![Alt text](bottom.png)

Once the backtest is complete a line plot of the closing prices of the security is plotted against the time 
selected for the backtest.
Also accompanying the line plot, is a dataframe of the backtest summary statistics.

### Understanding trade statistics
- total_returns: this is the total cumulative returns of the strategy over the period.
- annual_returns: this is the average annualized returns of the security over the period.
- annual_volatility: this is the standard deviation of the trading strategy's annual returns.
- max_dd: this is a measure of the largest peak-to-trough decline over the backtest period.
- max_dd_duration: this is a measure of how long it took for a strategy to recover from the maximum draw-down period.

Sharpe ratio is computed using the US 10Y government bond yields as the risk-free rate. 
The data for the US 10Y bond yield  was sourced from https://fred.stlouisfed.org/ and is included in the files.

*Since the Sharpe ratio incorporates the US 10-year government yield, the strategy's statistics are more comparable,
when the back tested financial instrument is traded in the US securities markets.*

## Download backtest results
The download backtest result allows you to download a more extensive backtest results dataset.
The results are in form of an Excel workbook with two worksheets:
- Backtest_statistics: contains the summary of the backtest statistics. 
It also has equity curves of both strategies for visual aid.
- Backtest_data: contains the ohlc data of the stock and returns columns.This is for sanity checking the strategy logic.

## Assumptions
- The trading strategies are long-only strategies: the sell signal is an exit signal and 
until the long signal is triggered again, the investor is effectively out of the market.
- There are no transaction fees and slippage during entry and exit.
