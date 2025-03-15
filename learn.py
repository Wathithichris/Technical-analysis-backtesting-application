import streamlit as st
import pandas as pd
import plotly.express as px
from backend import Returns
from treasury_yield import get_rf
from datetime import datetime


# Page layout
st.set_page_config(layout='wide')

# Create title widget and give overview of what app does
st.title("Stock Technical Analysis Backtester")
st.write("""This app is to backtest popular technical analysis strategies in the 
            stock market to get a feel of how they performed historically.""")

# Create ticker input widget to get ticker to pull date with
ticker = st.text_input("Stock ticker")
ticker = ticker.strip().lower()


# Create start and end date input widgets
today = datetime.today()

start_date = st.date_input("Start date", key='start', format='YYYY-MM-DD', max_value=today)
end_date = st.date_input("End date", key='end', format='YYYY-MM-DD', max_value=today)


# Create strategy select-box to select from different technical analysis strategies
strategy = st.selectbox("Select strategy", ('SMA crossover', 'Donchian Channel', 'EMA crossover'))

moving_averages = ['SMA crossover', 'EMA crossover']
# Create input widgets based on the strategy
if strategy in moving_averages:
    short_ma = int(st.number_input("short moving average", key='ma_s', format='%f'))
    long_ma = int(st.number_input("long moving average", key='ma_l', format="%f"))

    if short_ma != long_ma:
        if short_ma >= long_ma:
            st.warning("short moving average must be less than long moving average", icon=":material/error:")
    else:
        st.warning("The moving averages cannot be equal", icon=":material/error:")

if strategy == 'Donchian Channel':
    period = int(st.number_input("Channel Period", key='period', format="%0f"))

# Create backtest button
backtest = st.button(label="Run backtest")

# Carry out backtest
st.subheader("Backtest results")

if backtest:

    if strategy in moving_averages:
        if strategy == 'SMA crossover':
            data = Returns(ticker=ticker, start_date=start_date, end_date=end_date, strategy=strategy,
                       sma_long=long_ma, sma_short=short_ma)
        elif strategy == 'EMA crossover':
            data = Returns(ticker=ticker, start_date=start_date, end_date=end_date, strategy=strategy,
                       ema_long=long_ma, ema_short=short_ma)

    elif strategy=="Donchian Channel":
        data = Returns(ticker=ticker, start_date=start_date, end_date=end_date, strategy=strategy,
                       donchian_period=period)
    try:
        stats = None
        if data:
            stats = pd.DataFrame(Returns.strategy_stats(data.calculate()['log_returns']), index=['Buy_and_hold'])
            stats = pd.concat([stats, pd.DataFrame(Returns.strategy_stats(data.calculate()['strategy_log_returns']),
                                                   index=['Strategy_returns'])])

    except Exception as e:
        match e:
            case ValueError():
                st.write("Oops!, please enter a valid ticker")

    else:
        # Create figure object
        figure = data.visualize()

        # Visualize closing prices of the stock
        st.plotly_chart(figure_or_data=figure, height=1500, width=600)

        # Output stats dataframe
        st.dataframe(data=stats)

        # Get extensive backtest
        results = data.export(stats)

        # Download extensive results
        results_csv = results.to_csv(index=False).encode('utf-8').copy()
        st.download_button(label="Download results csv", data=results_csv,
                           file_name=f"{ticker}_{strategy}_backtest.csv")


