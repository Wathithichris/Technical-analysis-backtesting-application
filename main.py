import streamlit as st
import pandas as pd
import plotly.express as px
from backend import Returns
from treasury_yield import get_rf
from datetime import datetime

st.set_page_config(layout='wide')
# Create title
st.title("Stock Technical Analysis Backtester")
st.write("""This app is to backtest popular technical analysis strategies in the 
            stock market to get a feel of how they performed historically.""")

# Create web app body
ticker = st.text_input("Stock ticker")
ticker = ticker.strip().lower()

start_date = st.date_input("Start date", key='start', format='YYYY-MM-DD')
end_date = st.date_input("End date", key='end', format='YYYY-MM-DD')

# Create datetime objects from input
# start_date_str = start_date
# end_date_str = end_date
# date_format = '%Y-%m-%d'
# start_date_obj = datetime.strptime(start_date_str, date_format)
# end_date_obj = datetime.strptime(end_date_str, date_format)



strategy = st.selectbox("Select strategy", ('SMA crossover', 'Donchian Channel'))

if strategy=="SMA crossover":
    sma1 = int(st.number_input("Sma 1", key='sma1', format='%f'))
    sma2 = int(st.number_input("Sma 2", key='sma2', format="%f"))

elif strategy=='Donchian Channel':
    period = st.number_input("Channel Period", key='period', format="%0f")


backtest = st.button(label="Run backtest")

if backtest:
    st.subheader("Backtest results")
    data = Returns(ticker=ticker, start_date=start_date, end_date=end_date, strategy=strategy,
                   sma_long=sma1, sma_short=sma2)

    #rf = get_rf(start_date=start_date, end_date=end_date)
    results = pd.DataFrame(Returns.strategy_stats(data.calculate()['log_returns']), index=['Buy_and_hold'])
    results = pd.concat([results, pd.DataFrame(Returns.strategy_stats(data.calculate()['strategy_log_returns']),
                         index=['Strategy_returns'])])
    # Display the chart
    st.text(results)


    # Create download option
    st.download_button("Download results csv", data='homepage.png')



