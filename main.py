import streamlit as st
import pandas as pd
import plotly.express as px


st.set_page_config(layout='wide')
# Create title
st.title("Stock Technical Analysis Backtester")
st.write("""This app is to backtest popular technical analysis strategies in the 
            stock market to get a feel of how they performed historically.""")

# Create web app body
ticker = st.text_input("Stock ticker")
col1, col2 = st.columns(spec=2)
col1.date_input("Start date", key='start')
col2.date_input("End date", key='end')

strategy = st.selectbox("Select strategy", ('SMA crossover', 'Donchian Channel'))

if strategy=="SMA crossover":
    col3, col4 = st.columns(spec=2)
    col3.text_input("Sma 1", key='sma1')
    col4.text_input("Sma 2", key='sma2')
elif strategy=='Donchian Channel':
    st.text_input("Channel Period")

# Display the chart
st.subheader("Backtest results")
dates = ['2022-25-10', '2022-26-10', '2022-27-10']
temperatures = [10, 11, 15]
figure = px.line(x=dates, y=temperatures, labels={"x":'Date', "y":"Temperature (C)"})
col5, col6 = st.columns(spec=2)
col5.plotly_chart(figure)
col6.text("Performance metrics")

# Create download option
st.download_button("Download results as csv", data='homepage.png')



