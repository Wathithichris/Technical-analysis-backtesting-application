import yfinance  as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


class Data:
    def __init__(self, ticker, start_date, end_date):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.data = self.get()

    def get(self):
        data = yf.download(tickers=self.ticker, start=self.start_date, end=self.end_date, multi_level_index=False)
        data.columns = ['close', 'high', 'low', 'open', 'volume']
        return data

    def process_data(self):
        data = self.data.copy()
        data.drop(['volume'],axis=1,  inplace=True)
        return data



class TechnicalIndicators(Data):
    def sma(self, sma_short:int, sma_long:int):
        df = self.process_data()
        df[f'sma_{sma_short}'] = df['close'].rolling(window=sma_short).mean()
        df[f'sma_{sma_long}'] = df['close'].rolling(window=sma_long).mean()
        df['delta'] = df[f'sma_{sma_short}'] - df[f'sma_{sma_long}']
        df['delta_prev'] = df['delta'].shift(1)
        return df

    def donchian_channel(self, period):
        df = self.process_data().copy()
        df['upper_band'] = df['close'].rolling(window=period).max()
        df['upper_band_prev'] = df['upper_band'].shift(1)
        df['lower_band'] = df['close'].rolling(window=period).min()
        df['lower_band_prev'] = df['lower_band'].shift(1)
        df['mid_band'] = (df['upper_band']+df['lower_band'])/2
        return df


class Returns(TechnicalIndicators):
    def get_position(self, strategy):
        if strategy=='SMA crossover':
            df = self.sma(sma_short=20, sma_long=50)
            df['position'] = np.nan
            df['position'] = np.where((df['delta']>0) & (df['delta_prev']<0), 1,
                                      np.where((df['delta']<0) & (df['delta_prev']>0), 0, df['position']))
            df['position'] = df['position'].ffill().fillna(0)

        elif strategy=='Donchian Channel':
            df = self.donchian_channel(period=20)
            df['position'] = np.nan
            df['position'] = np.where(df['close']>df['upper_band_prev'], 1,
                                      np.where(df['close']<df['lower_band_prev'], 0, df['position']))
            df['position'] = df['position'].ffill().fillna(0)

        return df

    def calculate(self):
        data = self.get_position()

    def visualise(self):
        pass

    def export(self):
        pass






data = Returns('aapl', start_date='2020-01-01', end_date='2020-12-31')
print(data.sma(sma_short=20, sma_long=50))
print(data.get_position(strategy='Donchian Channel'))


