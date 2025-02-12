import yfinance  as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import xlsxwriter
pd.set_option('display.width', None)


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

    def visualize(self):
        df_plot = self.data.copy()
        fig, ax = plt.subplots()
        ax.plot(df_plot['close'])
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        ax.tick_params('x', rotation=45)
        ax.set_title(f"Daily {self.ticker.title()} closing prices")
        return plt.show()


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
    def __init__(self, ticker, start_date, end_date, strategy, sma_short=None, sma_long=None, donchian_period=None):
        super().__init__(ticker=ticker, start_date=start_date, end_date=end_date)
        self.strategy = strategy
        self.sma_short = sma_short
        self.sma_long = sma_long
        self.donchian_period = donchian_period
        self.position = self.get_position()
        self.returns_data = self.calculate()
        #self.stats = self.strategy_stats()

    def get_position(self):
        if self.strategy=='SMA crossover':
            df = self.sma(self.sma_short, self.sma_long)
            df['position'] = np.nan
            df['position'] = np.where((df['delta']>0) & (df['delta_prev']<0), 1,
                                      np.where((df['delta']<0) & (df['delta_prev']>0), 0, df['position']))
            df['position'] = df['position'].ffill().fillna(0)

        elif self.strategy=='Donchian Channel':
            df = self.donchian_channel(period=self.donchian_period)
            df['position'] = np.nan
            df['position'] = np.where(df['close']>df['upper_band_prev'], 1,
                                      np.where(df['close']<df['lower_band_prev'], 0, df['position']))
            df['position'] = df['position'].ffill().fillna(0)

        return df

    def calculate(self):
        data = self.position
        data['returns'] = data['close'].div(data['close'].shift(1))
        data['strategy_returns'] = data['returns'] * data['position'].shift(1)
        data['log_returns'] = np.log(data['returns'])
        data['strategy_log_returns'] = data['log_returns'] * data['position'].shift(1)
        data['cum_returns'] = data['log_returns'].cumsum().apply(np.exp)
        data['strategy_cum_returns'] = data['strategy_log_returns'].cumsum().apply(np.exp)
        data['peak'] = data['cum_returns'].cummax()
        data['strategy_peak'] = data['strategy_cum_returns'].cummax()
        return data

    @staticmethod
    def strategy_stats(log_returns:pd.Series, risk_free_rate:float=0.02):

        stats = {}
        stats['total_returns'] = (np.exp(log_returns.sum()) -  1) * 100
        stats['annual_returns'] = (np.exp(log_returns.mean() * 252) - 1) * 100
        stats['annual_volatility'] = log_returns.std() * np.sqrt(252) * 100
        annualized_downside =  (log_returns.loc[log_returns<0].std() * np.sqrt(252)) * 100
        stats['sortino_ratio'] = (stats['annual_returns'] - risk_free_rate*100) / annualized_downside
        stats['sharpe_ratio'] = (stats['annual_returns'] - risk_free_rate*100) / stats['annual_volatility']
        cumulative_returns = log_returns.cumsum()
        peak = log_returns.cummax()
        draw_down = peak - cumulative_returns
        max_idx = draw_down.argmax()
        stats['max_dd'] = 1 - np.exp(cumulative_returns.iloc[max_idx]) / np.exp(peak.iloc[max_idx])
        strat_dd = draw_down[draw_down==0]
        strat_dd_diff = strat_dd.index[1:] - strat_dd.index[:-1]
        strat_dd_days = strat_dd_diff.map(lambda x: x.days).values
        strat_dd_days = np.hstack([strat_dd_days, (draw_down.index[-1] - strat_dd.index[-1]).days])
        stats['max_dd_duration'] = strat_dd_days.max()

        stats = {k: np.round(v, 4) if type(v) == float else v for k, v in stats.items()}
        return stats

    def plot(self):
        df_plot = self.returns_data.copy()
        fig, ax = plt.subplots()
        ax.plot(df_plot['strategy_cum_returns'], label='Strategy returns')
        ax.plot(df_plot['cum_returns'], label='Buy and hold returns')
        ax.set_xlabel('Date')
        ax.set_ylabel('Cumulative Returns')
        ax.set_title("Strategy returns vs Buy and hold")
        plt.legend()
        plt.show()

    def export(self, statistics:pd.DataFrame):
        with pd.ExcelWriter(f"{self.ticker}_backtest.xlsx",
                            engine='xlsxwriter') as writer:

            workbook = writer.book
            sheet_name = f'{self.ticker.title()}_backtest'
            stats_df = statistics
            stats_df.to_excel(writer, sheet_name=sheet_name)
            worksheet = writer.sheets[sheet_name]
            df_plot = self.returns_data
            (max_rows, max_columns) = df_plot.shape
            trades_data = self.returns_data
            trades_data.to_excel(writer, sheet_name=sheet_name, startrow=5, startcol=0)
            for i, col in enumerate(trades_data.columns):
                width = max(trades_data[col].apply(lambda x: len(str(x))).max(), len(col))
                worksheet.set_column(i, i, width)

            chart = workbook.add_chart({'type': 'line'})
            chart.add_series({
                'values': f'={sheet_name}!$P$7:$P${max_rows}',
                'categories':f'={sheet_name}!$A$7:$A${max_rows}',
                'gap':2,
                'name': 'Buy and hold'
            })
            chart.add_series({
                'values': f'={sheet_name}!$Q$7:$Q${max_rows}',
                'gap':2,
                'name': 'Strategy return'
            })
            chart.set_x_axis({'name': 'period'})
            chart.set_y_axis({'name': 'cumulative returns', 'major_gridlines':{'visible':True}})
            chart.set_legend({'position': 'top'})
            worksheet.insert_chart('U2', chart)







if __name__ =='__main__':
    data = Returns('aapl', start_date='2020-01-01',
                   end_date='2020-12-31',strategy='Donchian Channel', donchian_period=20)
    stats = pd.DataFrame(Returns.strategy_stats(data.calculate()['log_returns']), index=['Buy and hold returns'])
    stats = pd.concat([stats, pd.DataFrame(Returns.strategy_stats(data.calculate()['strategy_log_returns']),
                                           index=['Strategy returns'])])
    print(data.export(stats))




