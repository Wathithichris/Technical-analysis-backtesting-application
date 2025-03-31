import yfinance  as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from requests import Session
from requests_cache import CacheMixin, SQLiteCache
from requests_ratelimiter import LimiterMixin, MemoryQueueBucket
from pyrate_limiter import Duration, RequestRate, Limiter
pd.set_option('display.width', None)
from treasury_yield import get_rf
class CachedLimiterSession(CacheMixin, LimiterMixin, Session):
    pass


class Data:
    """
    Represents the historical data object.

    Attributes:
        ticker: A str symbol representing a particular stock.
        start_date: A date object representing the first date of the backtest period.
        end_date: A date object representing the last date of the backtest period.

    """
    def __init__(self, ticker, start_date, end_date):
        """
        Initializes the instance based on specified arguments.
        Args:
            ticker(str): A unique symbol assigned that represents a particular stock listed on an exchange.
            start_date(date): A date object representing when actual backtesting should begin.
                            If the date is out of historical range, the first date recorded becomes the start_date.
            end_date(date): A date object representing when the backtest should end.
        """
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.data = self.get()

    def get(self):
        """
        Retrieves daily  historical data associated with a particular ticker symbol from Yahoo Finance API.
        Returns:
            Pandas DataFrame object: The historical dataframe for the symbol over the specified period.
        """
        # Catching rate-limiter to avoid corrupting the data
        session = CachedLimiterSession(
            limiter=Limiter(RequestRate(8000, Duration.DAY*1)),
            bucket_class=MemoryQueueBucket,
            backend=SQLiteCache('yfinance.cache'))
        session.headers['User-agent'] = 'my_program/1.0'
        ticker = yf.Ticker(self.ticker, session=session)
        # Get the data
        data = ticker.history(start=self.start_date, end=self.end_date)
        data = data.iloc[:, :4]
        # Rename the columns
        data.columns = ['open', 'high', 'low', 'close']
        data.reset_index(inplace=True)#  This allows you to access the Date column which was previously the index
        # Convert Date column into a date object for ease when plotting later on
        data['Date'] = pd.to_datetime(data['Date']).dt.date
        data.set_index(['Date'], inplace=True)
        return data

    def visualize(self):
        """
        Creates a matplotlib line chart of the closing prices of the data.
        Returns:
            matplotlib figure object: A line chart of the closing prices of the data against the date.
        """
        df_plot = self.data.copy()
        fig, ax = plt.subplots(figsize=(10,8))
        ax.plot(df_plot['close'])
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        ax.tick_params('x', rotation=45)
        ax.set_title(f"Daily {self.ticker.title()} closing prices")
        return fig


class TechnicalIndicators(Data):
    """
    Represents Technical indicators, extending the Data class to include some of the most popular technical indicators.
    """
    def sma(self, sma_short:int, sma_long:int):
        """
        Calculates the Simple Moving Average(sma) of the closing prices over a given period.
        Args:
            sma_short: The n period for the short moving average.
            sma_long: The n  period for the longer moving average.

        Returns:
            DataFrame: A DataFrame that contains sma short, sma long, delta and delta previous columns.
        """
        df = self.data.copy()
        df[f'sma_{sma_short}'] = df['close'].rolling(window=sma_short).mean()
        df[f'sma_{sma_long}'] = df['close'].rolling(window=sma_long).mean()
        # The difference between the short and the long moving average to be used later for entry and exit signals
        df['delta'] = df[f'sma_{sma_short}'] - df[f'sma_{sma_long}']
        df['delta_prev'] = df['delta'].shift(1)
        return df

    def donchian_channel(self, period:int):
        """
        Calculates donchian channels which include an upper band,
        a lower band and a middle band based on the specified n period.
        Args:
            period:  The number of days over which to consider the highest high and the lowest low.

        Returns:
            DataFrame: A DataFrame which contains upper_band, upper_band_prev, lower_band,
                        lower_band_prev and mid_band columns.
        """
        df = self.data.copy()
        df['upper_band'] = df['close'].rolling(window=period).max()
        df['upper_band_prev'] = df['upper_band'].shift(1)
        df['lower_band'] = df['close'].rolling(window=period).min()
        df['lower_band_prev'] = df['lower_band'].shift(1)
        df['mid_band'] = (df['upper_band']+df['lower_band'])/2
        return df

    def ema(self, ema_short:int, ema_long:int):
        """
        Calculates the Exponential weighted Moving average(ema) of the closing prices.
        Args:
            ema_short: The n period for the short ema.
            ema_long: The n period for the long ema.

        Returns:
            DataFrame: A DataFrame that contains short ema, long ema, delta and delta_prev columns.
        """
        df = self.data.copy()
        df[f"ema_{ema_short}"] = df['close'].ewm(span=ema_short, adjust=False, min_periods=ema_short).mean()
        df[f"ema_{ema_long}"] = df['close'].ewm(span=ema_long, adjust=False, min_periods=ema_long).mean()
        # The difference between the short and the long ema which will be used to for entry and exit logic
        df['delta'] = df[f"ema_{ema_short}"] - df[f"ema_{ema_long}"]
        df['delta_prev'] = df['delta'].shift(1)
        return df


class Returns(TechnicalIndicators):
    """
    Represents Returns. It extends the TechnicalIndicators class to include returns data.
    """
    def __init__(self, ticker, start_date, end_date, strategy, **kwargs):
        """
        Extends the Data initialize method to include strategy and **kwargs.
        Args:
            ticker(str): A unique identifier for a listed security.
            start_date(date): The date when backtest commences. The format for date is "YYYY-MM-DD".
            end_date(date): The date when the backtest ends. The format for date is "YYYY-MM-DD".
            strategy(str): The particular strategy selected from the list of strategies.
            **kwargs: An arbitrary keyword argument with int values.
        """
        super().__init__(ticker=ticker, start_date=start_date, end_date=end_date)
        self.strategy = strategy
        self.sma_short = kwargs.get('sma_short')
        self.sma_long = kwargs.get('sma_long')
        self.ema_short = kwargs.get('ema_short')
        self.ema_long = kwargs.get('ema_long')
        self.donchian_period = kwargs.get('donchian_period')
        self.position = self.get_position()
        self.returns_data = self.calculate()

    def get_position(self):
        """
        Retrieves the trade position for a particular strategy.
        Returns:
              DataFrame: A DataFrame that includes the trade position based on a given strategy.
        Notes:
              These strategies are long only and do not include short selling.
              To implement the short selling, you need to extend the positions logic to have a sell signal.
              For the case of this project, we are assuming you are buying a security when signal changes from 0 to 1
              and selling your security when signal changes from 1 to 0.
        """
        moving_averages = ['SMA crossover', 'EMA crossover']

        if self.strategy=='SMA crossover':
            # Create a DataFrame using the sma technical indicator
            df = self.sma(self.sma_short, self.sma_long)
            df['position'] = np.nan
            # For moving average strategies buy when delta changes from 0 to 1 and exit when inverse happens
            df['position'] = np.where((df['delta'] > 0) & (df['delta_prev'] < 0), 1,
                                      np.where((df['delta'] < 0) & (df['delta_prev'] > 0), 0, df['position']))
            df['position'] = df['position'].ffill().fillna(0)

        elif self.strategy=='EMA crossover':
            # Create a DataFrame using the ema technical indicator
            df = self.ema(self.ema_short, self.ema_long)

            df['position'] = np.nan
            # For moving average strategies buy when delta changes from 0 to 1 and exit when inverse happens
            df['position'] = np.where((df['delta']>0) & (df['delta_prev']<0), 1,
                                      np.where((df['delta']<0) & (df['delta_prev']>0), 0, df['position']))
            df['position'] = df['position'].ffill().fillna(0)

        elif self.strategy=='Donchian Channel':
            df =  self.donchian_channel(period=self.donchian_period)
            df['position'] = np.nan
            # Enter long when closing price is greater than the previous upper band
            # Exit long when closing price is below the previous lower band
            df['position'] = np.where(df['close']>df['upper_band_prev'], 1,
                                      np.where(df['close']<df['lower_band_prev'], 0, df['position']))
            df['position'] = df['position'].ffill().fillna(0)

        return df

    def calculate(self):
        """
        Calculates the returns of the particular strategy using the trade position.
        Returns:
            DataFrame: A DataFrame that includes the strategy's return data.
        Notes:
            Logarithmic returns measure the relative change in value of an asset.
            Logarithmic returns take account of the compounding effect of returns over time,
            making it a more accurate measure of percentage change in value of an asset.
        """
        data = self.position
        # Get the daily percentage change in closing prices
        data['returns'] = data['close'].div(data['close'].shift(1))
        data['log_returns'] = np.log(data['returns'])
        data['strategy_returns'] = data['returns'] * data['position'].shift(1)
        data['strategy_log_returns'] = data['log_returns'] * data['position'].shift(1)
        # Represents buy and hold equity curve over the period
        data['cum_returns'] = data['log_returns'].cumsum().apply(np.exp)
        # Represents the strategy's equity curve over the period
        data['strategy_cum_returns'] = data['strategy_log_returns'].cumsum().apply(np.exp)
        # Get the highest cumulative return for buy and hold strategy
        data['peak'] = data['cum_returns'].cummax()
        # Get the highest cumulative return for the strategy
        data['strategy_peak'] = data['strategy_cum_returns'].cummax()
        return data

    @staticmethod
    def strategy_stats(log_returns:pd.Series, risk_free_rate:float=0.03284):
        """
        Retrieves the trade statistics for a particular strategy to better assess its performance.
        Args:
            log_returns: A pandas Series object with the logarithmic returns.
            risk_free_rate: The average of the 10-Year US Treasury securities over the given period.

        Returns:
            dict: A python dictionary with the strategy's statistics.

        """
        stats = dict()
        stats['total_returns%'] = (np.exp(log_returns.sum()) -  1) * 100
        stats['annual_returns%'] = (np.exp(log_returns.mean() * 252) - 1) * 100
        stats['annual_volatility%'] = log_returns.std() * np.sqrt(252) * 100
        annualized_downside =  (log_returns.loc[log_returns<0].std() * np.sqrt(252)) * 100
        stats['sortino_ratio'] = (stats['annual_returns%'] - risk_free_rate*100) / annualized_downside
        stats['sharpe_ratio'] = (stats['annual_returns%'] - risk_free_rate*100) / stats['annual_volatility%']
        cumulative_returns = log_returns.cumsum()
        peak = log_returns.cummax()
        draw_down = peak - cumulative_returns
        max_idx = draw_down.argmax()
        stats['max_dd%'] = (1 - (np.exp(cumulative_returns.iloc[max_idx]) / np.exp(peak.iloc[max_idx])))*100
        strat_dd = draw_down[draw_down==0]
        strat_dd_diff = strat_dd.index[1:] - strat_dd.index[:-1]
        strat_dd_days = strat_dd_diff.map(lambda x: x.days).values
        strat_dd_days = np.hstack([strat_dd_days, (draw_down.index[-1] - strat_dd.index[-1]).days])
        stats['max_dd_duration'] =f"{strat_dd_days.max()} days"

        stats = {k: np.round(v, 4) if type(v) == float else v for k, v in stats.items()}
        return stats

    def export(self, statistics: pd.DataFrame, returns_data: pd.DataFrame):
        """
        Exports the trade statistics and returns data into an Excel file.
        Args:
            statistics: A DataFrame of the strategy statistics.
            returns_data: A DataFrame of the strategy returns.

        Returns:
            Excel file: A file containing all the trade statistics and returns data of a particular strategy.

        """
        file_name = f"{self.ticker}_backtest.xlsx"
        with pd.ExcelWriter(file_name, engine='xlsxwriter') as writer:
            workbook = writer.book
            sheet1 = 'Backtest_statistics'
            sheet2 = 'Backtest_data'

            # Write trade statistics DataFrame to Sheet 1
            statistics.to_excel(writer, sheet_name=sheet1)

            # Validate returns_data
            if returns_data is None or returns_data.empty:
                st.warning("No trade data available!")
                return None

            # Write returns data to Sheet 2
            returns_data.to_excel(writer, sheet_name=sheet2)

            # Get worksheet references
            worksheet1 = writer.sheets[sheet1]
            worksheet2 = writer.sheets[sheet2]

            max_row, max_col = returns_data.shape

            # Auto-adjust column width for both the Excel sheets
            for sheet in [worksheet1, worksheet2]:
                for i, col in enumerate(returns_data.columns):
                    col_width = max(returns_data[col].astype(str).apply(len).max(), len(col)) + 2
                    sheet.set_column(i, i, col_width)

            # Dynamically find column positions
            returns_data.reset_index(inplace=True)#  To get a Date column
            headers = returns_data.columns.tolist()
            date_col = headers.index('Date')
            buy_hold_col = headers.index('cum_returns')
            strategy_col = headers.index('strategy_cum_returns')

            # Convert column indexes to Excel column letters
            col_num_to_letters = (lambda col_num: ""if col_num < 0 else col_num_to_letters(col_num // 26 - 1) +
                                                                        chr( col_num % 26 + 65))

            date_col_letter = col_num_to_letters(date_col)
            buy_hold_col_letter = col_num_to_letters(buy_hold_col)
            strategy_col_letter = col_num_to_letters(strategy_col)

            # Create a chart
            chart = workbook.add_chart({'type': 'line'})
            chart.add_series({
                'name': 'Buy and Hold',
                'categories': f"={sheet2}!${date_col_letter}$2:${date_col_letter}${max_row}",
                'values': f"={sheet2}!${buy_hold_col_letter}$2:${buy_hold_col_letter}${max_row}",
                'gap': 2
            })
            chart.add_series({
                'name': 'Strategy Return',
                'values': f"={sheet2}!${strategy_col_letter}$2:${strategy_col_letter}${max_row}",
                'gap': 2
            })

            # Set chart properties
            chart.set_x_axis({'name': 'Period'})
            chart.set_y_axis({'name': 'Cumulative Returns', 'major_gridlines': {'visible': True}})
            chart.set_legend({'position': 'top'})

            # Insert the chart in Sheet1
            worksheet1.insert_chart('A10', chart)

        return file_name


if __name__ =='__main__':
    # Date parameters
    start = "2018-01-01"
    end = '2024-12-31'

    # Get returns data
    data = Returns("BAC", start_date=start,
                   end_date=end,strategy='Donchian Channel', donchian_period=20)

    st = pd.to_datetime(data.data.iloc[0].name)
    ed = pd.to_datetime(data.data.iloc[-1].name)

    # Get the risk-free rate for the period
    rf = get_rf(start_date=st, end_date=ed)

    # Get strategy statistics
    stats = pd.DataFrame(Returns.strategy_stats(data.calculate()['log_returns'], risk_free_rate=rf),
                         index=['Buy and hold returns'])
    stats = pd.concat([stats, pd.DataFrame(Returns.strategy_stats(data.calculate()['strategy_log_returns']),
                                           index=['Strategy returns'])])
    print(data.data)
    print(data.calculate())
    print(stats)
    print(data.export(stats, data.calculate()))




