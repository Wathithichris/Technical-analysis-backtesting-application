import pandas as pd

def get_rf(start_date, end_date):
    """
    Calculates the average risk-free rate using the US 10 year Treasury security for the specified period.
    Args:
        start_date(date): The first day of the backtest. Format for start_date is 'YYYY-MM-DD'.
                            If the date specified exceeds the period for which historical data is available,
                            the first date available for historical data becomes the start date.
        end_date(date): The last date of the backtest period. Format for end_date is 'YYYY-MM-DD'.

    Returns:
        float: The average rate for the period specified rounded to five decimal places.
    """
    df_yields = pd.read_csv('US10Y_yields.csv', parse_dates=['observation_date'])

    # Filter NaN values from the yields DataFrame
    df_yields = df_yields[df_yields['DGS10']>=0]

    # Filter the yields dataframe to period lying between the start and end date
    df_yields = df_yields.loc[df_yields['observation_date'].between(start_date, end_date, inclusive='both')]
    average_yield = (df_yields['DGS10'].sum() / len(df_yields))*0.01
    rounded_yield = round(average_yield, 5)
    return rounded_yield

if __name__=="__main__":
    rf = get_rf(start_date='2018-01-01', end_date='2018-12-31')
    print(rf)