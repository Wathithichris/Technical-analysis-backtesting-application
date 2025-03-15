import pandas as pd

def get_rf(start_date:str, end_date:str):
    df_yields = pd.read_csv('US10Y_yields.csv', parse_dates=['observation_date'])
    df_yields = df_yields[df_yields['DGS10']>=0]
    df_yields = df_yields.loc[df_yields['observation_date'].between(start_date, end_date, inclusive='both')]
    average_yield = (df_yields['DGS10'].sum() / len(df_yields))*0.01
    rounded_yield = round(average_yield, 5)
    return rounded_yield

if __name__=="__main__":
    rf = get_rf(start_date='2018-01-01', end_date='2018-12-31')
    print(rf)