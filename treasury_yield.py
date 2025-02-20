import pandas as pd
import numpy as np
from datetime import datetime


def get_rf(start_date:int, end_date:int):
    df_yields = pd.read_csv('US10Y_yields.csv', parse_dates=['observation_date'])
    df_yields['year'] = df_yields['observation_date'].dt.year
    df_yields = df_yields.loc[df_yields['year'].between(start_date, end_date, inclusive='both')]
    df_yields_annual = df_yields.groupby(by=['year'])['DGS10'].mean()
    average_yield = round(df_yields_annual.sum() / len(df_yields_annual), 4)*0.01
    return average_yield

rf = get_rf(2000, 2000)
print(rf)