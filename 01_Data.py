
from finrl.config import INDICATORS
from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split



import alpaca_trade_api as tradeapi
from alpaca_trade_api.rest import TimeFrame

import numpy as np
import pandas as pd

import exchange_calendars as tc
import pytz
import itertools


def download_data(
    ticker_list, start_date, end_date, time_interval, API_KEY=None, API_SECRET=None, API_BASE_URL=None
) -> pd.DataFrame:

    
    api = tradeapi.REST(API_KEY, API_SECRET, API_BASE_URL, "v2")

    start = start_date
    end = end_date
    time_interval = time_interval

    data_df = pd.DataFrame()

    # download
    NY = "America/New_York"
    start_date = pd.Timestamp(start_date + " 09:30:00", tz=NY)
    end_date = pd.Timestamp(end_date + " 15:59:00", tz=NY)
    barset = api.get_bars(
        ticker_list,
        time_interval,
        start=start_date.isoformat(),
        end=end_date.isoformat(),
    ).df
    

    # barset response has the timestamps in UTC - Convert to NY timezone
    barset = barset.reset_index()
    
    barset["timestamp"] = barset["timestamp"].apply(lambda x: x.tz_convert(NY))
    barset = barset.set_index("timestamp")

    # from trepan.api import debug;debug()
    # filter opening time of the New York Stock Exchange (NYSE) (from 9:30 am to 4:00 pm) if time_interval < 1D
    if time_interval != '1D':
        NYSE_open_hour = "09:30"  # in NY
        NYSE_close_hour = "15:59"  # in NY
        data_df = barset.between_time(NYSE_open_hour, NYSE_close_hour)
    else:
        data_df = barset
        

    data_df = data_df.reset_index()
    data_df.columns = [
            "date",
            "open",
            "high",
            "low",
            "close",
            "volume",
            'trade_count', 
            'vwap',
            "tic",
        ]
    
    data_df["day"] = data_df["date"].dt.dayofweek
    data_df["date"] = data_df.date.apply(lambda x: x.strftime("%Y-%m-%d"))

    
    data_df = data_df.dropna()
    data_df = data_df.reset_index(drop=True)

    data_df = data_df.sort_values(by=["date", "tic"]).reset_index(drop=True)        

    

    return data_df


TRAIN_START_DATE = '2005-01-01'
TRAIN_END_DATE = '2023-05-19'
TRADE_START_DATE = '2023-05-22'
TRADE_END_DATE = '2023-06-22'


API_KEY = ""
API_SECRET = ""
API_BASE_URL = 'https://paper-api.alpaca.markets'
data_url = 'wss://data.alpaca.markets'


ticker_list = [
    "AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "NVDA", "JPM", "BAC", "UNH", "MA", "WBA", "DIS",
    "V", "CMCSA", "HD", "KO", "CAT", "XOM", "WMT", "COST"
]


df_raw = download_data(ticker_list = ticker_list, start_date = TRAIN_START_DATE, end_date = TRADE_END_DATE, time_interval= '1D',
                       API_KEY= API_KEY, API_SECRET = API_SECRET, API_BASE_URL=API_BASE_URL)    

print('Data downladed:\n', df_raw);


fe = FeatureEngineer(use_technical_indicator=True,
                     tech_indicator_list = INDICATORS,
                     use_vix=True,
                     use_turbulence=True,
                     user_defined_feature = False)

processed = fe.preprocess_data(df_raw)

print('Data procesed:\n', processed);


list_ticker = processed["tic"].unique().tolist()
list_date = list(pd.date_range(processed['date'].min(),processed['date'].max()).astype(str))
combination = list(itertools.product(list_date,list_ticker))

processed_full = pd.DataFrame(combination,columns=["date","tic"]).merge(processed,on=["date","tic"],how="left")
processed_full = processed_full[processed_full['date'].isin(processed['date'])]
processed_full = processed_full.sort_values(['date','tic'])

processed_full = processed_full.fillna(0)

stock_dimension = len(processed_full.tic.unique())

# Calcular la longitud de los registros para cada fecha/hora
processed_full['record_length'] = processed_full.groupby('date')['tic'].transform('count')

# Eliminar los registros del DataFrame original
train = processed_full[processed_full['record_length'] == stock_dimension]

# Eliminar la columna 'record_length' si ya no es necesaria
processed_full = processed_full.drop('record_length', axis=1)

#Crear index agrupando por columna date (fecha\hora)
processed_full['index'] = processed_full.groupby('date').ngroup()
processed_full.set_index('index', inplace=True)

processed_full.head()

processed_full.to_csv('processed_full.csv')




train = data_split(processed_full, TRAIN_START_DATE,TRAIN_END_DATE)
trade = data_split(processed_full, TRADE_START_DATE,TRADE_END_DATE)
print(len(train))
print(len(trade))

train.to_csv('train_data.csv')
trade.to_csv('trade_data.csv')









