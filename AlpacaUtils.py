import pandas as pd
import alpaca_trade_api as tradeapi

class AlpacaUtils:
    def get_bars(self, ticker_list, start_date, end_date, time_interval, API_KEY=None, API_SECRET=None, API_BASE_URL=None
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
        data_df["date"] = data_df["date"].apply(lambda x: x.strftime("%Y-%m-%d %H:%M:%S"))

        data_df = data_df.dropna()
        data_df = data_df.reset_index(drop=True)

        data_df = data_df.sort_values(by=["date", "tic"]).reset_index(drop=True)       

        return data_df