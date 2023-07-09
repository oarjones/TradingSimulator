import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from stable_baselines3 import A2C, DDPG, PPO, SAC, TD3

from finrl.agents.stablebaselines3.models import DRLAgent
from finrl.config import INDICATORS, TRAINED_MODEL_DIR
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader

import tkinter as tk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import alpaca_trade_api as tradeapi
from sklearn.metrics import mean_absolute_error

#train = pd.read_csv('train_data.csv')
trade = pd.read_csv('trade_data.csv')

# If you are not using the data generated from part 1 of this tutorial, make sure 
# it has the columns and index in the form that could be make into the environment. 
# Then you can comment and skip the following lines.
#train = train.set_index(train.columns[0])
#train.index.names = ['']
#trade = trade.set_index(trade.columns[0])
#trade.index.names = ['']
trade = trade.sort_values(["date", "tic"]).reset_index(drop=True)
trade['index'] = trade.groupby('date').ngroup()
trade.set_index('index', inplace=True)


# Definir si se utiliza cada modelo
if_using_a2c = True
if_using_ddpg = True
if_using_ppo = True
if_using_td3 = True
if_using_sac = True

# Carga de los modelos entrenados si están siendo utilizados
trained_a2c = A2C.load(TRAINED_MODEL_DIR + "/agent_a2c") if if_using_a2c else None
trained_ddpg = DDPG.load(TRAINED_MODEL_DIR + "/agent_ddpg") if if_using_ddpg else None
trained_ppo = PPO.load(TRAINED_MODEL_DIR + "/agent_ppo") if if_using_ppo else None
trained_td3 = TD3.load(TRAINED_MODEL_DIR + "/agent_td3") if if_using_td3 else None
trained_sac = SAC.load(TRAINED_MODEL_DIR + "/agent_sac") if if_using_sac else None

stock_dimension = len(trade.tic.unique())
state_space = 1 + 2 * stock_dimension + len(INDICATORS) * stock_dimension
print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")

buy_cost_list = sell_cost_list = [0.001] * stock_dimension
num_stock_shares = [0] * stock_dimension

env_kwargs = {
    "hmax": 100,
    "initial_amount": 50000,
    "num_stock_shares": num_stock_shares,
    "buy_cost_pct": buy_cost_list,
    "sell_cost_pct": sell_cost_list,
    "state_space": state_space,
    "stock_dim": stock_dimension,
    "tech_indicator_list": INDICATORS,
    "action_space": stock_dimension,
    "reward_scaling": 1e-4
}



e_trade_gym = StockTradingEnv(df = trade, turbulence_threshold = 70,risk_indicator_col='vix', print_verbosity=2, **env_kwargs)
# env_trade, obs_trade = e_trade_gym.get_sb_env()

# Predicciones de los modelos sobre el entorno de trading

print('Predicción A2C:\n')
df_account_value_a2c, df_actions_a2c = DRLAgent.DRL_prediction(
    model=trained_a2c, 
    environment = e_trade_gym) if if_using_a2c else (None, None)

print('Predicción DDPG:\n')
df_account_value_ddpg, df_actions_ddpg = DRLAgent.DRL_prediction(
    model=trained_ddpg, 
    environment = e_trade_gym) if if_using_ddpg else (None, None)

print('Predicción PPO:\n')
df_account_value_ppo, df_actions_ppo = DRLAgent.DRL_prediction(
    model=trained_ppo, 
    environment = e_trade_gym) if if_using_ppo else (None, None)

print('Predicción TD3:\n')
df_account_value_td3, df_actions_td3 = DRLAgent.DRL_prediction(
    model=trained_td3, 
    environment = e_trade_gym) if if_using_td3 else (None, None)

print('Predicción SAC:\n')
df_account_value_sac, df_actions_sac = DRLAgent.DRL_prediction(
    model=trained_sac, 
    environment = e_trade_gym) if if_using_sac else (None, None)


## Obtener los valores reales y predichos
#y_true = trade['close'].values
#y_pred_a2c = df_account_value_a2c['account_value'].values
#y_pred_ddpg = df_account_value_ddpg['account_value'].values
#y_pred_ppo = df_account_value_ppo['account_value'].values
#y_pred_td3 = df_account_value_td3['account_value'].values
#y_pred_sac = df_account_value_sac['account_value'].values

## Calcular el error absoluto medio (MAE)
#mae_a2c  = mean_absolute_error(y_true, y_pred_a2c)
#mae_ddp  = mean_absolute_error(y_true, y_pred_ddpg)
#mae_ppo  = mean_absolute_error(y_true, y_pred_ppo)
#mae_td3  = mean_absolute_error(y_true, y_pred_td3)
#mae_sac  = mean_absolute_error(y_true, y_pred_sac)

## Imprimir el resultado
#print("A2C --> Error absoluto medio (MAE):", mae_a2c)
#print("DDP --> Error absoluto medio (MAE):", mae_ddp)
#print("PPO --> Error absoluto medio (MAE):", mae_ppo)
#print("TD3 --> Error absoluto medio (MAE):", mae_td3)
#print("SAC --> Error absoluto medio (MAE):", mae_sac)


## Crear el eje de tiempo
#time = range(len(y_true))

## Crear la figura y los subplots
#fig, ax = plt.subplots()

## Graficar los valores reales
#ax.plot(time, y_true, label='REAL')

## Graficar las predicciones
#ax.plot(time, y_pred_a2c, label='A2C')
#ax.plot(time, y_pred_ddpg, label='DDPG')
#ax.plot(time, y_pred_ppo, label='PPO')
#ax.plot(time, y_pred_td3, label='TD3')
#ax.plot(time, y_pred_sac, label='SAC')

## Establecer título y etiquetas de los ejes
#ax.set_title('Comparación de Valores Reales y Predicciones')
#ax.set_xlabel('Tiempo')
#ax.set_ylabel('Valor')

## Mostrar la leyenda
#ax.legend()

## Crear la ventana de tkinter
#root = tk.Tk()
#root.title("Gráfica de Predicciones")
#root.geometry("800x600")

## Crear el widget FigureCanvasTkAgg
#canvas = FigureCanvasTkAgg(fig, master=root)
#canvas.draw()

## Ubicar el widget en la ventana
#canvas.get_tk_widget().pack()

## Ejecutar el bucle principal de tkinter
#tk.mainloop()






#def process_df_for_mvo(df):
#  return df.pivot(index="date", columns="tic", values="close")

#def StockReturnsComputing(StockPrice, Rows, Columns): 
#  import numpy as np 
#  StockReturn = np.zeros([Rows-1, Columns]) 
#  for j in range(Columns):        # j: Assets 
#    for i in range(Rows-1):     # i: Daily Prices 
#      StockReturn[i,j]=((StockPrice[i+1, j]-StockPrice[i,j])/StockPrice[i,j])* 100 
      
#  return StockReturn

#StockData = process_df_for_mvo(train)
#TradeData = process_df_for_mvo(trade)

#TradeData.to_numpy()

##compute asset returns
#arStockPrices = np.asarray(StockData)
#[Rows, Cols]=arStockPrices.shape
#arReturns = StockReturnsComputing(arStockPrices, Rows, Cols)

##compute mean returns and variance covariance matrix of returns
#meanReturns = np.mean(arReturns, axis = 0)
#covReturns = np.cov(arReturns, rowvar=False)
 
##set precision for printing results
#np.set_printoptions(precision=3, suppress = True)

##display mean returns and variance-covariance matrix of returns
#print('Mean returns of assets in k-portfolio 1\n', meanReturns)
#print('Variance-Covariance matrix of returns\n', covReturns)



#from pypfopt.efficient_frontier import EfficientFrontier

#ef_mean = EfficientFrontier(meanReturns, covReturns, weight_bounds=(0, 0.5))
#raw_weights_mean = ef_mean.max_sharpe()
#cleaned_weights_mean = ef_mean.clean_weights()
#mvo_weights = np.array([1000000 * cleaned_weights_mean[i] for i in range(stock_dimension)])
#mvo_weights

#LastPrice = np.array([1/p for p in StockData.tail(1).to_numpy()[0]])
#Initial_Portfolio = np.multiply(mvo_weights, LastPrice)
#Initial_Portfolio

#Portfolio_Assets = TradeData @ Initial_Portfolio
#MVO_result = pd.DataFrame(Portfolio_Assets, columns=["Mean Var"])
#MVO_result

# Definición de las fechas de entrenamiento y trade
#TRAIN_START_DATE = '2020-01-01'
#TRAIN_END_DATE = '2023-06-23'
#TRADE_START_DATE = '2023-06-26'
#TRADE_END_DATE = '2023-06-30'

#def download_data(
#    ticker_list, start_date, end_date, time_interval
#) -> pd.DataFrame:
    

#    API_KEY = "PKGUYB2D41YL8QB345ZX"
#    API_SECRET = "aFoRfrYAmGn7Bed6eyyaLRPPfXdAwcyITmteLRdq"
#    API_BASE_URL = 'https://paper-api.alpaca.markets'

    
#    api = tradeapi.REST(API_KEY, API_SECRET, API_BASE_URL, "v2")

#    start = start_date
#    end = end_date
#    time_interval = time_interval

#    data_df = pd.DataFrame()

#    # download
#    NY = "America/New_York"
#    start_date = pd.Timestamp(start_date + " 09:30:00", tz=NY)
#    end_date = pd.Timestamp(end_date + " 15:59:00", tz=NY)
#    barset = api.get_bars(
#        ticker_list,
#        time_interval,
#        start=start_date.isoformat(),
#        end=end_date.isoformat(),
#    ).df
    

#    # barset response has the timestamps in UTC - Convert to NY timezone
#    barset = barset.reset_index()
    
#    barset["timestamp"] = barset["timestamp"].apply(lambda x: x.tz_convert(NY))
#    barset = barset.set_index("timestamp")

#    # from trepan.api import debug;debug()
#    # filter opening time of the New York Stock Exchange (NYSE) (from 9:30 am to 4:00 pm) if time_interval < 1D
#    if time_interval != '1D':
#        NYSE_open_hour = "09:30"  # in NY
#        NYSE_close_hour = "15:59"  # in NY
#        data_df = barset.between_time(NYSE_open_hour, NYSE_close_hour)
#    else:
#        data_df = barset
        

#    data_df = data_df.reset_index()
#    data_df.columns = [
#            "date",
#            "open",
#            "high",
#            "low",
#            "close",
#            "volume",
#            'trade_count', 
#            'vwap',
#            "tic",
#        ]
    
#    data_df["day"] = data_df["date"].dt.dayofweek
#    data_df["date"] = data_df.date.apply(lambda x: x.strftime("%Y-%m-%d %H:%M:%S"))

    
#    data_df = data_df.dropna()
#    data_df = data_df.reset_index(drop=True)

#    data_df = data_df.sort_values(by=["date", "tic"]).reset_index(drop=True)        

    

#    return data_df


#df_dji = download_data(ticker_list = ['DJIA'], start_date = TRADE_START_DATE, end_date = TRADE_END_DATE, time_interval= '15Min')    


#df_dji = df_dji[["date", "close"]]
#fst_day = df_dji["close"][0]
#dji = pd.merge(
#    df_dji["date"],
#    df_dji["close"].div(fst_day).mul(1000000),
#    how="outer",
#    left_index=True,
#    right_index=True,
#).set_index("date")

## Obtención de los resultados de los modelos y el MVO
#df_result_a2c = (
#    df_account_value_a2c.set_index(df_account_value_a2c.columns[0])
#    if if_using_a2c
#    else None
#)

#df_result_a2c = (
#    df_account_value_a2c.set_index(df_account_value_a2c.columns[0])
#    if if_using_a2c
#    else None
#)
#df_result_ddpg = (
#    df_account_value_ddpg.set_index(df_account_value_ddpg.columns[0])
#    if if_using_ddpg
#    else None
#)
#df_result_ppo = (
#    df_account_value_ppo.set_index(df_account_value_ppo.columns[0])
#    if if_using_ppo
#    else None
#)
#df_result_td3 = (
#    df_account_value_td3.set_index(df_account_value_td3.columns[0])
#    if if_using_td3
#    else None
#)
#df_result_sac = (
#    df_account_value_sac.set_index(df_account_value_sac.columns[0])
#    if if_using_sac
#    else None
#)

# Creación del DataFrame final con los resultados
#result = pd.DataFrame(
#    {
#        "a2c": df_result_a2c["account_value"] if if_using_a2c else None,
#        "ddpg": df_result_ddpg["account_value"] if if_using_ddpg else None,
#        "ppo": df_result_ppo["account_value"] if if_using_ppo else None,
#        "td3": df_result_td3["account_value"] if if_using_td3 else None,
#        "sac": df_result_sac["account_value"] if if_using_sac else None,
#        "mvo": MVO_result["Mean Var"],
#        "dji": dji["close"],
#    }
#)

#print(result.head(50))


## Crear la ventana de tkinter
#window = tk.Tk()
#window.title("Financial Results")

## Crear la figura de matplotlib
#fig = plt.figure(figsize=(20, 20))

## Generar la gráfica en la figura
#result.plot(ax=fig.gca())

## Crear el lienzo de tkinter para la gráfica
#canvas = FigureCanvasTkAgg(fig, master=window)
#canvas.draw()
#canvas.get_tk_widget().pack()

## Crear la etiqueta de tkinter para la tabla de resultados
#table_label = tk.Label(window, text=result.to_string())
#table_label.pack()

## Iniciar el bucle de eventos de tkinter
#window.mainloop()



