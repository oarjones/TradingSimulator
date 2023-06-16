﻿from finrl.config import INDICATORS
from finrl.config_tickers import DOW_30_TICKER
from AlpacaPaperTrading import AlpacaPaperTrading


ticker_list = DOW_30_TICKER
action_dim = len(DOW_30_TICKER)

API_KEY = "PK8XSVSUKK198Q6N0UO7"
API_SECRET = "1lDrm6cgZI3pDudsXShlPmgkQxSuly68MgKwRe6i"
API_BASE_URL = 'https://paper-api.alpaca.markets'
data_url = 'wss://data.alpaca.markets'

ERL_PARAMS = {"learning_rate": 3e-6,"batch_size": 2048,"gamma":  0.985,
        "seed":312,"net_dimension":[128,64], "target_step":5000, "eval_gap":30,
        "eval_times":1} 

state_dim = 1 + 2 + 3 * action_dim + len(INDICATORS) * action_dim


paper_trading_erl = AlpacaPaperTrading(ticker_list = DOW_30_TICKER, 
                                       time_interval = '5Min', 
                                       drl_lib = 'stable_baselines3', 
                                       agent = 'A2C', 
                                       cwd = 'trained_models', 
                                       net_dim = ERL_PARAMS['net_dimension'], 
                                       state_dim = state_dim, 
                                       action_dim= action_dim, 
                                       API_KEY = API_KEY, 
                                       API_SECRET = API_SECRET, 
                                       API_BASE_URL = API_BASE_URL, 
                                       tech_indicator_list = INDICATORS, 
                                       turbulence_thresh=30, 
                                       max_stock=1e2)
paper_trading_erl.run()