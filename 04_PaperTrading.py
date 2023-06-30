from finrl.config import INDICATORS
from AlpacaPaperTrading import AlpacaPaperTrading


ticker_list = [
    "AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "NVDA", "JPM", "BAC", "UNH", "MA", "WBA", "DIS",
    "V", "CMCSA", "HD", "KO", "CAT", "XOM", "WMT", "COST"
]
stock_dimension = len(ticker_list)

API_KEY = ""
API_SECRET = ""
API_BASE_URL = 'https://paper-api.alpaca.markets'
data_url = 'wss://data.alpaca.markets'

ERL_PARAMS = {"learning_rate": 3e-6,"batch_size": 2048,"gamma":  0.985,
        "seed":312,"net_dimension":[128,64], "target_step":5000, "eval_gap":30,
        "eval_times":1} 

#state_dim = 1 + 2 + 3 * stock_dimension + len(INDICATORS) * stock_dimension
state_dim = 1 + 2*stock_dimension + len(INDICATORS)*stock_dimension



paper_trading_erl = AlpacaPaperTrading(ticker_list = ticker_list, 
                                       time_interval = '5Min', 
                                       drl_lib = 'stable_baselines3', 
                                       agent = 'DDPG', 
                                       cwd = 'trained_models/', 
                                       net_dim = ERL_PARAMS['net_dimension'], 
                                       state_dim = state_dim, 
                                       action_dim= stock_dimension, 
                                       API_KEY = API_KEY, 
                                       API_SECRET = API_SECRET, 
                                       API_BASE_URL = API_BASE_URL, 
                                       tech_indicator_list = INDICATORS, 
                                       turbulence_thresh=30, 
                                       max_stock=1e2)
paper_trading_erl.run()