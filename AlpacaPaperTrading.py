# Disclaimer: Nothing herein is financial advice, and NOT a recommendation to trade real money. Many platforms exist for simulated trading (paper trading) which can be used for building and developing the methods discussed. Please use common sense and always first consult a professional before trading or investing.
# Setup Alpaca Paper trading environment
from __future__ import annotations

import datetime
import threading
import time

import alpaca_trade_api as tradeapi
import gym
import numpy as np
import pandas as pd
import torch

from AlpacaProcessor import AlpacaProcessor
from finrl.meta.paper_trading.common import AgentPPO
from finrl.config import TRAINED_MODEL_DIR
import itertools
import ta
import random


class AlpacaPaperTrading:
    def __init__(
        self,
        ticker_list,
        time_interval,
        drl_lib,
        agent,
        cwd,
        net_dim,
        state_dim,
        action_dim,
        API_KEY,
        API_SECRET,
        API_BASE_URL,
        tech_indicator_list,
        turbulence_thresh=30,
        max_stock=1e2,
        latency=None,
    ):
        
        self.CASH_THRESHOLD = 15000 # Cash threshold for buying/selling
        self.BUY_THRESHOLD = 500;

        self.ticker_list = ticker_list
        self.tech_indicator_list = tech_indicator_list
        # load agent
        self.drl_lib = drl_lib
        if agent == "ppo":
            if drl_lib == "elegantrl":
                agent_class = AgentPPO
                agent = agent_class(net_dim, state_dim, action_dim)
                actor = agent.act
                # load agent
                try:
                    cwd = cwd + "/actor.pth"
                    print(f"| load actor from: {cwd}")
                    actor.load_state_dict(
                        torch.load(cwd, map_location=lambda storage, loc: storage)
                    )
                    self.act = actor
                    self.device = agent.device
                except BaseException:
                    raise ValueError("Fail to load agent!")

            elif drl_lib == "rllib":
                from ray.rllib.agents import ppo
                from ray.rllib.agents.ppo.ppo import PPOTrainer

                config = ppo.DEFAULT_CONFIG.copy()
                config["env"] = StockEnvEmpty
                config["log_level"] = "WARN"
                config["env_config"] = {
                    "state_dim": state_dim,
                    "action_dim": action_dim,
                }
                trainer = PPOTrainer(env=StockEnvEmpty, config=config)
                trainer.restore(cwd)
                try:
                    trainer.restore(cwd)
                    self.agent = trainer
                    print("Restoring from checkpoint path", cwd)
                except:
                    raise ValueError("Fail to load agent!")

            elif drl_lib == "stable_baselines3":
                from stable_baselines3 import PPO

                try:
                    # load agent
                    self.model = PPO.load(cwd)
                    print("Successfully load model", cwd)
                except:
                    raise ValueError("Fail to load agent!")

            else:
                raise ValueError(
                    "The DRL library input is NOT supported yet. Please check your input."
                )
        elif agent == 'DDPG':
            if drl_lib == "stable_baselines3":                
                from stable_baselines3 import DDPG

                try:
                    # load agent
                    self.model = DDPG.load(TRAINED_MODEL_DIR + "/agent_ddpg")
                    print("Successfully load model", cwd)
                except:
                    raise ValueError("Fail to load agent!")

        elif agent == "A2C":
            from stable_baselines3 import A2C

            try:
                # load agent
                self.model = A2C.load(TRAINED_MODEL_DIR + "/agent_a2c")
                print("Successfully load model", cwd)
            except:
                raise ValueError("Fail to load agent!")      
        elif agent == "*":
            from stable_baselines3 import A2C
            from stable_baselines3 import DDPG
            from stable_baselines3 import SAC

            try:
                # load agent
                model_a2c = A2C.load(TRAINED_MODEL_DIR + "/agent_a2c")
                model_ddpg = DDPG.load(TRAINED_MODEL_DIR + "/agent_ddpg")
                model_sac = SAC.load(TRAINED_MODEL_DIR + "/agent_sac")

                self.models = [model_a2c, model_ddpg, model_sac]

                print("Successfully load model", cwd)
            except:
                raise ValueError("Fail to load agent!")   
        


        else:
            raise ValueError("Agent input is NOT supported yet.")

        # connect to Alpaca trading API
        try:
            self.alpaca = tradeapi.REST(API_KEY, API_SECRET, API_BASE_URL, "v2")
        except:
            raise ValueError(
                "Fail to connect Alpaca. Please check account info and internet connection."
            )

        # read trading time interval
        if time_interval == "1s":
            self.time_interval = 1
        elif time_interval == "5s":
            self.time_interval = 5
        elif time_interval == "1Min":
            self.time_interval = 60
        elif time_interval == "5Min":
            self.time_interval = 60 * 5
        elif time_interval == "15Min":
            self.time_interval = 60 * 15
        else:
            raise ValueError("Time interval input is NOT supported yet.")

        # read trading settings
        self.tech_indicator_list = tech_indicator_list
        self.turbulence_thresh = turbulence_thresh
        self.max_stock = max_stock

        # initialize account
        self.stocks = np.asarray([0] * len(ticker_list))  # stocks holding
        self.stocks_cd = np.zeros_like(self.stocks)
        self.cash = None  # cash record
        self.stocks_df = pd.DataFrame(
            self.stocks, columns=["stocks"], index=ticker_list
        )
        self.asset_list = []
        self.price = np.asarray([0] * len(ticker_list))
        self.stockUniverse = ticker_list
        self.turbulence_bool = 0
        self.equities = []

    def test_latency(self, test_times=10):
        total_time = 0
        for i in range(0, test_times):
            time0 = time.time()
            self.get_state()
            time1 = time.time()
            temp_time = time1 - time0
            total_time += temp_time
        latency = total_time / test_times
        print("latency for data processing: ", latency)
        return latency

    def current_executing_orders(self):
         # Obtener todas las órdenes activas
        orders = self.alpaca.list_orders(status='filled')

        # Crear una lista para almacenar los símbolos únicos de las órdenes activas
        symbols = []

        # Recorrer las órdenes y obtener los símbolos
        for order in orders:
            symbols.append(order.symbol)

        # Imprimir los símbolos únicos de las órdenes activas
        unique_symbols = list(set(symbols))
        return unique_symbols

    

    def run(self):               
        
        

        orders = self.alpaca.list_orders(status="open")
        
        for order in orders:
            self.alpaca.cancel_order(order.id)


        self.trade()
        last_equity = float(self.alpaca.get_account().last_equity)
        cur_time = time.time()
        self.equities.append([cur_time, last_equity])
        time.sleep(self.time_interval)
            




        ## Wait for market to open.
        #print("Waiting for market to open...")
        #self.awaitMarketOpen()
        

        #print("Market opened.")
        #while True:
        #    # Figure out when the market will close so we can prepare to sell beforehand.
        #    clock = self.alpaca.get_clock()
        #    closingTime = clock.next_close.replace(
        #        tzinfo=datetime.timezone.utc
        #    ).timestamp()
        #    currTime = clock.timestamp.replace(tzinfo=datetime.timezone.utc).timestamp()
        #    self.timeToClose = closingTime - currTime

        #    if self.timeToClose < (60 * 5):
        #        # Close all positions when 5 minutes til market close.  Any less and it will be in danger of not closing positions in time.

        #        print("Market closing soon.  Closing positions.")

        #        threads = []
        #        positions = self.alpaca.list_positions()
        #        for position in positions:
        #            if position.side == "long":
        #                orderSide = "sell"
        #            else:
        #                orderSide = "buy"
        #            qty = abs(int(float(position.qty)))
        #            respSO = []
        #            tSubmitOrder = threading.Thread(
        #                target=self.submitOrder(qty, position.symbol, orderSide, respSO)
        #            )
        #            tSubmitOrder.start()
        #            threads.append(tSubmitOrder)  # record thread for joining later

        #        for x in threads:  #  wait for all threads to complete
        #            x.join()

        #        # Run script again after market close for next trading day.
        #        print("Sleeping until market close (15 minutes).")
        #        time.sleep(60 * 15)

        #    else:
        #        self.trade()
        #        last_equity = float(self.alpaca.get_account().last_equity)
        #        cur_time = time.time()
        #        self.equities.append([cur_time, last_equity])
        #        time.sleep(self.time_interval)

    def awaitMarketOpen(self):
        isOpen = self.alpaca.get_clock().is_open
        while not isOpen:
            clock = self.alpaca.get_clock()
            openingTime = clock.next_open.replace(
                tzinfo=datetime.timezone.utc
            ).timestamp()
            currTime = clock.timestamp.replace(tzinfo=datetime.timezone.utc).timestamp()
            timeToOpen = int((openingTime - currTime) / 60)
            print(str(timeToOpen) + " minutes til market open.")
            time.sleep(60)
            isOpen = self.alpaca.get_clock().is_open

    def get_decision(self, state):
        predictions = []
        weights = [0.7, 0.9, 0.8]  # Adjust weights based on your preference A2C --> 0.7, DDPG --> 0.9, SAC --> 0.8
    

        for model in self.models:
            prediction = model.predict(state)[0]
            predictions.append(prediction)
    
        weighted_avg = sum(weight * pred for weight, pred in zip(weights, predictions))
        return weighted_avg
    

    



    
    

    def trade(self):
        
        cash = float(self.alpaca.get_account().cash)        

        state = self.get_state()
        
        predictions = []
        weights = [0.7, 0.9, 0.8]  # Adjust weights based on your preference A2C --> 0.7, DDPG --> 0.9, SAC --> 0.8
        for model in self.models:
            prediction = model.predict(state)[0]
            predictions.append(prediction)

        ddpg_prediction = predictions[1]
        
        self.stocks_cd += 1        

        if self.turbulence_bool == 0:

            threads = []
            for index in range(len(ddpg_prediction)):
                tic = self.stockUniverse[index];
                action = ddpg_prediction[index]

                #Vender acciones
                if action == -1:                     
                        
                    sell_num_shares = self.stocks[index]
                    qty = abs(int(sell_num_shares))

                    if qty > 0:
                        respSO = []
                        tSubmitOrder = threading.Thread(
                            target=self.submitOrder(
                                qty, self.stockUniverse[index], 'sell', respSO
                            )
                        )
                        tSubmitOrder.start()
                        threads.append(tSubmitOrder)  # record thread for joining later
                    
                    self.cash = float(self.alpaca.get_account().cash)
                    self.stocks_cd[index] = 0

                #Comprar acciones
                elif action == 1:
                    
                    #if cash < self.CASH_THRESHOLD:
                    #    print("Insufficient cash. Waiting for more funds.")
                    #    return

                    if self.cash < 0:
                        tmp_cash = 0
                    else:
                        tmp_cash = self.cash                        

                    
                    #buy_num_shares = min(
                    #    tmp_cash // self.price[index], abs(int(action*self.max_stock))
                    #)

                    buy_num_shares = min(
                        tmp_cash // self.price[index],  # Número máximo de acciones según el cash
                        self.BUY_THRESHOLD // self.price[index]  # Número máximo de acciones según el límite de compra
                    )

                    qty = abs(int(buy_num_shares))

                    if qty > 0:
                        respSO = []
                        tSubmitOrder = threading.Thread(
                            target=self.submitOrder(
                                qty, self.stockUniverse[index], "buy", respSO
                            )
                        )
                        tSubmitOrder.start()
                        threads.append(tSubmitOrder)  # record thread for joining later

                    self.cash = float(self.alpaca.get_account().cash)
                    self.stocks_cd[index] = 0

            for x in threads:  #  wait for all threads to complete
                x.join()

        else:  # sell all when turbulence
            threads = []
            positions = self.alpaca.list_positions()
            for position in positions:
                #CHATGPT --> Please explain to me the following if:
                if position.side == "long":
                    orderSide = "sell"
                else:
                    orderSide = "buy"
                qty = abs(int(float(position.qty)))
                respSO = []
                tSubmitOrder = threading.Thread(
                    target=self.submitOrder(qty, position.symbol, orderSide, respSO)
                )
                tSubmitOrder.start()
                threads.append(tSubmitOrder)  # record thread for joining later

            for x in threads:  #  wait for all threads to complete
                x.join()

            self.stocks_cd[:] = 0

    def get_state(self):
        alpaca = AlpacaProcessor(api=self.alpaca)

        price, tech, turbulence, lastdata_df = alpaca.fetch_latest_data(
            ticker_list=self.stockUniverse,
            time_interval='15Min',
            tech_indicator_list=self.tech_indicator_list
        )

        turbulence_bool = 1 if turbulence >= self.turbulence_thresh else 0

        turbulence = (
            self.sigmoid_sign(turbulence, self.turbulence_thresh) * 2**-5
        ).astype(np.float32)

        tech = tech * 2**-7
        positions = self.alpaca.list_positions()
        stocks = [0] * len(self.stockUniverse)

        for position in positions:
            ind = self.stockUniverse.index(position.symbol)
            stocks[ind] = abs(int(float(position.qty)))

        stocks = np.asarray(stocks, dtype=float)
        cash = float(self.alpaca.get_account().cash)

        self.cash = cash
        self.stocks = stocks
        self.turbulence_bool = turbulence_bool
        self.price = price

        amount = np.array(self.cash * (2**-12), dtype=np.float32)
        scale = np.array(2**-6, dtype=np.float32)
        
        stock_dimension = len(self.ticker_list)
        num_stock_shares = [0] * stock_dimension
         
        
        state = (                    
                    [50000]
                    + lastdata_df.close.values.tolist()
                    + num_stock_shares
                    + sum(
                        (
                            lastdata_df[tech].values.tolist()
                            for tech in self.tech_indicator_list
                        ),
                        [],
                    )
                )  # append initial stocks_share to initial state, instead of all zero
        
        state = np.array(state)
        state[np.isnan(state)] = 0.0
        state[np.isinf(state)] = 0.0
        
        return state

    def submitOrder(self, qty, stock, side, resp):
        if qty > 0:
            try:
                self.alpaca.submit_order(stock, qty, side, "market", "day")
                print(
                    "Market order of | "
                    + str(qty)
                    + " "
                    + stock
                    + " "
                    + side
                    + " | completed."
                )
                resp.append(True)
            except:
                print(
                    "Order of | "
                    + str(qty)
                    + " "
                    + stock
                    + " "
                    + side
                    + " | did not go through."
                )
                resp.append(False)
        else:
            """
            print(
                "Quantity is 0, order of | "
                + str(qty)
                + " "
                + stock
                + " "
                + side
                + " | not completed."
            )
            """
            resp.append(True)

    @staticmethod
    def sigmoid_sign(ary, thresh):
        def sigmoid(x):
            return 1 / (1 + np.exp(-x * np.e)) - 0.5

        return sigmoid(ary / thresh) * thresh


class StockEnvEmpty(gym.Env):
    # Empty Env used for loading rllib agent
    def __init__(self, config):
        state_dim = config["state_dim"]
        action_dim = config["action_dim"]
        self.env_num = 1
        self.max_step = 10000
        self.env_name = "StockEnvEmpty"
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.if_discrete = False
        self.target_return = 9999
        self.observation_space = gym.spaces.Box(
            low=-3000, high=3000, shape=(state_dim,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=-1, high=1, shape=(action_dim,), dtype=np.float32
        )

    def reset(
        self,
        *,
        seed=None,
        options=None,
    ):
        return

    def step(self, actions):
        return