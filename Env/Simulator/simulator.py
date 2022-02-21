import numpy as np
from numpy.random import normal
import pandas as pd

from collections import deque
from typing import List, Dict, Any, Optional, Union


def geometric_movement(t, drift=0.1, volatility=0.01, g0=10, drift_as_trend=False) -> np.ndarray:
    trend = drift - 0.5 * volatility ** 2
    return g0 * np.exp((drift if drift_as_trend else trend) * t)


def standard_movement(t, dt=None) -> np.ndarray:
    n = len(t) if isinstance(t, (np.ndarray, list, tuple)) else 1
    std = np.random.standard_normal(size=n)
    return np.cumsum(std) * np.sqrt(dt if dt is not None else t[1] - t[0])


def gbm(t, drift=0.1, volatility=0.01, g0=10, dt=None) -> np.ndarray:
    return geometric_movement(t, drift, volatility, g0) * np.exp(volatility * standard_movement(t, dt))


class Stock:
    def __init__(self,
                 initial_f_price: float,
                 f_drift: float,
                 f_volatility: float,
                 code=None,
                 buffer_size: int = 5000,
                 drift_as_trend: bool = False):

        self.initial_f_price = initial_f_price
        self.f_drift = f_drift
        self.f_volatility = f_volatility
        self.code = hex(id(self) // 54321)[2:] if code is None else code

        self.f_trend = self.f_drift - 0.5 * self.f_volatility ** 2 if not drift_as_trend else self.f_drift

        # for price
        self.f_price = self.initial_f_price
        # self.current_price = self.f_price
        self.price_record = deque([self.initial_f_price], maxlen=buffer_size)

        # for f_price generating
        self.s_move = deque([0], maxlen=buffer_size)
        self.g_move = self.initial_f_price
        self.dt = 1

    @property
    def price(self):
        return self.price_record[-1]

    @price.setter
    def price(self, price):
        self.price_record.append(price)

    def progress(self, price=None):
        self.g_move *= np.exp(self.f_trend)
        self.s_move.append(self.s_move[-1] + normal(0, 1) * self.f_volatility * np.sqrt(self.dt))
        self.f_price = self.g_move * np.exp(self.s_move[-1])
        self.price = price if price is not None else self.f_price

    def __repr__(self):
        return hex(id(self))[2:]

    def get_f_price_record(self):
        t = np.arange(len(self.price_record))
        return self.initial_f_price*np.exp(self.f_trend*t + self.s_move)

    def info(self):
        print(f"===== stock_{self.code} =====")
        print(f'g0: {self.initial_f_price}, mu: {self.f_drift}, sigma: {self.f_volatility}, trend: {self.f_trend}')
        print(f'price: {self.price}, f_price: {self.f_price}')


class FCN:
    def __init__(self,
                 window_size: Optional[Union[int, List[int]]],
                 order_margin: Optional[Union[float, List[float]]],
                 code: Union[str, int] = None):
        self.window_size = window_size
        self.order_margin = order_margin
        self.code = hex(id(self) // 54321)[2:] if code is None else code

        self.f_weight = normal(0.3, 0.03)
        self.c_weight = normal(0.3, 0.03)
        self.n_weight = normal(0.3, 0.03)

        # self.observing_stock = observing_stock

        self.last_action_time = 0

    def get_reward(self, stock, t_const=10000) -> float:
        f = np.log(stock.f_price / stock.price) / t_const
        c = np.log(stock.price / stock.price_record[-self.window_size-1]) / t_const
        n = normal(0, 0.0001)

        return (self.f_weight * f + self.c_weight * c + self.n_weight * n) / \
               (self.f_weight + self.c_weight + self.n_weight)

    def predict(self, stock, t_const=10000) -> float:
        return stock.price * np.exp(self.get_reward(stock, t_const) * self.window_size)

    def order(self, stock, market_time, t_const=10000) -> Optional[Dict]:
        assert len(stock.price_record) == market_time+1, \
            f'stock price update has error! len(price_record): {len(stock.price_record)}  ' \
            f'market_time: {market_time}'

        if market_time - self.last_action_time <= self.window_size:     # agent's freezing time
            return None

        self.last_action_time = market_time
        prediction = self.predict(stock, t_const)
        k = self.order_margin

        vol = np.random.randint(1, 6)
        order = {'instrument': stock.code,
                 'price': 0,
                 'quantity': 0,
                 'side': 0,
                 'time': market_time,
                 'validity': self.window_size,
                 'client': self.code,
                 'client_id': hex(id(self) % 54321)[2:]}

        # buy:1
        if prediction > stock.price:
            order['price'] = (1 - k) * prediction
            order['quantity'] = vol
            order['side'] = 1
            return order
        # sell:2
        elif prediction < stock.price:
            order['price'] = (1 + k) * prediction
            order['quantity'] = vol
            order['side'] = 2
            return order

    def __repr__(self):
        return f'FCN[C:{self.code}, id:{hex(id(self) % 54321)[2:]}]'


class Market:
    def __init__(self,
                 # n_groups: int=1,
                 mu: Optional[Union[float, List[float]]] = None,
                 sigma: Optional[Union[float, List[float]]] = None,
                 initial_f: Optional[Union[float, List[float]]] = None,
                 window_sizes: Optional[Union[int, List[int]]] = None,
                 order_margins: Optional[Union[float, List[float]]] = None,
                 stocks_kwargs: Optional[Dict[str, Any]] = None,
                 fcn_kwargs: Optional[Dict[str, Any]] = None):
        """
        Market which has stocks
        # :param n_groups: number of stock groups that shares similar trend
        :param mu: drift of each stock groups
        :param sigma: volatility of each stock groups
        """
        self.mu = mu
        self.sigma = sigma
        self.initial_f = initial_f
        self.window_sizes = window_sizes
        self.order_margins = order_margins

        self.stock_kwargs = {} if stocks_kwargs is None else stocks_kwargs
        self.fcn_kwargs = {} if fcn_kwargs is None else fcn_kwargs

        self.stocks: List[Stock] = []
        self.fcn_agents: List[FCN] = []
        self.market_time = 0

        self.orderbooks: List[pd.DataFrame] = []

        self.set_stocks()
        self.set_fcn()

    def set_stocks(self):
        """
        add stocks and corresponding orderbooks
        """
        for drift, volatility, initial_f_price in zip(self.mu, self.sigma, self.initial_f):
            self.stocks.append(
                Stock(initial_f_price=initial_f_price,
                      f_drift=drift,
                      f_volatility=volatility,
                      code=len(self.stocks) + 1,
                      **self.stock_kwargs))

            self.orderbooks.append(
                pd.DataFrame(columns=[
                    'instrument', 'price', 'quantity', 'side', 'time', 'validity', 'client', 'client_id']))

    def set_fcn(self):
        """
        add fcn agents
        """
        for window_size, order_margin in zip(self.window_sizes, self.order_margins):
            self.fcn_agents.append(FCN(window_size, order_margin, code=len(self.fcn_agents) + 1))

    def fcn_ordering(self):
        """
        sample agents and submit orders according to the current stock prices
        """
        # agent sampling
        n_fcn = len(self.fcn_agents)
        for stock in self.stocks:
            n_selected_fcn = np.random.randint(1, max(n_fcn // 2, 1) + 1)   # uniform(1 ~ n_fcn//2)
            selected_fcn = np.random.choice(self.fcn_agents, size=n_selected_fcn)

            for fcn in selected_fcn:
                order = fcn.order(stock, market_time=self.market_time)
                if order is not None:
                    instrument = order['instrument']
                    self.orderbooks[instrument - 1] = \
                        self.orderbooks[instrument - 1].append(order, ignore_index=True)

    def order_processing_and_update(self, init=False, render=False):
        for orderbook, stock in zip(self.orderbooks, self.stocks):
            transaction_price = None
            if len(orderbook) and not init:
                # find prevailing order
                buy_orders = orderbook[orderbook['side'] == 1]
                sell_orders = orderbook[orderbook['side'] == 2]

                # if there are balanced
                if len(buy_orders) and len(sell_orders):
                    buy_orders_idx = buy_orders['price'].idxmax()
                    sell_orders_idx = sell_orders['price'].idxmin()

                    buy_order = orderbook.loc[buy_orders_idx]
                    sell_order = orderbook.loc[sell_orders_idx]
                    prevailing_order = sell_order if sell_order['time'] < buy_order['time'] else buy_order

                    if render:
                        print(f'market_time: {self.market_time}')
                        print(pd.DataFrame([buy_order, sell_order], columns=orderbook.columns))

                    # make transaction and update price
                    if buy_order['price'] >= sell_order['price']:
                        transaction_price = prevailing_order['price']
                        # stock.price = prevailing_order['price']     # update price
                        orderbook.drop(buy_orders_idx, inplace=True)  # update market
                        orderbook.drop(sell_orders_idx, inplace=True)

                # remove old orders
                orderbook.drop(orderbook[orderbook['time'] + orderbook['validity'] <= self.market_time].index,
                               inplace=True)

            stock.progress(transaction_price)

    def progress(self, init=False, render=False):
        self.fcn_ordering()
        self.order_processing_and_update(init, render=render)
        self.market_time += 1

    def simulate(self, max_time, render=False):
        while self.market_time < max_time:
            if self.market_time < 1000:
                self.progress(init=True)
            else:
                self.progress(init=False, render=render)


# n_stocks = 1
# n_fcns = 2
# mu = normal(1e-4, 1e-4, n_stocks)
# sigma = normal(1e-2, 1e-3, n_stocks)
# g0 = normal(500, 50, n_stocks)
#
# window_size = np.random.randint(50, 151, size=n_fcns)
# order_marin = np.random.rand(n_fcns) * 0.05

mu, sigma, g0, window_size, order_marin = (np.array([0.00012871]),
                                           np.array([0.00950263]),
                                           np.array([521.4674998]),
                                           np.array([136,  95]),
                                           np.array([0.0475047, 0.00171147]))
maxlen = 5000
stock_kwargs = {'buffer_size': maxlen+1, 'drift_as_trend': True}
market = Market(mu, sigma, g0, window_size, order_marin, stock_kwargs)

market.simulate(maxlen, render=False)

stock = market.stocks[0]
f_prices = stock.get_f_price_record()
prices = stock.price_record