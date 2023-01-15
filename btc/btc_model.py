# -*- coding: utf-8 -*-
# Model attempts to maximise return from BTC

import numpy as np
import pandas as pd
from enum import Enum


class History(object):
    """
    Record what happens when a model runs
    """

    def __init__(self):
        self.actions = []
        self.btc_balance = []
        self.cash_balance = []
        self.portfolio_balance = []
        self.pct_change_buy = []
        self.pct_change_sell = []

    def add_event(self, new_action, new_btc_balance, new_cash_balance,
                  new_portfolio_balance, new_pct_change_buy, new_pct_change_sell):
        self.actions.append(new_action)
        self.btc_balance.append(new_btc_balance)
        self.cash_balance.append(new_cash_balance)
        self.portfolio_balance.append(new_portfolio_balance)
        self.pct_change_buy.append(new_pct_change_buy)
        self.pct_change_sell.append(new_pct_change_sell)

    def to_df(self):
        return pd.DataFrame({
            "Actions": self.actions,
            "BTC_Balance": self.btc_balance,
            "Cash_Balance": self.cash_balance,
            "Portfolio_Balance": self.portfolio_balance,
            "PCT_Change_Buy": self.pct_change_buy,
            "PCT_Change_Sell": self.pct_change_sell,
        })


class Action(Enum):
    NOTHING = 0
    BUY_BTC = 1
    SELL_BTC = 2


class BTCModel(object):

    def __init__(self, btc_historic_price, do_history,
                 cash_balance=100.0, btc_balance=0.0,
                 buy_action_pct=0.05, sell_action_pct=0.05,
                 buy_time_window=6, sell_time_window=6,
                 start_time=0,
                 ):
        """
        Constructor.

        Parameters
        ----------
        btc_historic_price : list
            The historic bitcoin price
        do_history : bool
            Whether or not to calculate the full history.
        cash_balance : float, optional
            Current account balance. The default is 100.
        btc_balance  : float, optional
            The current bitcoin balance. The default is 0.
        buy_action_pct, sell_action_pct : float, optional
            Percentage change in the BTC price to cause buy/sell action. Both
            assumed to be positive (e.g. to sell if it drops 7% set
            sell_action_pct=0.07). The defaults are 0.05 (i.e. 5%).
        buy_time_window, sell_time_window : float, optional
            The size of the window to look back over when calculating
            percentage change (in multiples of 10 minutes).
            The default is one hour (6 units)
        start_time : int, optional
            The time to start the model.
            The default is 0 (i.e. start on the first day for which price
            data are available).

        Returns
        -------
        None.
        """
        self.btc_historic_price = btc_historic_price

        self.cash_balance = cash_balance
        self.btc_balance = btc_balance
        self.buy_action_pct = buy_action_pct
        self.sell_action_pct = sell_action_pct
        self.buy_time_window = int(np.round(buy_time_window))
        self.sell_time_window = int(np.round(sell_time_window))

        self.current_time = start_time
        self.do_history = do_history
        if self.do_history:
            self.history = History()  # Record what happens in a model
        else:
            self.history = None

    def __call__(self, input_params_dict):
        """
        Run the model, compatible with pyABC. This is an alternative to calling the run() function.
        In pyABC, this function is called by the library
        and random variables (parameters) are passed in as a dictionary (input_params_dict).

        Example usage:
        m = BTCModel(btc_historic_price=historic_price_data, do_history=False)
        portfolio = m()

        Or for the model history:
        m2 = BTCModel(btc_historic_price=historic_price_data, do_history=True)
        history = m2()

        :return: The value of the portfolio every day (as a list value in a
            dictionary as required by the pyabc package) unless do_history was True when
            the BTCModel instance was created, in which case this is a list of History objects.
        """
        # Check that all input parametrers are not negative
        # (No longer doing this because testing if but/sell percentages can go negative
        #for k, v in input_params_dict.items():
        #    if v < 0:
        #        raise Exception(f"The parameter {k}={v} < 0. "
        #                        f"All parameters: {input_params_dict}")
        # print(f"Running with params: {input_params_dict}")

        # Run the model, passing any parameters (probably just the time window and action percentage).

        m = BTCModel(do_history=self.do_history, btc_historic_price=self.btc_historic_price,
                     **input_params_dict)
        portfolio = m.run()

        # Decide what to return
        if self.do_history:
            return m.history
        else:  # pyabc needs the results returned in a dictionary
            return {"data": portfolio}

    def step(self, _):
        """
        Step the model. Look at the price change over the last `self.time_window`
        iterations and decide whether to buy, sell, or do nothing.
        The function takes an argument (because it is called with map() later)
        but it ignores this.

        Returns
        -------
        The current portfolio balance

        """
        # Might need to run for a few days before we have enough price history
        if self.current_time < max(self.buy_time_window, self.sell_time_window):
            self.current_time += 1
            # Put soemthing in the history so that its index aligns with btc_historic_price
            if self.history is not None:
                self.history.add_event(Action.NOTHING, self.btc_balance,
                                       self.cash_balance, self.cash_balance, 0.0, 0.0)
            # Need to return a balance, as it's the start of the model no bitcoin have been bought
            # so the cash_balance is fine
            return self.cash_balance

        if self.current_time > len(self.btc_historic_price) - 1:
            raise Exception(f"Model iteration {self.current_time} is greater than " +
                            f"available history ({len(self.btc_historic_price)}")

        # Find the percentage change over the last time_window minutes (different change depending on buying
        # or selling because they might have different time windows
        current_btc_price = self.btc_historic_price[self.current_time]
        old_btc_price_buy = self.btc_historic_price[self.current_time - self.buy_time_window]
        old_btc_price_sell = self.btc_historic_price[self.current_time - self.sell_time_window]
        pct_change_buy = (current_btc_price - old_btc_price_buy) / old_btc_price_buy
        pct_change_sell = (current_btc_price - old_btc_price_sell) / old_btc_price_sell

        # Decide whether to buy, sell, or do nothing
        action = Action.NOTHING

        # On the last iteration, sell all the BTC
        if self.current_time == len(self.btc_historic_price)-1:
            action = Action.SELL_BTC
            self.cash_balance = self.btc_balance * current_btc_price
            self.btc_balance = 0
        # If there is growth in the price and we have cash then buy BTC
        elif pct_change_buy > self.buy_action_pct and self.cash_balance > 0:
            action = Action.BUY_BTC
            self.btc_balance = self.cash_balance / current_btc_price
            self.cash_balance = 0
        # Vice versa
        elif pct_change_sell < -1 * self.sell_action_pct and self.btc_balance > 0:
            action = Action.SELL_BTC
            self.cash_balance = self.btc_balance * current_btc_price
            self.btc_balance = 0

        # Update the clock by 1 minute
        self.current_time += 1

        # Current portfolio balance (cash and btc)
        portfolio_balance = self.cash_balance + (self.btc_balance * current_btc_price)

        # Record the history
        if self.history is not None:
            self.history.add_event(action, self.btc_balance, self.cash_balance,
                                   portfolio_balance, pct_change_buy, pct_change_sell)

        return portfolio_balance

    def run(self):
        """Run the model, returning the value of the portfolio at every iteration.
        It will run for every time step for which we have data for"""
        return list(map(self.step, range(len(self.btc_historic_price))))


    @staticmethod
    def performance(portfolio):
        """Quantify the performance of a given portfolio"""
        # Could take the mean value, highest value, etc
        # return np.mean(portfolio)  # Mean across all time
        return portfolio[-1]  # Final value


