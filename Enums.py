import numpy as np

class Assets:
    XBTUSD = 0
    ETHUSD = 1
    XRPZ18 = 2
    XRPUSD = 3
    XRPXBT = 4

AssetsInverse = {0: 'XBTUSD', 1: 'ETHUSD', 2:'XRPZ18', 3:'XRPUSD', 4:'XRPXBT'}


class Direction:
    short = 0
    long = 1

    @staticmethod
    def str(val):
        try:
            return {0: 'short',
                    1: 'long'}[val]
        except KeyError:
            return '{}'.format(val)


class ActionSpace:
    # hold leverage and exit trade
    n = 2

    def sample(s):
        np.random.randint(0, s.n)


class StateSpace:
    p_long = 0
    p_short = 1
    elapsed = 2
    profit_rel = 3
    roll_max_profit_rel = 4
    trailing_max_profit_rel = 5
    # regr, next tick probabilities, ema, ma


class StatisticsFields:
    action = 0
    reward = 1
    prob_long = 2
    prob_short = 3
    close_ma = 4
    profit = 5

class FillsSchema:
    ts = 0
    direction = 1
    ts_fill_au = 2
    price_fill_au = 3
