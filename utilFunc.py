from Enums import *
import plotly.graph_objs as go


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __getstate__(s):
        # Deliberately do not return self.value or self.last_change.
        # We want to have a "blank slate" when we unpickle.
        __getattr__ = dict.get
        __setattr__ = dict.__setitem__
        __delattr__ = dict.__delitem__
        return s

    def __setstate__(s, state):
        # Make self.history = state and last_change and value undefined
        __getattr__ = object.__getattribute__
        __setattr__ = object.__setattr__
        __delattr__ = object.__delattr__


def gen_long(side, data, dist):
    y = []
    x = []
    for i in range(len(data)):
        if side[i] == Direction.long:
            y.append(data[i] + dist)
            x.append(i)
    return x, y

def gen_short(side, data, dist):
    y = []
    x = []
    for i in range(len(data)):
        if side[i] == Direction.short:
            y.append(data[i] - dist)
            x.append(i)
    return x, y


def plot_rl_sine_side(close, stats, states, StatisticsFields, StateSpace, **kwargs):
    # sec_multiplier = kwargs['sec_multiplier']
    close = close[::60]
    stats = stats[::60]
    states = states[::60]
    look_ahead = kwargs['look_ahead']
    direction = [Direction.long] * len(states)
    map_d = lambda x: -1 if x == 0 else 1
    direction_bin = [map_d(d) for d in direction]
    side = stats[:, StatisticsFields.action]
    rewards = stats[:, StatisticsFields.reward]
    prob_long = stats[:, StatisticsFields.prob_long]
    prob_short = stats[:, StatisticsFields.prob_short]
    x_fill_long, y_fill_long = gen_long(direction, close, close[0] / 50)
    x_fill_short, y_fill_short = gen_short(direction, close, close[0] / 50)
    x_long, y_long = gen_long(side, close, close[0] / 100)

    x_short, y_short = gen_short(side, close, close[0] / 100)
    traces = {(1, 1): [], (2, 1): [], (1,2): [], (2,2): []}
    traces[(1, 1)] = []
    traces[(1, 1)].append(go.Scatter(name='close',
                                     x=list(range(len(close))),
                                     y=close,
                                     mode='markers',
                                     marker=dict(
                                         size=4,
                                         color='blue',
                                         symbol=5)
                                     ))
    # traces[(1, 1)].append(go.Scatter(name='close_ma',
    #                                  x=list(range(len(close))),
    #                                  y=stats[:, StatisticsFields.close_ma],
    #                                  mode='markers', marker=dict(size=4, color='orange', symbol=5)))
    # traces[(1, 1)].append(go.Scatter(name='close_ma_lookahead',
    #                                  x=list(range(len(close)))[:-look_ahead],
    #                                  y=stats[:, StatisticsFields.close_ma][look_ahead:],
    #                                  mode='markers', marker=dict(size=4, color='pink', symbol=5)))
    traces[(1, 1)].append(go.Scatter(name='longs',
                                     x=x_long,
                                     y=y_long,
                                     mode='markers',
                                     marker=dict(size=3,color='green',symbol=4)))
    traces[(1, 1)].append(go.Scatter(name='short',
                                     x=x_short,
                                     y=y_short,
                                     mode='markers',
                                     marker=dict(size=3,color='red',symbol=4)))
    traces[(1, 1)].append(go.Scatter(name='fill_short',
                                     x=x_fill_short,
                                     y=y_fill_short,
                                     mode='markers', marker=dict(size=3, color='red', symbol=4)))
    traces[(1, 1)].append(go.Scatter(name='fill_long',
                                     x=x_fill_long,
                                     y=y_fill_long,
                                     mode='markers', marker=dict(size=3, color='green', symbol=4)
                                     ))
    traces[(1, 2)].append(go.Scatter(name='cum_rewards',
                                     x=list(range(0, len(side))),
                                     y=np.cumsum(rewards),
                                     mode='markers',
                                     marker=dict(size=3,color='green',symbol=4)))
    traces[(2, 1)].append(go.Scatter(name='prob_long',
                                     x=list(range(0, len(side))),
                                     y=prob_long,
                                     mode='markers', marker=dict(size=3, color='green', symbol=5)))
    traces[(2, 1)].append(go.Scatter(name='prob_short',
                                     x=list(range(0, len(side))),
                                     y=prob_short,
                                     mode='markers', marker=dict(size=3, color='red', symbol=6)))
    traces[(2, 2)].append(go.Scatter(name='cum_profit',
                                     x=list(range(1, len(direction))),
                                     y=np.cumsum(np.multiply(direction_bin[1:], np.subtract(close[1:], close[:-1]))),
                                     mode='markers',
                                     marker=dict(size=3,color='blue',symbol=3)))