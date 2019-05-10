import datetime
from talib import abstract
from utilFunc import *
import pickle
import os


class EnvTrade:
    def __init__(s, backtest=False, market_order=False, resolution=['1m', '5s'][0]):
        """
        - load mid prices and trade ohlc like in BT
        - load entry preds
        - get all possible entry timestamps, seed randomize them
        - as states, use preds as simple start! bulid for option to expand. dont even bother naming...
        - train on 2 entries.
        - dynamic plotting
        """
        s.model_direction = Direction.long  # can only be long in here.
        s.backtest = backtest
        s.resolution = 1 if not backtest else 1
        s.db_resolution = '1min' if s.resolution == 60 else '1s'
        s.sec_multiplier = 1 if s.resolution == 60 else 60 // s.resolution
        s.market_order = market_order
        s.asset = Assets.ETHUSD
        s.ts_start = datetime.datetime(2019, 4, 1)
        s.ts_end = datetime.datetime(2019, 4, 1, 23, 59, 59)
        s.params = dotdict()
        s.params.asset = 'ethusd'#s.asset
        s.params.data_start = s.ts_start
        s.params.data_end = s.ts_end
        s.params.exchange = "bitmex"
        s.steps = 0
        for fn in ['ohlc', 'ohlc_bid', 'ohlc_ask']:
            with open(os.path.join('./data/{}'.format(fn)), 'rb') as f:
                s.__setattr__(fn, pickle.load(f))
        with open(os.path.join('./data/predictions'), 'rb') as f:
            preds = pickle.load(f)
        s.ix_close = s.ohlc.columns.get_loc('close')
        # temp fix
        preds = preds[np.where(preds[:, 0]==s.ohlc.index[0])[0][0]:1+np.where(preds[:, 0]==s.ohlc.index[-1])[0][0]]
        s.states_ts = preds[:, 0]
        s.close_preds = preds[:, 1:]
        first_derivation = np.zeros_like(s.states_ts)
        first_derivation[1:] = np.subtract(preds[1:, 1], preds[:-1, 1])
        s.entry_ts = s.states_ts[np.where((preds[:, 1] >= 0.16) & (first_derivation > 0))]
        # cut some for testing here:
        s.entry_ts = s.entry_ts[::10]
        s.trade_entry_ix = np.where(s.states_ts==s.entry_ts[0])[0][0]
        # s.close_preds = s.load_from_db()
        # s.match_resolution()
        s.ma_period = 10 * s.sec_multiplier  # int in min
        s.close_ma = s.get_talib_on_close(data=s.ohlc.iloc[:, s.ix_close].astype(float), inputParams=dict(timeperiod=s.ma_period))
        # s.reset_start_ix()
        s.close = s.ohlc.iloc[:, s.ix_close]
        s.states = s.init_state_array(len(preds))
        s.sql_to_state()
        s.state_size = s.states.shape[1]
        s.init_statistics()
        s.t_last_direction_change = 0
        s.action_space = ActionSpace()
        s.action_size = s.action_space.n
        s.ix = 0
        s.ts_now: datetime.datetime = s.to_ts(0)
        s.done = False
        s.actions = []
        s.rewards = []
        s.order_fills = []
        s.order_fee = {Assets.ETHUSD: 0.0, Assets.XBTUSD: 0}[s.asset]
        # if s.backtest:
        #     s.vb = VectorizedBacktest(ts_start=pd.to_datetime(s.ts_start), ts_end=pd.to_datetime(s.ts_end))

    def get_talib_on_close(s, data: np.ndarray, f='MA', inputParams=dict(timeperiod=10)):
        return getattr(abstract, f)({'close': data}, **inputParams)

    def reset_start_ix(s):
        s.close_preds = s.close_preds[s.ma_period:]
        s.close_ma = s.close_ma[s.ma_period:]

    def match_resolution(s):
        if s.resolution < 60 and s.resolution != 1:
            s.close_preds = s.close_preds[::s.resolution]

    def init_state_array(s, n_ticks):
        state_cols = [attr for attr in dir(StateSpace) if '__' not in attr]
        s.states = np.empty((n_ticks, len(state_cols)))
        return s.states

    def init_statistics(s):
        s.statistics = np.zeros((len(s.states), len([attr for attr in dir(StatisticsFields) if '__' not in attr])))
        s.statistics[:, [StatisticsFields.prob_long, StatisticsFields.prob_short]] = 0.5
        s.statistics[:, StatisticsFields.close_ma] = s.close_ma
        s.statistics[:, StatisticsFields.profit] = 0
        s.statistics[0, StatisticsFields.action] = s.model_direction

    def sql_to_state(s):
        s.states[:, StateSpace.p_long] = s.close_preds[:, 0]
        s.states[:, StateSpace.p_short] = s.close_preds[:, 1]
        # for i in range(3, 3+len(s.target_regr_cols)):
        #     s.states[:, i] = s.close_preds[:, i+1]
        # state[:, StateSpace.p_net] = np.divide(np.subtract(state[:, StateSpace.p_long], state[:, StateSpace.p_short]) + 0.5, 2)

    # def load_from_db(s):
    #     regr_col_names = ','.join(['`{}`'.format(col) for col in s.target_regr_cols])
    #     sub_regr_col_names = ','.join(['sub1.`{}`'.format(col) for col in s.target_regr_cols])
    #     sql = '''select sub2.ts, `close`, sub1.`long`, sub1.short, {3} from trade.ohlcv_{0}_{5} sub2
    #         left outer join (select ts, asset, `long`, short, {4} from trade.predictions) sub1
    #         on sub2.ts = sub1.ts and sub2.asset = sub1.asset
    #         where sub2.asset = '{0}' and
    #         sub2.ts >= '{1}' and
    #         sub2.ts <= '{2}';'''.format(s.asset, s.ts_start, s.ts_end, sub_regr_col_names, regr_col_names, s.db_resolution)
    #     return np.array(s.db.fetchall(sql))
    #
    # def load_fills(s):
    #     sql = '''select ts, convert(direction, unsigned integer), ts_fill_auto_update, price_fill_auto_update from trade.fills
    #         where asset = '{0}'
    #         and ts >= '{1}'
    #         and ts <= '{2}';'''.format(s.asset, s.ts_start, s.ts_end)
    #     return np.array(s.db.fetchall(sql))
        # nda[:, FillsSchema.direction] = np.apply_along_axis(lambda x: int(x), 0, [nda[:, FillsSchema.direction]])
        # return nda

    def reset(s):
        s.done = False
        # s.vb.reset()
        print(
              f'Episode rewards: {sum(s.statistics[:, StatisticsFields.reward])} | '
              f'profit: {sum(s.statistics[:, StatisticsFields.profit])} | '
              f'Long Actions: {sum(s.statistics[:, StatisticsFields.action])} | '
              f'Hold Actions: {s.steps - sum(s.statistics[:, StatisticsFields.action])} | '
        )
        s.init_statistics()
        s.actions = []
        s.rewards = []
        s.ix = np.where(s.states_ts == s.entry_ts[0])[0][0]
        new_state = s.states[s.ix, :]
        s.steps = 0
        s.trade_entry_ix = np.where(s.states_ts == s.entry_ts[0])[0][0]
        return new_state

    def step(s, action):
        s.ts_now = s.to_ts()
        if action is None:
            action = np.random.randint(0, 2)
        s.statistics[s.ix, StatisticsFields.action] = action

        reward = s.step_calc_reward(action)
        # s.step_log_last_direction_change(action)
        # s.step_handle_side(action)
        s.step_whether_done()
        s.step_store_profit(action)
        s.move_to_next_entry(action)
        new_state = s.states[s.ix, :]
        s.steps += 1
        return new_state, float(reward), s.done, None


    def move_to_next_entry(s, action):
        if action == 0:
            # just move forward
            s.ix += 1
        elif action == 1:
            # set time stamp to next entry timestamp
            ts = s.to_ts(s.ix)
            next_ts = s.entry_ts[s.entry_ts > ts]
            if len(next_ts) == 0:
                ts = s.entry_ts[0]
                s.done = True
            else:
                ts = next_ts[0]
            s.ix = np.where(s.states_ts == ts)[0][0]
            s.trade_entry_ix = s.ix
        else:
            raise

    def to_ts(s, ix: int = False):
        if ix:
            return s.states_ts[ix]
        else:
            return s.states_ts[s.ix]

    def plot(s):
        plot_rl_sine_side(s.close, s.statistics, s.states, StatisticsFields, StateSpace,
                          **dict(look_ahead=0, sec_multiplier=s.sec_multiplier))

    def plot_vb(s):
        # s.vb.present_backtest()
        pass

    def step_whether_done(s):
        # either not exiting befor end of data or starting withentry from beginning
        if s.ix >= len(s.states) or s.to_ts(s.ix) > s.entry_ts[-1]:  # - (2 + s.look_ahead):
            s.done = True

    def step_calc_reward(s, action):
        # reward = s.step_calc_close_reward(action)
        # reward_smooth = s.step_calc_smoothed_reward(action)  # incentivize early reversal
        reward = s.step_calc_profit_reward(action)  # incentivize being right on spot with profit
        # reward = s.step_calc_profit_ma_reward(action)
        # reward_action_change = s.step_calc_reward_action_change(action)  # disincentivize frequent direction changes / whipsawin
        # reward = reward_action_change + float(reward_profit) / 2
        s.statistics[s.ix+1, StatisticsFields.reward] = reward
        return reward

    def step_calc_reward_action_change(s, action):
        if s.action_direction_changed(action) and \
                max(s.statistics[s.ix, StatisticsFields.prob_long], s.statistics[s.ix, StatisticsFields.prob_short]) >= 0.5:
            reward_action_change = -s.order_fee  # buy the spread, disincentivize frequent changes or sorta cover market order fee
        else:
            reward_action_change = 0
        return reward_action_change

    def step_calc_close_reward(s, action):
        if action == Direction.short:
            reward = s.close[s.ix] - s.close[s.ix+1]
        elif action == Direction.long:
            reward = s.close[s.ix+1] - s.close[s.ix]
        else:
            raise ('Action unknown. Cannot calc reward')
        return reward

    def step_calc_smoothed_reward(s, action):
        if action == Direction.short:
            reward = s.close_ma[s.ix + s.look_ahead] - s.close_ma[s.ix + s.look_ahead + 1]
        elif action == Direction.long:
            reward = s.close_ma[s.ix + s.look_ahead + 1] - s.close_ma[s.ix + s.look_ahead]
        else:
            raise ('Action unknown. Cannot calc reward')
        return reward

    def step_calc_profit_reward(s, action):
        if action == 0:
            reward = 0
        elif action == 1:  # exiting whole game
            if s.model_direction == Direction.short:
                reward = s.close[s.ix] - s.close[s.ix+1]
            elif s.model_direction == Direction.long:
                reward = s.ohlc_bid.iloc[s.ix, s.ix_close] - s.close[s.trade_entry_ix]  # (1 - 0.0006225) *
        else:
            raise ('Action unknown. Cannot calc reward')
        return reward

    def step_calc_profit_ma_reward(s, action):
        if s.states[s.ix, StateSpace.side] == Direction.short:
            reward = s.close_ma[s.ix] - s.close_ma[s.ix+1]
        elif s.states[s.ix, StateSpace.side] == Direction.long:
            reward = s.close_ma[s.ix+1] - s.close_ma[s.ix]
        else:
            raise ('Action unknown. Cannot calc reward')
        return reward

    def step_store_profit(s, action):
        if s.model_direction == Direction.short:
            s.statistics[s.ix + 1, StatisticsFields.profit] = s.close[s.trade_entry_ix] - s.ohlc_ask.iloc[s.ix, s.ix_close]
        elif s.model_direction == Direction.long:
            if action == 1:
                s.statistics[s.ix + 1, StatisticsFields.profit] = (1 - 0.0006225) * s.ohlc_bid.iloc[s.ix, s.ix_close] - s.close[s.trade_entry_ix]
            else:
                s.statistics[s.ix + 1, StatisticsFields.profit] = 0
            s.states[s.ix + 1, StateSpace.profit_rel] = ((1 - 0.0006225) * s.ohlc_bid.iloc[s.ix, s.ix_close] - s.close[s.trade_entry_ix]) / s.close[s.trade_entry_ix]
            try:
                s.states[s.ix + 1, StateSpace.roll_max_profit_rel] = np.max(s.states[s.trade_entry_ix:s.ix + 1, StateSpace.profit_rel])
            except ValueError:
                s.states[s.ix + 1, StateSpace.roll_max_profit_rel] = 0
            s.states[s.ix + 1, StateSpace.trailing_max_profit_rel] = s.states[s.ix + 1, StateSpace.roll_max_profit_rel] - s.states[s.ix + 1, StateSpace.profit_rel]
            s.states[s.ix + 1, StateSpace.elapsed] = (s.ix + 1 - s.trade_entry_ix) / 20000

    def store_profit_ma(s):
        s.statistics[0:s.ix+1, StatisticsFields.profit_ma] = \
            s.get_talib_on_close(np.array(s.statistics[0:s.ix+1, StatisticsFields.profit]))

    @staticmethod
    def scale_direction(direction):
        if direction == 0:
            return -1
        else:
            return direction

    def store_probs(s, probs):
        s.statistics[s.ix, StatisticsFields.prob_long] = probs[1]
        s.statistics[s.ix, StatisticsFields.prob_short] = probs[0]


if __name__ == '__main__':
    env = EnvTrade()
