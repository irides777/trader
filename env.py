import numpy as np
import pandas as pd
import gym
import random
from gym import spaces

class BaseMarket(gym.Env):
    '''
    训练环境
    '''
    def __init__(self, datas,  back_length, time_limit, direct=1):

        self.deal_lambda = 0.1 #使用指数分布模拟挂单成交情况
        self.np_random = np.random.RandomState()
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=((back_length+1)*5,)
        )

        self.datas = []
        self.lens = []
        for data in datas:
            self.datas.append(data[['bid','ask','bidv','askv','volume','tur', 'ori_vol']].copy()) #
            self.lens.append(data.shape[0])
        self.datas = pd.concat(self.datas).values
        self.lens = np.cumsum(self.lens)
        # self.data.loc[:,'tur'] = self.data['tur']/self.data['volume']

        self.back_length = back_length
        self.time_limit = time_limit

        self.is_eval = False
        self.direct = direct
        

    def _update(self):
        '''
        更新tick数据
        '''
        self.time = self.time+1
        self.tick_state = self.episode_data[self.time+self.back_length-1]
        self.last_bid = self.bid
        self.last_ask = self.ask
        self.bid = self.tick_state[0]
        self.ask = self.tick_state[1]
        if self.time+self.back_length >= len(self.episode_data):
            self.time = self.time_limit
        else:
            self.state = np.vstack([
                np.array([self.time_limit-self.time,0,0,0,0]),
                self.episode_data[self.time:self.time+self.back_length,:5]
            ]).reshape(-1)
    
    def _ifdeal(self):

        state = self.tick_state
        vol = state[6]
        tur = state[5]
        askv = tur-vol*self.bid
        bidv = vol-askv
        # print(vol, tur, askv, bidv)
        if not (askv<0 or bidv<0):
            return False
        
        v = bidv if self.direct==1 else askv
        if self.pos_in_queue <= v:
            return True
        self.pos_in_queue -= v
        return False


    def step(self, action):
        '''
        环境接收到智能体的操作信号后，首先更新下单价格与下单仓位，之后更新tick数据，判断是否成交，返回相应状态和奖励
        0:在bid0价格上等待 1:以ask价格成交
        '''
        done = False
        reward = 0

        if self.direct == 1:
            if action == 1:
                reward = -self.ask+1
                done = True
            
            self._update()


            if self.bid < 0:
                reward = 1
                done = True
            elif self.bid == 0 and self.last_bid == 0 and self._ifdeal():
                reward = 1
                done = True
            elif self.time >= self.time_limit:
                reward = -self.ask+1
                done = True

        elif self.direct == -1:
            if action == 1:
                reward = self.bid+1
                done = True
            
            self._update()

            if self.ask > 0:
                reward = 1
                done = True
            elif self.ask == 0 and self.last_ask == 0 and self._ifdeal():
                reward = 1
                done = True
            elif self.time >= self.time_limit:
                reward = self.bid+1
                done = True

        assert not np.isnan(self.state).any()

        return self.state, reward, done, {}

    def reset(self):

        self.time = 0
        ub = int(len(self.lens))*0.95
        ok = False
        self.length = self.np_random.exponential(1/self.deal_lambda)

        while not ok:

            if not self.is_eval: 
                day = self.np_random.randint(low=0,high=ub)
            else:
                day = self.np_random.randint(low=ub,high=len(self.lens))
            # for day in range(len(self.lens)):
            beg = self.lens[day-1] if day>0 else 0
            data = self.datas[beg:self.lens[day]]
            #     assert not np.isnan(data).any()

            # print(self.time_limit, self.back_length, data.shape[0], day, self.lens[day-1])
            if data.shape[0] - self.time_limit < self.back_length:
                ok = False
                continue
            begin_time = self.np_random.randint(
                low=self.back_length, 
                high=data.shape[0]-self.time_limit
            )
            end_time = begin_time+self.time_limit

            self.episode_data = data[begin_time-self.back_length+1:end_time+1].copy()
            if self.direct == 1:
                self.target_price = self.episode_data[self.back_length-1,0]
                self.pos_in_queue = self.episode_data[self.back_length-1,2]+1
            elif self.direct == -1:
                self.target_price = self.episode_data[self.back_length-1,1]
                self.pos_in_queue = self.episode_data[self.back_length-1,3]+1

            ok = self._data_check()
        
        self.episode_data[:,0] = self.episode_data[:,0]-self.target_price
        self.episode_data[:,1] = self.episode_data[:,1]-self.target_price
        self.episode_data[:,5] = self.episode_data[:,5]-self.target_price*self.episode_data[:,6]

        self.tick_state = self.episode_data[self.back_length-1]
        self.bid = self.tick_state[0]
        self.ask = self.tick_state[1]
        self.state = np.vstack([
            np.array([self.time_limit,0,0,0,0]),
            self.episode_data[:self.back_length,:5]
        ]).reshape(-1)
        return self.state

    def seed(self, seed=None):
        self.np_random = np.random.RandomState(seed)
        self.is_eval=True

    def _data_check(self):
        data = self.episode_data
        ok = np.all(data[:,0] < data[:,1]) \
            and np.all(data != np.inf) \
            and np.all(data != -np.inf)
        return ok

class TestMarket(BaseMarket):
    '''
    测试环境:通过reset给定一组交易数据
    '''

    def __init__(self, back_length, time_limit, direct):
        
        self.np_random = np.random.RandomState()
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=((back_length+1)*5,)
        )
        # self.data.loc[:,'tur'] = self.data['tur']/self.data['volume']

        self.back_length = back_length
        self.time_limit = time_limit

        self.is_eval = False
        self.direct = direct

    def step(self, action):
        '''
        环境接收到智能体的操作信号后，首先更新下单价格与下单仓位，之后更新tick数据，判断是否成交，返回相应状态和奖励
        0:在bid0价格上等待 1:以ask价格成交
        '''
        done = False
        reward = 0

        if self.direct == 1:
            if action == 1:
                reward = -self.ask+1
                done = True
            
            self._update()

            if action == 0 and self.bid < 0:
                reward = 1
                done = True
            elif action == 0 and self.time >= self.time_limit:
                reward = -self.ask+1
                done = True

        elif self.direct == -1:
            if action == 1:
                reward = self.bid+1
                done = True
            
            self._update()

            if action == 0 and self.ask > 0:
                reward = 1
                done = True
            elif action == 0 and self.time >= self.time_limit:
                reward = self.bid+1
                done = True

        assert not np.isnan(self.state).any()

        return self.state, reward, done, {}

    def reset(self, data):

        self.time = 0

        self.episode_data = data.copy()

        if self.direct == 1:
            self.target_price = self.episode_data[self.back_length-1,0]
        elif self.direct == -1:
            self.target_price = self.episode_data[self.back_length-1,1]

        self._data_check()
        
        self.episode_data[:,0] = self.episode_data[:,0]-self.target_price
        self.episode_data[:,1] = self.episode_data[:,1]-self.target_price

        self.tick_state = self.episode_data[self.back_length-1]
        self.bid = self.tick_state[0]
        self.ask = self.tick_state[1]
        self.state = np.vstack([
            np.array([self.time_limit,0,0,0,0]),
            self.episode_data[:self.back_length]
        ]).reshape(-1)
        return self.state

    def _data_check(self):
        data = self.episode_data
        ok = np.all(data != np.inf) \
            and np.all(data != -np.inf)
        if not ok:
            print(data[data[:,0] >= data[:,1]])
            raise ValueError
