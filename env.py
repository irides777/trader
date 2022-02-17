from importlib_metadata import List
import numpy as np
import gym
from gym import spaces

class BaseMarket(gym.Env):
    '''
    基础环境
    '''

    def __init__(self, datas:List,  back_length, time_limit):

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=((back_length+1)*5,)
        )

        self.datas = []
        for data in datas:
            self.datas.append(data[['bid','ask','bidv','askv','volume']].copy()) #
        # self.data.loc[:,'tur'] = self.data['tur']/self.data['volume']

        self.back_length = back_length
        self.time_limit = time_limit
        

    def _update(self):
        '''
        更新tick数据
        '''
        self.time = self.time+1
        self.tick_state = self.episode_data.iloc[self.time+self.back_length-1].values
        self.bid = self.tick_state[0]
        self.ask = self.tick_state[1]
        self.state = np.vstack([
            np.array([self.order_price, self.order_position,self.time_limit-self.time,0,0]),
            self.episode_data.iloc[self.time:self.time+self.back_length].values
        ]).reshape(-1)



    def step(self, action):
        '''
        环境接收到智能体的操作信号后，首先更新下单价格与下单仓位，之后更新tick数据，判断是否成交，返回相应状态和奖励
        0:在bid0价格上等待 1:以ask价格成交
        '''
        if action == 1:
            reward = -self.ask+1
            done = True
        # elif action == 3:
        #     self.order_price = np.inf
        #     self.order_position = 1
        
        self._update()

        if self.bid < self.order_price and self.order_position > 0:
            reward = -self.order_price+1
            done = True
        elif action == 2:
            reward = -self.ask+1
            done = True
        elif self.time >= self.time_limit:
            reward = -self.ask+1
            done = True
        else:
            reward = -0.03
            done = False

        return self.state, reward, done, {}

    def reset(self):

        self.order_price = 0
        self.order_position = 0
        self.time = 0
        
        data = self.datas[self.np_random.randint(low=0,high=len(self.datas))]
        begin_time = self.np_random.randint(
            low=self.back_length, 
            high=data.shape[0]-self.time_limit
        )
        end_time = begin_time+self.time_limit

        self.episode_data = data.iloc[begin_time-self.back_length+1:end_time+1].copy()
        self.target_price = self.episode_data.iloc[self.back_length-1,0]
        
        self.episode_data.loc[:,'bid'] = self.episode_data['bid']-self.target_price
        self.episode_data.loc[:,'ask'] = self.episode_data['ask']-self.target_price

        self.tick_state = self.episode_data.iloc[self.back_length-1].values
        self.bid = self.tick_state[0]
        self.ask = self.tick_state[1]
        self.state = np.vstack([
            np.array([0,0,self.time_limit,0,0]),
            self.episode_data.iloc[:self.back_length].values
        ]).reshape(-1)
        return self.state

    def seed(self, seed=None):
        self.np_random = np.random.RandomState(seed)

    