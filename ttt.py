import numpy as np
import pandas as pd
import gym
from gym import spaces

def preprocess(tickpath, filename):
    tickdf = pd.read_csv(os.path.join(tickpath, filename),names=['date','time','ms','lastprice','volume','bid','bidv','ask','askv','opi','tur','contract'])
    tickdf.loc[:,'datestr'] = tickdf['date'].apply(lambda x: '%s-%s-%s'%(str(x)[:4], str(x)[4:6], str(x)[6:8]))
    tickdf['timestr'] = tickdf['time'].apply(lambda x: '%s:%s:%s'%(format(x, '06d')[:-4], format(x, '06d')[-4:-2], format(x, '06d')[-2:]))
    tickdf['timestr'] = tickdf['timestr']+'.'+tickdf['ms'].apply(lambda x: format(x, '03d'))
    tickdf = tickdf.set_index(pd.to_datetime(tickdf['datestr']+' '+tickdf['timestr']))
    tickdf.index.name = 'datetime'
    tickdf.loc[tickdf['bid']<0.9*tickdf['lastprice'], 'bid'] = np.nan
    tickdf.loc[tickdf['ask']<0.9*tickdf['lastprice'], 'ask'] = np.nan
    tickdf[['bid','ask']] = tickdf[['bid','ask']].fillna(method='ffill').fillna(tickdf.iloc[0]['lastprice'])
    tickdf['midprice'] = (tickdf['ask']+tickdf['bid'])/2.0

    tdata = tickdf[['bid','ask','bidv','askv','volume']]
    std = tdata.iloc[0,0]
    tdata.loc[:,'bid'] = tdata['bid']-std
    tdata.loc[:,'ask'] = tdata['ask']-std

    def norm(x):
        return (x-x.mean())/x.std()

    tdata.loc[:,'askv'] = norm(tdata.askv)
    tdata.loc[:,'bidv'] = norm(tdata.bidv)
    tdata.loc[:,'volume'] = norm(tdata.volume)
    return tdata

class BaseMarket(gym.Env):
    '''
    基础环境
    '''

    def __init__(self, datas,  back_length, time_limit):
        
        self.np_random = np.random.RandomState()
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=((back_length+1)*5,)
        )

        self.datas = []
        self.lens = []
        for data in datas:
            self.datas.append(data[['bid','ask','bidv','askv','volume']].copy()) #
            self.lens.append(data.shape[0])
        self.datas = pd.concat(self.datas).values
        self.lens = np.cumsum(self.lens)
        # self.data.loc[:,'tur'] = self.data['tur']/self.data['volume']

        self.back_length = back_length
        self.time_limit = time_limit

        self.is_eval = False
        

    def _update(self):
        '''
        更新tick数据
        '''
        self.time = self.time+1
        self.tick_state = self.episode_data[self.time+self.back_length-1]
        self.bid = self.tick_state[0]
        self.ask = self.tick_state[1]
        self.state = np.vstack([
            np.array([self.time_limit-self.time,0,0,0,0]),
            self.episode_data[self.time:self.time+self.back_length]
        ]).reshape(-1)



    def step(self, action):
        '''
        环境接收到智能体的操作信号后，首先更新下单价格与下单仓位，之后更新tick数据，判断是否成交，返回相应状态和奖励
        0:在bid0价格上等待 1:以ask价格成交
        '''
        done = False
        reward = 0
        if action == 1:
            reward = -self.ask+1
            done = True
        # elif action == 3:
        #     self.order_price = np.inf
        #     self.order_position = 1
        
        self._update()

        if action == 0 and self.bid < 0:
            reward = 1
            done = True
        elif action == 0 and self.time >= self.time_limit:
            reward = -self.ask+1
            done = True
        # elif action == 2:
        #     reward = -self.ask+1
        #     done = True
        # elif self.time >= self.time_limit:
        #     reward = -self.ask+1
        #     done = True
        # else:
        #     reward = -0.03
        #     done = False

        return self.state, reward, done, {}

    def reset(self):

        self.time = 0
        
        ub = int(len(self.lens)*0.95)

        if not self.is_eval: 
            day = self.np_random.randint(low=0,high=ub)
        else:
            day = self.np_random.randint(low=ub,high=len(self.lens))
        beg = self.lens[day-1] if day>0 else 0
        data = self.datas[beg:self.lens[day]]

        begin_time = self.np_random.randint(
            low=self.back_length, 
            high=data.shape[0]-self.time_limit
        )
        end_time = begin_time+self.time_limit

        self.episode_data = data[begin_time-self.back_length+1:end_time+1].copy()
        self.target_price = self.episode_data[self.back_length-1,0]
        
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

    def seed(self, seed=None):
        self.np_random = np.random.RandomState(seed)
        self.is_eval=True

import os
import time
import warnings

warnings.filterwarnings('ignore')

time_start = time.time()
cwd = '/Data/database/data_zltick/rb'
datas = []
for _,__, files in os.walk(cwd):
    for file in files:
        if int(file[:4])<2020:
            continue
        data = preprocess(cwd, file)
        datas.append(data)
time_end = time.time()
print(time_end-time_start)

time_start = time.time()
env = BaseMarket(datas=datas,back_length=100,time_limit=600)
tot = 0
env.seed(567)
for i in range(1024):
    done = False
    state = env.reset()
    action = 0
    while not done:
        state, reward, done, _ = env.step(action)
        if state[0] == 1:
            action = 1
    # print(reward)
    tot += reward
print(tot/1024)
time_end = time.time()
print(time_end-time_start)