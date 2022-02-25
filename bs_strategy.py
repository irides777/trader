import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from env import *
import sys

def ntimes_hd(env, hd_time):
    done = False
    action = 0
    tot = 0
    state = env.reset()
    while not done:
        if hd_time == 1:
            action = 1
        state, reward, done, _ = env.step(action)
        tot += reward
        hd_time -= 1
    return tot

def nticks_lv(env, lv_tick):
    done = False
    action = 0
    tot = 0
    state = env.reset()
    while not done:
        bid = state[-5]
        if bid >= lv_tick:
            action = 1
        state, reward, done, _ = env.step(action)
        tot += reward
    return tot


def preprocess(tickpath, filename):
    tickdf = pd.read_csv(os.path.join(tickpath, filename),names=['date','time','ms','lastprice','volume','bid','bidv','ask','askv','opi','tur','contract'])
    tickdf['datestr'] = tickdf['date'].apply(lambda x: '%s-%s-%s'%(str(x)[:4], str(x)[4:6], str(x)[6:8]))
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


if __name__ == '__main__':

    cwd = '/data/data_zltick/rb'
    datas = []
    for _,__, files in os.walk(cwd):
        for file in files:
            if int(file[:4])<2021:
                continue
            data = preprocess(cwd, file)
            datas.append(data)

    env = BaseMarket(datas, 100, 600)
    times = int(sys.argv[1])
    env.seed(123)
    tot = 0
    tmp = 0
    
    if sys.argv[2] == 'hd':
        hd_time = int(sys.argv[3])
        for i in range(times):
            reward = ntimes_hd(env, hd_time)
            tot += reward
            tmp += reward
            if i % 1000 == 0:
                print(f'1000 times avg: {tmp/1000}')
                tmp = 0
        print(tot/times)
    elif sys.argv[2] == 'lv':
        lv_ticks = int(sys.argv[3])
        for i in range(times):
            reward = nticks_lv(env, lv_ticks)
            tot += reward
            tmp += reward
            if i % 1000 == 0:
                print(f'1000 times avg: {tmp/1000}')
                tmp = 0
        print(tot/times)