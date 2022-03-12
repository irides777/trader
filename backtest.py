import numpy as np
import pandas as pd
import os
from datetime import timedelta, date
from env import *

def preprocess(tickpath, filename):
    tickdf = pd.read_csv(os.path.join(tickpath, filename),names=['date','time','ms','lastprice','volume','bid','bidv','ask','askv','opi','tur','contract'])
    tickdf.loc[:,['lastprice','volume','bid','bidv','ask','askv','opi','tur']] = tickdf[['lastprice','volume','bid','bidv','ask','askv','opi','tur']].astype('float32')
    tickdf.loc[:,'datestr'] = tickdf['date'].apply(lambda x: '%s-%s-%s'%(str(x)[:4], str(x)[4:6], str(x)[6:8]))
    tickdf.loc[:,'timestr'] = tickdf['time'].apply(lambda x: '%s:%s:%s'%(format(x, '06d')[:-4], format(x, '06d')[-4:-2], format(x, '06d')[-2:]))
    tickdf.loc[:,'timestr'] = tickdf['timestr']+'.'+tickdf['ms'].apply(lambda x: format(x, '03d'))
    tickdf = tickdf.set_index(pd.to_datetime(tickdf['datestr']+' '+tickdf['timestr']))
    tickdf.index.name = 'datetime'
    # try:
    tickdf.loc[tickdf['bid']<0.9*tickdf['lastprice'], 'bid'] = np.nan
    tickdf.loc[tickdf['ask']<0.9*tickdf['lastprice'], 'ask'] = np.nan
    # except TypeError as e:
        
    #     print((tickdf['bid'][tickdf.bid.apply(lambda x:type(x) is str)]))
    #     print(float(tickdf['bid'].iloc[0]))
    #     print((tickdf['ask'][tickdf.bid.apply(lambda x:type(x) is str)]))
    #     print(float(tickdf['ask'].iloc[0]))
    #     raise e
    tickdf.loc[:,['bid','ask']] = tickdf[['bid','ask']].fillna(method='ffill').fillna(tickdf.iloc[0]['lastprice'])
    tickdf.loc[:,'midprice'] = (tickdf['ask']+tickdf['bid'])/2.0

    tdata = tickdf[['bid','ask','bidv','askv','volume']].copy()
    std = tdata.iloc[0,0]
    tdata.loc[:,'bid'] = tdata['bid']-std
    tdata.loc[:,'ask'] = tdata['ask']-std

    def norm(x):
        return (x-x.mean())/x.std()

    tdata.loc[:,'askv'] = norm(tdata.askv)
    tdata.loc[:,'bidv'] = norm(tdata.bidv)
    tdata.loc[:,'volume'] = norm(tdata.volume)
    # print(type(tdata.isnull()))
    if pd.isnull(tdata).any().any():
        print('drop it')
        raise ValueError
    return tdata

def daily_data_generator(obj, dates, cwd='.'):
    dir_path = os.path.join(cwd, obj)
    for date in dates:
        file_path = os.path.join(dir_path, date+'.csv')
        if not os.path.exists(file_path):
            continue
        data = preprocess(dir_path, date+'.csv')
        yield data
            # except IndexError:
            #     pass

def daterange(start_date, end_date):
    for n in range(int ((end_date - start_date).days)):
        yield start_date + timedelta(n)

if __name__ == '__main__':

    start_date = date(2021, 1, 1)
    end_date = date(2021, 6, 2)
    bl = 100
    tl = 600

    dates = [n.strftime("%Y%m%d") for n in daterange(start_date, end_date)]
    env = TestMarket(back_length=bl, time_limit=tl)

    for data in daily_data_generator('rb', dates, '/Data/database/data_zltick'):
        trade_data = data.iloc[:bl+tl]
        env.reset(trade_data)
        done = False
        action = 0
        while not done:
            state, reward, done, _ = env.step(action)
            if state[0] == 1:
                action = 1
        print(reward)
        break
