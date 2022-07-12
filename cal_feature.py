import numpy as np
import pandas as pd
import os
import sys
from datetime import timedelta, date
from env import *
from elegantrl.run import *
import torch
import elegantrl.agent as agent
from elegantrl.config import Arguments
from util import *
from multiprocessing import Process, Manager, Pool
import time
import signal
# args.init_before_training()
def daily_data_generator(obj, dates, cwd='.'):
    dir_path = os.path.join(cwd, obj)
    for date in dates:
        file_path = os.path.join(dir_path, date+'.csv')
        if not os.path.exists(file_path):
            continue
        data, contract = preprocess(dir_path, date+'.csv')
        yield date, data.values

def log_data_feature_cal(dat, fea_func, fea_name, obj='AP',
                       log_path='/Data/database/logs_thunder',
                       tick_path='/Data/database/data_zltick/', **kwargs):
        # [x.agg_cost, x.pss_cost, ]
    tick_path = os.path.join(tick_path, obj)
    log_path = os.path.join(log_path, dat)
    if not os.path.exists(os.path.join(tick_path, dat+'.csv')):
        return
    if not os.path.exists(os.path.join(log_path,'acc_sub.txt')):
        return
    try:
        a, acc = SingleDayHDAll('acc_sub.txt', log_path, dat) 
    except KeyError:
        print('keyerror!')
        return

    tick_data, contract = preprocess(tick_path, dat+'.csv')
    tick_data = tick_data.reset_index()#[['datetime','orig_v']]
    tick_data['ticktime'] = tick_data.datetime.astype('str').apply(lambda x:x.split(' ')[1])
    tick_data['tickdate'] = tick_data.datetime.dt.date
    last_date = tick_data.tickdate.iloc[0]

    last_data, _ = preprocess(tick_path, last_date.strftime("%Y%m%d")+'.csv')
    last_data = last_data.reset_index()
    last_data['ticktime'] = last_data.datetime.astype('str').apply(lambda x:x.split(' ')[1])
    last_data['tickdate'] = last_data.datetime.dt.date
    all_tick_len = len(last_data)

    tick_data = pd.concat([last_data, tick_data], axis=0)
    ll = len(tick_data)

    jud = (a.ticker==obj)&(a.symbol==contract)&\
            (((a.fx>0)&(a.odprice==a.bidprice)) |\
            ((a.fx<0)&(a.odprice==a.askprice))
            )
    a = a[jud][['symbol', 'agg_cost', 'pss_cost', 'ticktime', 'fx', 'filled_value', 'filled_lots',
       'filled_seconds', 'filled_date', 'pv', 'filled_price',]]
    a['ticktime'] = a.ticktime.apply(lambda x:x+'00' if x.split('.')[1]=='0' else x)

    feas = []
    times = []

    for _,trade in a.iterrows():
        
        if trade.isnull().any():
            continue

        tim = trade['ticktime']
        s, ms = tim.split('.')
        tims = s+'.000' if int(ms)<500 else s+'.500'
        tim = pd.to_datetime(trade.filled_date+' '+tims)

        if obj in ['m'] and int(tims.split(':')[0])>=21:
            # print('tim', tim, last_date)
            tim = pd.to_datetime(str(last_date)+' '+tims)
            # print('tim', tim, last_date)
        # print(type(day))
        # print(trade)
        # print(tick_data[tick_data.ticktime==tim])
        # assert 
        ord = tick_data.datetime.searchsorted(tim, side='right')-1
        # print(tick_data.iloc[ord,:5])
        # print(dat)
        # print(tim)
        # print(trade)
        # print(tick_data.iloc[ord-3:ord+3])
        # print('')
        # ord = tick_data[tick_data.ticktime==tim].index[0]
        # print(ord, all_tick_len)
        if ord-100+1<all_tick_len:
            continue
        if ord+120>=ll:
            continue
        # print(dat, tim)

        feas.append(fea_func(tick_data, ord, **kwargs))
        times.append(tims)
    return pd.DataFrame(list(zip(times, feas)), columns=['ticktime', fea_name])

def vol_ratio(tick_data, ord, sv_range, lv_range):
    # sv:较短的交易量统计区间,以tick为单位
    # lv:较长的交易量统计区间
    sv = tick_data.orig_v.iloc[ord-sv_range+1:ord+1].mean()
    # sv = tick_data.orig_v.iloc[ord:ord+sv_range+1].mean()
    lv = tick_data.orig_v.iloc[ord-lv_range+1:ord+1].mean()
    # lv = tick_data.orig_v.iloc[ord:ord+lv_range+1].mean()
    return sv/lv

def trend(tick_data, ord, length):
    diff = tick_data.midprice.iloc[ord-length+1:ord+1].diff()
    # diff = tick_data.midprice.iloc[ord:ord+length+1].diff()
    return (diff*diff.shift(1)<0).sum()
    # return np.random.rand()

def volatility(tick_data, ord, roc_range, std_range):
    # roc_range 使用具体的时间单位
    roc = tick_data.midprice.iloc[ord-std_range+1:ord+1:roc_range].diff()
        # .resample(roc_range, label='right').last()
    return roc.std()

def ema(tick_data, ord, length):
    a = tick_data.orig_midprice.iloc[ord-length+1:ord+1]
    return (a.max()-a.min())/tick_data.orig_midprice.iloc[ord]

def daterange(start_date, end_date):
    for n in range(int ((end_date - start_date).days)):
        yield start_date + timedelta(n)


def work(dat, fea_func, fea_name, obj, **kwargs):
    try:
        data = log_data_feature_cal(dat , fea_func, fea_name, obj=obj,
        **kwargs)
        pth = os.path.join('./feas', obj, dat+'.csv')
        last_data = pd.read_csv(pth)
        data = pd.concat((last_data, data.iloc[:,1]), axis=1)
        data.to_csv(pth, index=False)
    except Exception as e:
        print(e)


if __name__ == '__main__':

    start_date = date(2022, 1, 1)
    end_date = date(2022, 6, 29)

    dates = [n.strftime("%Y%m%d") for n in daterange(start_date, end_date)]
    
    obj = 'AP'
    model = 'exp003_fea'

    ps = Pool(192)

    data = pd.read_csv('symbol_instrumentid2.csv')
    objs = [i for i in data.pz.unique() if i not in ['AP','rb']]

    for obj in objs:
        if not os.path.exists(f'./feas/{obj}'):
            os.mkdir(f'./feas/{obj}')
        for dat in dates:
            if os.path.exists(os.path.join('./res', model, obj, dat+'.csv')):
                # ps.apply_async(work, kwds={'dat':dat, 'fea_func':vol_ratio,
                #  'fea_name':'vol_ratio', 'obj':obj, 
                #  'sv_range':360,'lv_range':3600})
                # ps.apply_async(work, kwds={'dat':dat, 'fea_func':trend,
                #  'fea_name':'trend', 'obj':obj, 'length':300})
                # ps.apply_async(work, kwds={'dat':dat, 'fea_func':volatility,
                # 'fea_name':'volatility', 'obj':obj,
                # 'roc_range':60, 'std_range':600})
                ps.apply_async(work, kwds={'dat':dat, 'fea_func':ema,
                 'fea_name':'ema', 'obj':obj, 'length':300})
    ps.close()
    ps.join()



    # data = log_data_feature_cal('20220321' ,volatility, 'vol', obj='rb',
    # roc_range=30, std_range=600)
    # print(data)
    # data.to_csv(os.path.join('./feas', obj, dat+'.csv'), index=False)