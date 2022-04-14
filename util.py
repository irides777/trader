import datetime
import os
import numpy as np
import pandas as pd
import re

def get_unit_cost(name):
    name = name.lower()

    if name == 'a':
        cost = 10.0
        tick_size = 10
        weight = 40.0
    elif name == 'al':
        tick_size = 5
        cost = 25.0
        weight = 23.0
    elif name == 'ag':
        tick_size = 15.0
        cost = 18.0
        weight = 24.0
    elif name == 'au':
        tick_size = 1000
        cost = 25.0
        weight = 5.0
    elif name == 'bu':
        tick_size = 10
        cost = 20.0
        weight = 78.0
    elif name == 'c':
        tick_size = 10
        cost = 15.0
        weight = 98.0
    elif name == 'cf':
        tick_size = 5
        cost = 30.0
        weight = 19.0
    elif name == 'cs':
        tick_size = 10
        cost = 15.0
        weight = 80.0
    elif name == 'cu':
        tick_size = 5
        cost = 50.0
        weight = 8.0
    elif name == 'fg':
        tick_size = 20
        cost = 10.0
        weight = 65.0
    elif name == 'hc':
        tick_size = 10
        cost = 15.0
        weight = 52.0
    elif name == 'i':
        tick_size = 100
        cost = 50.0
        weight = 34.0
    elif name == 'j':
        tick_size = 100
        cost = 50.0
        weight = 10.0
    elif name == 'jm':
        tick_size = 60
        cost = 50.0
        weight = 20.0
    elif name == 'jd':
        tick_size = 10
        cost = 10.0
        weight = 10.0
    elif name == 'l':
        tick_size = 5
        cost = 25.0
        weight = 31.0
    elif name == 'm':
        tick_size = 10
        cost = 10.0
        weight = 51.0
    elif name == 'ma':
        tick_size = 10
        cost = 10.0
        weight = 65.0
    elif name == 'me':
        tick_size = 10
        cost = 10.0
        weight = 0.0
    elif name == 'ni':
        tick_size = 1
        cost = 15
        weight = 18.0
    elif name == 'oi':
        tick_size = 10
        cost = 15.0
        weight = 21.0
    elif name == 'p':
        tick_size = 10
        cost = 20.0
        weight = 25.0
    elif name == 'pb2':
        tick_size = 5
        cost = 25.0
        weight = 0.0
    elif name == 'pp':
        tick_size = 5
        cost = 10.0
        weight = 37.0
    elif name == 'rb':
        cost = 10.0
        tick_size = 10
        weight = 59.0
    elif name == 'pb':
        cost = 25.0
        tick_size = 5
        weight = 15.0
    elif name == 'rmz':
        tick_size = 10
        cost = 10.0
        weight = 0.0
    elif name == 'rm':
        tick_size = 10
        cost = 10.0
        weight = 64.0
    elif name == 'ru':
        tick_size = 10
        cost = 50.0
        weight = 11.0
    elif name == 'sm':
        tick_size = 5
        cost = 10.0
        weight = 35.0
    elif name == 'sf':
        tick_size = 5
        cost = 10.0
        weight = 35.0
    elif name == 'sr':
        tick_size = 10
        cost = 10.0
        weight = 26.0
    elif name == 'ta':
        tick_size = 5
        cost = 10.0
        weight = 59.0
    elif name == 'cj':
        tick_size = 5
        cost = 25.0
        weight = 22.0
    elif name == 'eg':
        tick_size = 10
        cost = 10.0
        weight = 30.0
    elif name == 'yy' or name == 'y':
        tick_size = 10
        cost = 20.0
        weight = 22.0
    elif name == 'zc':
        tick_size = 100
        cost = 24.0
        weight = 25.0
    elif name == 'ap':
        tick_size = 10
        cost = 30.0
        weight = 3.75
    elif name == 'sc':
        tick_size = 1000
        cost = 100.0
        weight = 3.8
    elif name == 'zn':
        tick_size = 5
        cost = 25.0
        weight = 16.0
    elif name == 'fu':
        tick_size = 10
        cost = 10.0
        weight = 55.0
    elif name == 'sp':
        tick_size = 10
        cost = 20.0
        weight = 30.0
    elif name == 't':
        tick_size = 10000
        cost = 50.0
        weight = 5.0
    elif name == 'tf':
        tick_size = 10000
        cost = 50.0
        weight = 5.0
    elif name == 'v':
        tick_size = 5
        cost = 25.0
        weight = 45.0
    elif name == 'b':
        tick_size = 10
        cost = 10.0
        weight = 40.0
    elif name == 'sn':
        tick_size = 1
        cost = 10.0
        weight = 12.0
    elif name == 'ur':
        tick_size = 20
        cost = 20.0
        weight = 49.0
    elif name == 'eb':
        tick_size = 5
        cost = 5.0
        weight = 45.0
    elif name == 'nr':
        tick_size = 10
        cost = 50.0
        weight = 15.0
    elif name == 'ss':
        tick_size = 5
        cost = 25
        weight = 22
    elif name == 'pg':
        tick_size = 20
        cost = 20
        weight = 23
    elif name == 'sf':
        tick_size = 5
        cost = 10
        weight = 49
    elif name == 'sa':
        tick_size = 20
        cost = 20
        weight = 50
    elif name == 'cy':
        tick_size = 5
        cost = 25
        weight = 14
    elif name == 'lu':
        tick_size = 10
        cost = 10
        weight = 40
    elif name == 'pf':
        tick_size = 5
        cost = 10
        weight = 30
    elif name == 'lh':
        tick_size = 16
        cost = 80
        weight = 2.5
    elif name == 'pk':
        tick_size = 5
        cost = 10
        weight = 20
    elif name == 'if':
        tick_size = 300
        cost = 60.0 * 2.5
        weight = 1.0
    elif name == 'yellow':
        tick_size = 200
        cost = 40.0 * 3.5
        weight = 0.8
    elif name == 'ic':
        tick_size = 200
        cost = 40.0 * 3.5
        weight = 0.8
    elif name == 'ih':
        tick_size = 300
        cost = 60.0 * 2.5
        weight = 1.0
    elif name == 'rr':
        tick_size = 10
        cost = 10.0
        weight = 30.0 
    else:
        print ('[get_unit_cost]Warning: no this commodity!, %s'%name)
        cost = 10.0
        tick_size = 10
        weight = 10.0
    cost = cost * 2.0 / tick_size
    return cost, tick_size, weight

def SingleDayHDAll(fname, srcdir, datenum0, slip_mode=0, exclude_comms=[]):
    
#     try:
        #slip_mode: 0 - lastprice; 1 - midprice
#     print fname
    f = open(os.path.join(srcdir, fname), encoding='gbk')

    date0 = datetime.datetime(int(str(datenum0)[:4]), int(str(datenum0)[4:6]), int(str(datenum0)[6:8]))
    lst_date = date0 - datetime.timedelta(days=1)


    today_datestr = date0.strftime('%Y-%m-%d')
    lst_datestr = lst_date.strftime('%Y-%m-%d')

    orders_reqid_dict = {}
    accs_setting = {}

    cnt = 0
    for line0 in f:
        try:
            account = line0.split('#')[0]
            line = line0.split('#')[1]
        except:
            continue
        
        if line.split(',')[0]=='PaperTrading':
            continue
        
        timestr = line[:8]
        line = line[10:]

        if int(timestr.split(':')[0])>20:
            datestr = lst_datestr
        else:
            datestr = today_datestr

        p = line.split(',')


        if p[0] == 'indb': # 新报单

            reqid = int(line.split('RequestID=')[1].split(',')[0])


            lastprice = float(line.split('lastprice=')[1].split(',')[0])
            bid = float(line.split('bidprice=')[1].split(',')[0])
            ask = float(line.split('askprice=')[1].split(',')[0])
            
            
            if len(line.split('bidvol='))>1:
                bidvol = float(line.split('bidvol=')[1].split(',')[0])
                askvol = float(line.split('askvol=')[1].split(',')[0])
            else:
                bidvol = np.nan
                askvol = np.nan

            odprice = float(re.split(':|,',line)[4]) 

            symbol = p[1].split(':')[0]
            ticker = re.match(r"([a-z]+)([0-9]+)", symbol, re.I).groups()[0]

            
            odtype = re.match(r"([a-z]+)([0-9]+)", p[1].split(':')[1], re.I).groups()[0]
            
            if len(line.split('nowtime='))>1:
                nowtime = (line.split('nowtime=')[1].split(',')[0]).split(' ')[-1]
            else:
                nowtime = timestr
            #print nowtime

            if p[1].split(':')[1][:3]=='Buy':
                fx = 1
            elif p[1].split(':')[1][:4]=='Sell':
                fx = -1
            else:
                break

            od = {}
            od['acc'] = account
            od['reqid'] = reqid
            od['datetime'] = datestr+' '+timestr
            od['ticktime'] = nowtime
            od['symbol'] = symbol
            od['ticker'] = ticker#.lower()
            od['odtype'] = odtype
            od['odprice'] = odprice
            od['midprice'] = (bid+ask)/2
            od['askprice'] = ask
            od['bidprice'] = bid
            od['askvol'] = askvol
            od['bidvol'] = bidvol
            od['lastprice'] = lastprice
            od['fx'] = fx
            od['filled_value'] = 0
            od['filled_lots'] = 0
            od['filled_time'] = np.nan
            od['filled_seconds'] = 0
            orders_reqid_dict['%s-%d'%(account, reqid)] = od

        elif p[0].split(':')[0] == u'成交回报':

            reqid = int(line.split('RequestID:')[1].split(',')[0])
            odkey = '%s-%d'%(account, reqid)
            if odkey not in orders_reqid_dict.keys():
                continue

            symbol = re.split(':|,',line)[1]

            if symbol!=orders_reqid_dict[odkey]['symbol']:
                continue

            price = float(re.split(':|,',line)[3])
            lots = int(re.split(':|,',line)[5])
            
            if len(line.split('TradeDate:'))>1:
                tradedate = line.split('TradeDate:')[1].split(',')[0]
                tradetime = line.split('TradeTime:')[1].split(',')[0]
            else:
                tradedate = datestr
                tradetime = timestr
            
            orders_reqid_dict[odkey]['filled_value'] += price*lots
            orders_reqid_dict[odkey]['filled_lots'] += lots
            orders_reqid_dict[odkey]['filled_date'] = tradedate
            orders_reqid_dict[odkey]['filled_time'] = tradetime
            orders_reqid_dict[odkey]['filled_seconds'] += lots*(pd.to_datetime(orders_reqid_dict[odkey]['filled_time'])-pd.to_datetime(orders_reqid_dict[odkey]['ticktime'])).total_seconds()
        
        elif p[0] == 'hdseconds':
            accs_setting[account] = int(p[2])
            
        cnt += 1

    f.close()

    orderdf = pd.DataFrame.from_dict(orders_reqid_dict,orient='index')
    

    orderdf.index.name='acc_requestid'

    orderdf['datetime'] = pd.to_datetime(orderdf['datetime'])
    
    orderdf['filled_seconds'] = orderdf['filled_seconds']/orderdf['filled_lots']
    orderdf.loc[orderdf['filled_seconds']<0, 'filled_seconds'] = 0

    orderdf['pv'] = orderdf['ticker'].apply(lambda x: get_unit_cost(x.lower())[1])
    orderdf['filled_price'] = orderdf['filled_value']/orderdf['filled_lots']
    
    orderdf = orderdf[~orderdf['ticker'].isin(exclude_comms)]


    orderdf['agg_cost'] = (orderdf['askprice']-orderdf['bidprice'])*orderdf['pv']

    orderdf['pss_cost'] = (orderdf['askprice']-orderdf['filled_price'])*orderdf['pv']
    orderdf.loc[orderdf['fx']>0, 'pss_cost'] = ((orderdf['filled_price']-orderdf['bidprice'])*orderdf['pv'])[orderdf['fx']>0]
    
    orderdf.loc[orderdf['pss_cost']<0, 'agg_cost'] = orderdf.loc[orderdf['pss_cost']<0, 'pss_cost']
    
    orderdf = orderdf.reset_index().set_index('datetime')


    return orderdf, accs_setting

def preprocess(tickpath='/Data/database/data_zltick/rb', filename='20220321.csv'):

    symb = pd.read_csv('symbol_instrumentid2.csv')
    symb = {a:b for a, b in zip(symb['pz'], symb['tick'])}
    obj = tickpath.split('/')[-1]

    tickdf = pd.read_csv(os.path.join(tickpath, filename),names=['date','time','ms','lastprice','volume','bid','bidv','ask','askv','opi','tur','contract'])
    contract = tickdf['contract'].iloc[0]
    tickdf[['lastprice','volume','bid','bidv','ask','askv','opi','tur']] = tickdf[['lastprice','volume','bid','bidv','ask','askv','opi','tur']].astype('float32')
    tickdf['datestr'] = tickdf['date'].apply(lambda x: '%s-%s-%s'%(str(x)[:4], str(x)[4:6], str(x)[6:8]))
    tickdf['timestr'] = tickdf['time'].apply(lambda x: '%s:%s:%s'%(format(x, '06d')[:-4], format(x, '06d')[-4:-2], format(x, '06d')[-2:]))
    tickdf['timestr'] = tickdf['timestr']+'.'+tickdf['ms'].apply(lambda x: format(x, '03d'))
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
    tickdf[['bid','ask']] = tickdf[['bid','ask']].fillna(method='ffill').fillna(tickdf.iloc[0]['lastprice'])
    tickdf['midprice'] = (tickdf['ask']+tickdf['bid'])/2.0

    tdata = tickdf[['bid','ask','bidv','askv','volume']].copy()
    std = tdata.iloc[0,0]
    tdata.loc[:,'bid'] = (tdata['bid']-std)/symb[obj]
    tdata.loc[:,'ask'] = (tdata['ask']-std)/symb[obj]

    def norm(x):
        return (x-x.mean())/x.std()

    tdata.loc[:,'askv'] = norm(tdata.askv)
    tdata.loc[:,'bidv'] = norm(tdata.bidv)
    tdata.loc[:,'volume'] = norm(tdata.volume)
    # print(type(tdata.isnull()))
    if pd.isnull(tdata).any().any():
        print('drop it')
        print(filename)
        raise ValueError
    return tdata, contract