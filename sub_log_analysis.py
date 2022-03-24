import os
import string
import sys
import time
import traceback
import chardet
import zmq
import numpy as np
import datetime
import re
import pandas as pd
import matplotlib.pyplot as plt


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
    f = open(os.path.join(srcdir, fname))

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
#     except:
#         print 'Error: ',fname, datenum0
#         return pd.DataFrame()

os.chdir(r'C:/LogSync/')

select_accs = []
not_select_accs = []

# select_accs = ['16610879']
# select_accs = ['91001727']
# select_accs = ['71310232']
# select_accs = ['16609398']
# not_select_accs = ['11801606', '88998510', 'yuanlan1']

dfcomm = pd.read_csv('symbol_instrumentid2.csv')
dfcomm['Ticker'] = dfcomm['pz'].apply(lambda x: x.lower())
dfcomm = dfcomm.set_index('Ticker')
dfcomm = dfcomm[~dfcomm.index.duplicated(keep='first')]



log_file = 'acc_sub.txt'
idatestr = '20220322'
datenum0 = int(idatestr)
srcdir = 'C:/LogSync/log/%d/'%datenum0
slip_mode = 2
rdf0, acc_setting= SingleDayHDAll(log_file, srcdir, datenum0) 
rdf0 = rdf0.sort_values(by='datetime')
# rdf0 = rdf0.dropna(how='any')

eps = 0.0001
rdf0['if_pss'] = 0
rdf0.loc[((rdf0['fx']>0)&(rdf0['odprice']<=rdf0['bidprice']+eps)), 'if_pss'] = 1
rdf0.loc[((rdf0['fx']<0)&(rdf0['odprice']>=rdf0['askprice']-eps)), 'if_pss'] = 1



print ('=========pss trades==========')
rdf0_pss = rdf0[rdf0['if_pss']==1]

accsdf = rdf0_pss.groupby('acc').agg({'agg_cost':'sum','pss_cost':'sum'})

accsdf['acc'] = accsdf.index


accsdf.loc['Total', 'agg_cost'] = accsdf['agg_cost'].sum()
accsdf.loc['Total', 'pss_cost'] = accsdf['pss_cost'].sum()

mom_twap_g1 = ['wk190422','yz008','YLAN26','YLAN158','wkzq191021','wkmr191112','666197']
zy_twap_g1 = ['11801606', '71310232', '71310223']

accsdf.loc['Mom_twap', 'agg_cost'] = accsdf[accsdf['acc'].isin(mom_twap_g1)]['agg_cost'].sum()
accsdf.loc['Mom_twap', 'pss_cost'] = accsdf[accsdf['acc'].isin(mom_twap_g1)]['pss_cost'].sum()

accsdf.loc['zy_twap', 'agg_cost'] = accsdf[accsdf['acc'].isin(zy_twap_g1)]['agg_cost'].sum()
accsdf.loc['zy_twap', 'pss_cost'] = accsdf[accsdf['acc'].isin(zy_twap_g1)]['pss_cost'].sum()

accsdf = accsdf.drop(columns=['acc'])

accsdf['hdseconds'] = 300
for iacc in accsdf.index:
    if iacc in acc_setting.keys():
        accsdf.loc[iacc, 'hdseconds'] = acc_setting[iacc]

accsdf['ratio'] = accsdf['pss_cost']/accsdf['agg_cost']

# avg_ratio = accsdf['pss_cost'].sum()/accsdf['agg_cost'].sum()
avg_ratio = accsdf.loc['Total', 'ratio']

print ('avg_ratio: ', avg_ratio)
print (accsdf.sort_values(by='agg_cost', ascending=False))

trades = len(rdf0_pss)
pss_trades = len(rdf0_pss[rdf0_pss['pss_cost']<=0])

titlename = '[pss_trades] avg_ratio: %.2f, trades: %d, pss_trades: %d, pss_ratio: %.2f'%(avg_ratio, trades, pss_trades, pss_trades/trades)
rdf0_pss[['agg_cost','pss_cost']].cumsum().plot(figsize=(16,9), grid=True, title=titlename)



print ('*************************')
print ('=========agg trades==========')

rdf0_agg = rdf0[rdf0['if_pss']==0]

accsdf = rdf0_agg.groupby('acc').agg({'agg_cost':'sum','pss_cost':'sum'})


accsdf.loc['Total', 'agg_cost'] = accsdf['agg_cost'].sum()
accsdf.loc['Total', 'pss_cost'] = accsdf['pss_cost'].sum()

accsdf['hdseconds'] = 300
for iacc in accsdf.index:
    if iacc in acc_setting.keys():
        accsdf.loc[iacc, 'hdseconds'] = acc_setting[iacc]

accsdf['ratio'] = accsdf['pss_cost']/accsdf['agg_cost']


avg_ratio = accsdf.loc['Total', 'ratio']

trades = len(rdf0_agg)
pss_trades = len(rdf0_agg[rdf0_agg['pss_cost']<=0])


print ('avg_ratio: ', avg_ratio)
print (accsdf.sort_values(by='agg_cost', ascending=False))

titlename = '[agg_trades] avg_ratio: %.2f, trades: %d, pss_trades: %d, pss_ratio: %.2f'%(avg_ratio, trades, pss_trades, pss_trades/trades)
rdf0_agg[['agg_cost','pss_cost']].cumsum().plot(figsize=(16,9), grid=True, title=titlename)


if len(select_accs)>0:
    rdf0 = rdf0[rdf0['acc'].isin(select_accs)]
    
if len(not_select_accs)>0:
    rdf0 = rdf0[~rdf0['acc'].isin(not_select_accs)]

rdf0['pss_filled'] = 0
rdf0.loc[rdf0['pss_cost']<=0, 'pss_filled'] = 1



rdf0 = rdf0.reset_index(drop=True)

alltickers = rdf0['ticker'].unique().tolist()
rdf0['lst_trade_seconds'] = 300

dflist = list()
for ticker in alltickers:
    tmpdf = rdf0[rdf0['ticker']==ticker]
    rdf0.loc[tmpdf.index, 'lst_trade_seconds'] = ((pd.to_datetime(tmpdf['ticktime'])-pd.to_datetime(tmpdf['ticktime'].shift(1))).apply(lambda x: x.total_seconds()))#.values


rdf0.loc[rdf0['lst_trade_seconds']<0, 'lst_trade_seconds'] = 300

acc_groups = {}
acc_groups['kxl'] = ['11801606', '2120007']
acc_groups['zlw'] = ['71310232', '16610879']
acc_groups['peng'] = ['16609398', '71310223']
acc_groups['yz'] = ['71310231', '16609561']

symbollimits = {}
symbollimits['ZC'] = 50
symbollimits['j'] = 50
symbollimits['jm'] = 50
symbollimits['CJ'] = 100
symbollimits['SF'] = 500

limit_ratio = 0.9

for ticker in symbollimits.keys():
    idf = rdf0[rdf0['ticker']==ticker]
    upper_limit = limit_ratio*symbollimits[ticker]
    for iacc in acc_groups.keys():
        tmpdf = idf[idf['acc'].isin(acc_groups[iacc])]
        sumlots = tmpdf[tmpdf['odtype'].isin(['BuyOpen','SellOpen'])]['filled_lots'].sum() 
        print (ticker, iacc, sumlots/upper_limit)
        
rdf0['bva_ratio'] = rdf0['bidvol']/rdf0['askvol']
rdf0.loc[rdf0['fx']<0, 'bva_ratio'] = (rdf0['askvol']/rdf0['bidvol'])[rdf0['fx']<0]

rdf0['bva_diff'] = rdf0['bidvol']-rdf0['askvol']
rdf0.loc[rdf0['fx']<0, 'bva_diff'] = (rdf0['askvol']-rdf0['bidvol'])[rdf0['fx']<0]

bvadf = rdf0[['acc', 'ticker', 'fx','bidvol', 'askvol', 'bva_ratio', 'bva_diff', 'pss_cost','agg_cost','if_pss']]
bvadf = bvadf[bvadf['if_pss']==1]
bvadf = bvadf.dropna()

bvadf['pss_ntick'] = bvadf['pss_cost']/bvadf['agg_cost']

rdf0 = rdf0.dropna(how='any')
dates = rdf0['filled_date'].astype(int).unique().tolist()

rdf0['trade_date'] = np.max(dates)
rdf0.loc[pd.to_datetime(rdf0['filled_time']).apply(lambda x: x.time())>datetime.time(20), 'trade_date'] = np.min(dates)

rdf0['filled_datetime'] = pd.to_datetime(rdf0['trade_date'].astype(str)+' '+rdf0['filled_time'])

rdf0 = rdf0.sort_values(by='filled_datetime')


tickerdf = rdf0.groupby('ticker')[['agg_cost','pss_cost']].sum()
tickerdf['ratio'] = tickerdf['pss_cost']/tickerdf['agg_cost']
tickerdf = tickerdf.sort_values(by='ratio', ascending=False)

longdf = rdf0[rdf0['fx']>0]
shortdf = rdf0[rdf0['fx']<0]

longdf = longdf.set_index('filled_datetime')

total_pss_trades = len(rdf0_pss)
for second in[5, 10, 20, 30, 40, 50, 60, 90, 120]:
    tmpdf = rdf0_pss[rdf0_pss['filled_seconds']<=second]
    print (second, len(tmpdf)*1.0/total_pss_trades)
    



# plt.scatter(bvadf['bva_ratio'].values, bvadf['pss_ntick'].values)