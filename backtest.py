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

def log_data_generator(dat, obj='rb',
                       log_path='/Data/database/logs_thunder',
                       tick_path='/Data/database/data_zltick/'):
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
    tick_data = tick_data.reset_index()
    tick_data['ticktime'] = tick_data.datetime.astype('str').apply(lambda x:x.split(' ')[1])
    tick_data['tickdate'] = tick_data.datetime.dt.date
    all_tick_len = len(tick_data)
    last_date = tick_data.tickdate.iloc[0]

    jud = (a.ticker==obj)&(a.symbol==contract)&\
            (((a.fx>0)&(a.odprice==a.bidprice)) |\
            ((a.fx<0)&(a.odprice==a.askprice))
            )
    a = a[jud][['symbol', 'agg_cost', 'pss_cost', 'ticktime', 'fx', 'filled_value', 'filled_lots',
       'filled_seconds', 'filled_date', 'pv', 'filled_price',]]
    a['ticktime'] = a.ticktime.apply(lambda x:x+'00' if x.split('.')[1]=='0' else x)

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
        # print(dat, tim)
        # print(trade)
        # print(tick_data[tick_data.ticktime==tim])
        # assert 
        ord = tick_data.datetime.searchsorted(tim, side='right')-1
        # print(dat)
        # print(tim)
        # print(trade)
        # print(tick_data.iloc[ord-3:ord+3])
        # print('')
        # ord = tick_data[tick_data.ticktime==tim].index[0]
        if ord-100+1<0:
            continue
        if ord+120>=all_tick_len:
            continue
        trade_data = tick_data.iloc[ord-100+1:ord+600+1]
        trade_data = trade_data[['bid','ask','bidv','askv','volume']]
        # trade_data['bid'] = trade_data['bid']/10
        # trade_data['ask'] = trade_data['ask']/10
        # yield trade.fx, trade.agg_cost, trade.pss_cost, trade_data.values
        yield trade, trade_data.values


def daterange(start_date, end_date):
    for n in range(int ((end_date - start_date).days)):
        yield start_date + timedelta(n)

class Agent(object):

    def __init__(self, name, cwd='.'):
        self.name = name
        self.cwd = cwd
        self.log = []

    def policy(self, state):
        raise NotImplementedError
    
    def logging(self, date, rewards, steps, fins):
        self.log.append([
            date,
            np.mean(rewards), np.std(rewards),
            np.mean(steps), np.std(steps),
            np.mean(fins)
        ])
    
    def output(self):
        path = os.path.join(self.cwd, self.name+'.csv')
        pd.DataFrame(self.log,
            columns=['date', 'avgR','stdR', 'avgS', 'stdS', 'fin']
        ).to_csv(path, index=False)
        
        self.log.clear()

class HDAgent(Agent):

    def __init__(self, hd_ticks, obj):
        name = f'HD{hd_ticks}'
        cwd = f'./res/{obj}'
        super().__init__(name, cwd)
        self.hd_ticks = hd_ticks

    def policy(self, state):
        return 1 if state[0] < 600-self.hd_ticks else 0

class LVAgent(Agent):

    def __init__(self, lv_units, obj):
        name = f'LV{lv_units}'
        cwd = f'./res/{obj}'
        super().__init__(name, cwd)
        self.lv_units = lv_units 

    def policy(self, state):
        return 1 if state[-5] > self.lv_units else 0
class RLAgent(Agent):

    def __init__(self, obj, age, device, name='DRL'):
        cwd = f'./res/{obj}'
        super().__init__(name, cwd)
        self.age = age
        self.device = device
    
    def policy(self, state):
        ten_s = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        ten_a = self.age.act(ten_s)
        action = ten_a.argmax(dim=1).cpu().numpy()[0]
        return action

def backtest(obj, gpus, comment):
    # gpu_id = int(sys.argv[2])
    gpu_id = gpus if gpus==0 else gpus.get()
    print(f'{obj} backtest begins, gpu id:{gpu_id}')
    mdir = '/Data/hongyuan'
    # direct = int(sys.argv[3])
    tick = pd.read_csv('symbol_instrumentid2.csv')
    tick = tick[tick.pz==obj].tick.values[0]
    chengshu = get_unit_cost(obj.lower())[1]
    buy_path = os.path.join(mdir, comment, f'{obj}1')
    sell_path = os.path.join(mdir, comment, f'{obj}-1')
    if not os.path.exists(buy_path) or not os.path.exists(sell_path):
        raise NotImplementedError
    if len(os.listdir(buy_path))==0 or len(os.listdir(sell_path))==0:
        raise NotImplementedError
    env_args = {
        'env_num': 1,
        'env_name': 'executioner',
        'max_step': 600,
        'state_dim': (100+1)*5,
        'action_dim': 2,
        'if_discrete': True,
        'target_return': 0.35
    }
    args = Arguments(agent.AgentD3QN, env_args=env_args)
    args.learner_gpus = gpu_id
    args.cwd = buy_path
    args.target_step = args.max_step * 50
    args.eval_times = 2 ** 10
    args.net_dim = 512
    args.batch_size = 1024
    args.gamma = 1
    args.explore_rate = 0.1
    args.seed = 123
    args.learning_rate = 2**-15
    args.eval_gap = 2 ** 6

    bl = 100
    tl = 600

    buyenv = TestMarket(back_length=bl, time_limit=tl, direct=1)
    buyenv = build_env(buyenv, args.env_func, env_args)
    sellenv = TestMarket(back_length=bl, time_limit=tl, direct=-1)
    sellenv = build_env(sellenv, args.env_func, env_args)
    age = init_agent(args, gpu_id)
    device = torch.device(f'cuda:{args.learner_gpus}')
    # device = torch.device('cpu')

    start_date = date(2022, 1, 4)
    end_date = date(2022, 3, 22)

    dates = [n.strftime("%Y%m%d") for n in daterange(start_date, end_date)]

    # exec = HDAgent(15, obj)
    buy_exec = RLAgent(obj, age, device, 'DRLBuy')

    # args.cwd = './executioner_D3QN_sell'
    args.cwd = sell_path
    args.learner_gpus = gpu_id
    age = init_agent(args, gpu_id)
    device = torch.device(f'cuda:{args.learner_gpus}')
    sell_exec = RLAgent(obj, age, device, 'DRLSell')
    summary = []


    for dat in dates:
        b = []
        c = []
        not_trade_date = True
        # for dir, agg, pss, trade in log_data_generator(dat):
        try:
            for trade, tdata in log_data_generator(dat, obj):
                if trade.fx == 1:
                    env = buyenv
                    exec = buy_exec
                elif trade.fx == -1:
                    env = sellenv
                    exec = sell_exec
                else:
                    raise ValueError
                not_trade_date = False
                state = env.reset(tdata)
                done = False
                stepn = 0
                action = exec.policy(state)
                while not done:
                    state, reward, done, _ = env.step(action)  # different
                    action = exec.policy(state)
                    stepn += 1
                b.append([stepn/2.0, tick*chengshu*(1-reward), trade.pss_cost, trade.agg_cost]) 
                c.append(trade)
            if not_trade_date:
                continue
            b = pd.DataFrame(b, columns=['stepn','drl','pss','agg']).reset_index().drop('index', axis=1)
            c = pd.DataFrame(c).reset_index().drop('index', axis=1)
            c = pd.concat([b,c], axis=1)

            c['drl'] = c.apply(lambda x:x.pss if x.filled_seconds<=x.stepn else x.drl, axis=1)
            c = c[['ticktime', 'drl','pss','agg','stepn','filled_seconds']]
            save_path = os.path.join('res', comment, obj)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            c.to_csv(os.path.join(save_path, str(dat)+'.csv'), index=False)
            summ = c.iloc[:,1:].sum()
            lens = len(c)
            summ['date'] = dat
            summ['avg_drl_sec'] = summ.stepn/lens
            summ['avg_pss_sec'] = summ.filled_seconds/lens
            summ = summ[['date','drl','pss','agg','avg_drl_sec','avg_pss_sec']]
            summary.append(summ.to_frame().T)
            print(summ.to_frame().T)
        except (ValueError, IndexError) as e:
            print(obj)
            print('data error!')
        except Exception as e:
            print(e)
        finally:
            print('gg')
    
    summary = pd.concat(summary, axis=0)
    summary.to_csv(os.path.join('res', comment, obj, 'sum.csv'), index=False)

    gpus.put(gpu_id)
    print(f'{obj} backtest ends successfully, release gpu id:{gpu_id}')
    os.kill(os.getpid(), signal.SIGTERM)



if __name__ == '__main__':
    torch.set_grad_enabled(False)

    comment = sys.argv[1]
    # comment = 'exp005'

    # objs = ['CF','nr','sc','rb','al','UR','fu','MA','c','SR','y','TA','eg','v','SA','FG','ru','zn','eb','a','SF','T','au','PK','TF','l','bu']
    data = pd.read_csv('symbol_instrumentid2.csv')
    # objs = data.pz.unique()
    objs = ['OI','bu','i','rb']
    # objs = ['rb']
    
    
    gpus = Manager().Queue(8)
    for i in range(8):
        gpus.put(i)

    print(f'main process:{os.getpid()}')

    ps = Pool(60)
    
    for obj in objs:
        ps.apply_async(backtest, (obj, gpus, comment))
    
    ps.close()
    ps.join()
    # for obj in ['m', 'ag', 'al', 'v', 'pp', 'SA', 'rb']:
    # env = buyenv
    # exec = buy_exec
    # for dat, data in daily_data_generator(obj, dates, '/Data/database/data_zltick'):
    #     tot = []
    #     steps = []
    #     fin = []
    #     for begin in range(bl, data.shape[0]-tl, 30):
    #         trade_data = data[begin-bl+1:begin+tl+1]
    #         state = env.reset(trade_data)
    #         done = False
    #         stepn = 0
    #         action = exec.policy(state)
    #         while not done:
    #             state, reward, done, _ = env.step(action)  # different
    #             action = exec.policy(state)
    #             stepn += 1
    #         steps.append(stepn)
    #         tot.append(reward)
    #         fin.append(1 if reward>0 else 0)
    #     exec.logging(dat, tot, steps, fin)
    # exec.output()
