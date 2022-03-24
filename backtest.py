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
# args.init_before_training()

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
        yield date, data.values
            # except IndexError:
            #     pass

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

    def __init__(self, obj, age, device):
        name = 'DRL'
        cwd = f'./res/{obj}'
        super().__init__(name, cwd)
        self.age = age
        self.device = device
    
    def policy(self, state):
        ten_s = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        ten_a = self.age.act(ten_s)
        action = ten_a.argmax(dim=1).cpu().numpy()[0]
        return action


if __name__ == '__main__':
    torch.set_grad_enabled(False)

    obj = sys.argv[1]
    lv = 90
    gpu_id = int(sys.argv[2])
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
    args.cwd = './executioner_D3QN_w'
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

    env = TestMarket(back_length=bl, time_limit=tl)
    env = build_env(env, args.env_func, env_args)
    age = init_agent(args, gpu_id)
    # device = torch.device(f'cuda:{args.learner_gpus}')
    device = torch.device('cpu')

    start_date = date(2021, 7, 1)
    end_date = date(2021, 7, 2)

    dates = [n.strftime("%Y%m%d") for n in daterange(start_date, end_date)]

    outputs = [] 
    # exec = HDAgent(600, obj)
    exec = RLAgent(obj, age, device)
    # for obj in ['m', 'ag', 'al', 'v', 'pp', 'SA', 'rb']:
    for dat, data in daily_data_generator(obj, dates, '/Data/database/data_zltick'):
        tot = []
        steps = []
        fin = []
        for begin in range(bl, data.shape[0]-tl, 30):
            trade_data = data[begin-bl+1:begin+tl+1]
            state = env.reset(trade_data)
            done = False
            stepn = 0
            action = exec.policy(state)
            while not done:
                state, reward, done, _ = env.step(action)  # different
                action = exec.policy(state)
                stepn += 1
            steps.append(stepn)
            tot.append(reward)
            fin.append(1 if reward>0 else 0)
        exec.logging(dat, tot, steps, fin)
    exec.output()
