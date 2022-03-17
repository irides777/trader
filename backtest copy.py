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

if __name__ == '__main__':
    torch.set_grad_enabled(False)

    obj = sys.argv[1]
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
    device = torch.device(f'cuda:{args.learner_gpus}')

    start_date = date(2021, 10, 28)
    end_date = date(2022, 1, 1)

    dates = [n.strftime("%Y%m%d") for n in daterange(start_date, end_date)]

    ff = open('out.txt','w')
    sys.stdout = ff
    outputs = [] 
    # for obj in ['m', 'ag', 'al', 'v', 'pp', 'SA', 'rb']:
    for dat, data in daily_data_generator(obj, dates, '/Data/database/data_zltick'):
        tot = []
        steps = []
        for begin in range(bl, data.shape[0]-tl):
            trade_data = data[begin-bl+1:begin+tl+1]
            state = env.reset(trade_data)
            done = False
            stepn = 0
            a = []
            b = [state[-5:]]
            while not done:
                ten_s = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                ten_a = age.act(ten_s)
                a.append(ten_a)
                ten_a = ten_a.argmax(dim=1).cpu().numpy()[0]
                state, reward, done, _ = env.step(ten_a)  # different
                b.append(state[-5:])
                stepn += 1
            
            # action = 0
            # while not done:
            #     state, reward, done, _ = env.step(action)
            #     if state[0] == 1:
            #         action = 1
            steps.append(stepn)
            tot.append(reward)
            if reward < -10 :
                print(begin//2, begin//2/60, begin//2/3600)
                for aa, bb in zip(a,b):
                    print(bb)
                    print(aa)
                print(reward)
        outputs.append([dat, np.mean(tot), np.std(tot), np.mean(steps), np.std(steps)])
        break
    print(outputs)
    ff.close()
    # pd.DataFrame(outputs, columns=['date','avgR','stdR', 'avgS', 'stdS']).to_csv(obj+'.csv', index=False)
