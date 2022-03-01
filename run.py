# %%
import numpy as np
import pandas as pd

# %%
from env import *

# %%
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

# %%
import os
cwd = '/data/data_zltick/rb'
datas = []
for _,__, files in os.walk(cwd):
    for file in files:
        if int(file[:4])<2021:
            continue
        data = preprocess(cwd, file)
        datas.append(data)

# %%
# import matplotlib.pyplot as plt
# fig, ax1 = plt.subplots()

# ax2 = ax1.twinx()
# ax1.plot(tdata.bid[500:800])
# ax2.plot(tdata.volume[500:800], color='red')

# plt.show()

# %%
env = BaseMarket(datas=datas,back_length=100,time_limit=600)

# %%
tot = 0
env.seed(123)
for i in range(1024):
    done = False
    state = env.reset()
    action = 0
    while not done:
        state, reward, done, _ = env.step(action)
        if state[0] == 1:
            action = 1
    tot += reward
print(tot/1024)

# %%
env.seed(123)
tot = 0
for i in range(1024):
    done = False
    state = env.reset()
    while not done:
        action = env.action_space.sample()
        state, reward, done, _ = env.step(action)
    tot += reward
print(tot/1024)

# %%
import elegantrl.agent as agent
from elegantrl.config import Arguments
env_args = {
    'env_num': 1,
    'env_name': 'executioner',
    'max_step': 600,
    'state_dim': (100+1)*5,
    'action_dim': 2,
    'if_discrete': True,
    'target_return': 0.2
}

# %%
args = Arguments(agent.AgentD3QN,env=env, env_args=env_args)

# %%
from elegantrl.run import train_and_evaluate

# %%
args.target_step = args.max_step * 50
args.eval_times = 2 ** 10
args.net_dim = 512
args.batch_size = 1024
args.gamma = 0.95
args.explore_rate = 0.1
args.seed = 123
args.learning_rate = 2**-15
train_and_evaluate(args)

# %%
# age = args.agent().save_or_load_agent(cwd='./executioner_D3QN_0', if_save=False)
from elegantrl.run import *

# %%
import torch
torch.set_grad_enabled(False)
# args.init_before_training()
gpu_id = args.learner_gpus

'''init'''
env = build_env(args.env, args.env_func, args.env_args)

age = init_agent(args, gpu_id, env)

# %%
age.save_or_load_agent('./executioner_D3QN_0', if_save=False)

# %%
env.seed(1)
tot = 0
for i in range(1000):
    done = False
    state = env.reset()
    steps = 0
    while not done:
        steps += 1
        ten_s = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
        ten_a = age.act(ten_s).argmax(dim=1)
        state, reward, done, _ = env.step(ten_a[0].numpy())  # different
    tot += reward
    print(steps)
print(tot/1000)

# %%
tot = 0
env.seed(1)
for i in range(10000):
    done = False
    state = env.reset()
    action = 1
    while not done:
        state, reward, done, _ = env.step(action)
        action = 0
        if state[2] == 1:
            action = 2
    tot += reward
print(tot/10000)

# %%
tot = 0
env.seed(1)
for i in range(10000):
    done = False
    state = env.reset()
    action = 2
    while not done:
        state, reward, done, _ = env.step(action)
    tot += reward
print(tot/10000)

# %%
tot = 0
env.seed(1)
for i in range(10000):
    done = False
    state = env.reset()
    while not done:
        action = env.action_space.sample()
        state, reward, done, _ = env.step(action)
    tot += reward
print(tot/10000)

# %%
tmp = np.load(r'executioner_D3QN_0\replay_0.npz')

# %%
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('tkagg')
plt.plot(data['bid'])
plt.show()

# %%
tmp['buf_other']

# %%
a = [(1,2,3),(3,4,5),(5,6,5),(7,8,5)]
b = list(map(list, zip(*a)))

# %%
import torch
b


