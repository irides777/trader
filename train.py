import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import os
import sys
from env import *
import elegantrl.agent as agent
from elegantrl.config import Arguments
from elegantrl.run import train_and_evaluate
from util import *
from multiprocessing import Process, Manager, Pool
import time
import signal


def get_train_datas(begin, end, obj='rb', cwd='/Data/database/data_zltick'):
    cwd = os.path.join(cwd, obj)
    datas = []
    for _,__, files in os.walk(cwd):
        for file in files:
            if int(file[:8])<begin or int(file[:8])>end:
                continue
            try:
                data, contract = preprocess(cwd, file)
                datas.append(data)
            except (IndexError, ValueError):
                pass
    return datas

def train_trader(begin, end, obj, direct, gpus=0):

    try:
        datas = get_train_datas(begin, end, obj=obj)
        gpu_id = gpus if gpus==0 else gpus.get()
        print(f'{obj}{direct} training begins, gpu id:{gpu_id}')

        env = BaseMarket(datas=datas,back_length=100,time_limit=600, direct=direct)
        env_args = {
            'env_num': 1,
            'env_name': 'executioner',
            'max_step': 600,
            'state_dim': (100+1)*5,
            'action_dim': 2,
            'if_discrete': True,
            'target_return': 0.9
        }
        args = Arguments(agent.AgentD3QN,env=env, env_args=env_args)
        args.target_step = args.max_step * 50
        args.eval_times = 2 ** 10
        args.net_dim = 512
        args.batch_size = 1024
        args.gamma = 1
        args.explore_rate = 0.1
        args.seed = 666
        args.learning_rate = 2**-15
        args.eval_gap = 2 ** 6
        args.learner_gpus = gpu_id
        args.cwd = os.path.join(mdir, obj+str(direct))
        args.break_step = 4e6
        train_and_evaluate(args)
        torch.cuda.empty_cache()
        
        gpus.put(gpu_id)
        print(f'{obj}{direct} training ends successfully, release gpu id:{gpu_id}')
        os.kill(os.getpid(), signal.SIGTERM)
    except Exception as e:
        print(f'{obj} error!')

# def gogo(obj, gpus):
#     gpu_id = gpus.get()
#     np.random.seed()
#     sleep = np.random.randint(5,10)
#     print(f'begin {obj} {gpu_id} {sleep}')
#     time.sleep(sleep)
#     gpus.put(gpu_id)
#     print(f'end {obj} {gpu_id}')

mdir = '/Data/hongyuan'

if __name__ == '__main__':
    data = pd.read_csv('symbol_instrumentid2.csv')
    objs = ['rr', 'pb', 'sn', 'al']

    gpus = Manager().Queue(8)
    for i in range(8):
        gpus.put(i)

    print(f'main process:{os.getpid()}')

    # print(len(objs))
    ps = Pool(120)
    for obj in objs:
        ps.apply_async(train_trader, (20180101, 20220101, obj, 1, gpus))
        ps.apply_async(train_trader, (20180101, 20220101, obj, -1, gpus))
        # train_trader(20180101, 20220101, obj, 1)
        # train_trader(20180101, 20180601, obj, 1, gpus)
    ps.close()
    ps.join()
    print('ok')