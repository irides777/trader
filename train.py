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

obj_list = ['SM',
 'au',
 'hc',
 'lh',
 'cu',
 'PF',
 'p',
 'ZC',
 'sp',
 'i',
 'jm',
 'SF',
 'jd',
 'pg',
 'ni',
 'ag',
 'RM',
 'OI',
 'CY',
 'lu',
 'UR',
 'CJ',
 'cs',
 'sn',
 'j',
 'pb',
 'AP',
 'pp']

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

def get_multi_obj_datas(begin, end, objs=obj_list, cwd='/Data/database/data_zltick'):
    
    datas = []
    for obj in objs:
        data = get_train_datas(begin, end, obj=obj)
        datas.append(data)
    return np.concatenate(datas, axis=0)

def single_trainer(datas, direct, gpu_id, comment, name):

    mdir = '/Data/hongyuan'
    env = BaseMarket(datas=datas,back_length=100,time_limit=600, direct=direct)
    env_args = {
        'env_num': 1,
        'env_name': 'executioner',
        'max_step': 600,
        'state_dim': (100+1)*5+18,
        'action_dim': 2,
        'if_discrete': True,
        'target_return': 100
    }
    args = Arguments(agent.AgentD3QN, env_args=env_args)
    args.env = env
    args.target_step = args.max_step * 50
    args.eval_times = 2 ** 10
    args.net_dim = 512
    args.batch_size = 1024
    args.gamma = 1
    args.explore_rate = 0.05
    args.seed = 666
    args.learning_rate = 2**-15
    args.eval_gap = 2 ** 6
    args.learner_gpus = gpu_id
    args.cwd = os.path.join(mdir, comment, name+str(direct))
    args.break_step = 3e6
    train_and_evaluate(args)
    torch.cuda.empty_cache()


def single_obj_train(begin, end, obj, direct, gpu_id, comment, **kwargs):
    '''
    train single obj using gpu_id
    '''
    datas = get_train_datas(begin, end, obj=obj)
    print(f'{obj}{direct} training begins, gpu id:{gpu_id}')

    single_trainer(datas=datas, direct=direct, gpu_id=gpu_id, comment=comment, name=obj)
    
    print(f'{obj}{direct} training ends successfully, release gpu id:{gpu_id}')

def multi_cotrain(begin, end, name, direct, gpu_id, comment, **kwargs):
    '''
    train single obj using gpu_id
    '''
    datas = get_multi_obj_datas(begin, end)#, objs=kwargs['objs'])
    print(f'multi{name}{direct} training begins, gpu id:{gpu_id}')

    single_trainer(datas=datas, direct=direct, gpu_id=gpu_id, comment=comment, name=name)
    
    print(f'multi{name}{direct} training ends successfully, release gpu id:{gpu_id}')

def multi_trainer(begin, end, name, direct, gpus, comment, trainer):

    mdir = '/Data/hongyuan'
    print(os.path.join(mdir, comment,name+str(direct)))
    try:
        if False:#os.path.exists(os.path.join(mdir, comment,name+str(direct))):
            print(f'{name}{direct} has been trained')
        else:
            gpu_id = gpus.get()
            trainer(begin, end, name, direct, gpu_id, comment)
            gpus.put(gpu_id)
        os.kill(os.getpid(), signal.SIGTERM)
    except Exception as e:
        print(f'{name} error!')

def multi_objs_train(begin, end, names, comment, trainer=single_obj_train, **kwargs):

    gpus = Manager().Queue(8)
    for i in range(8):
        gpus.put(i)

    print(f'main process:{os.getpid()}')

    # print(len(objs))
    ps = Pool(120)
    for name in names:
        # if (not os.path.exists(buy_path) or not os.path.exists(sell_path))\
        #     or (len(os.listdir(buy_path))==0 or len(os.listdir(sell_path))==0):
        #     print(obj)
        ps.apply_async(multi_trainer, (begin, end, name , 1, gpus, comment, trainer))
        ps.apply_async(multi_trainer, (begin, end, name , -1, gpus, comment, trainer))
    ps.close()
    ps.join()
    print('ok')



if __name__ == '__main__':
    data = pd.read_csv('symbol_instrumentid2.csv')
    # objs = ['OI','bu','i','rb']
    # objs = ['all_list']
    objs = data.pz.unique()
    # objs = ['FG']
    # objs = ['CF','nr','sc','rb','al','UR','fu','MA','c','SR','y','TA','eg','v','SA','FG','ru','zn','eb','a','SF','T','au','PK','TF','l','bu']
    comment = sys.argv[1]

    multi_objs_train(20210101, 20220101, objs, comment, single_obj_train)
    # multi_objs_train(20210701, 20220101, objs, comment, multi_cotrain)
    # multi_cotrain(20210101,20220101,'znsnpb',1,0,comment)
    # multi_objs_train(20180101, 20220101, objs, comment)
    # single_obj_train(20210101, 20220102, 'rb', direct=1, gpu_id=0, comment=comment)
