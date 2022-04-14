import multiprocessing as mp
import os
import time

import numpy as np
import torch
from elegantrl.agent import AgentBase

from elegantrl.config import build_env, Arguments
from elegantrl.evaluator import Evaluator
from elegantrl.replay_buffer import ReplayBuffer, ReplayBufferList

'''[ElegantRL.2022.01.01](github.com/AI4Fiance-Foundation/ElegantRL)'''


def train_and_evaluate(args: Arguments):
    torch.set_grad_enabled(False)
    args.init_before_training()
    gpu_id = args.learner_gpus

    '''init'''
    env = build_env(args.env, args.env_func, args.env_args)

    agent = init_agent(args, gpu_id, env)
    buffer = init_buffer(args, gpu_id)
    evaluator = init_evaluator(args, gpu_id)

    agent.state = env.reset()
    if args.if_off_policy:
        trajectory = agent.explore_env(env, args.target_step)
        buffer.update_buffer((trajectory,))

    '''start training'''
    cwd = args.cwd
    break_step = args.break_step
    target_step = args.target_step
    if_allow_break = args.if_allow_break
    seed = args.seed
    del args

    if_train = True
    while if_train:
        trajectory = agent.explore_env(env, target_step)
        steps, r_exp = buffer.update_buffer((trajectory,))

        torch.set_grad_enabled(True)
        logging_tuple = agent.update_net(buffer)
        torch.set_grad_enabled(False)

        (if_reach_goal, if_save_agent) = evaluator.evaluate_save_and_plot(
            agent.act, steps, r_exp, logging_tuple, seed=seed
        )
        if if_save_agent:
            agent.save_or_load_agent(cwd, if_save=if_save_agent)
            buffer.save_or_load_history(cwd, if_save=True) if agent.if_off_policy else None

        dont_break = not if_allow_break
        not_reached_goal = not if_reach_goal
        stop_dir_absent = not os.path.exists(f'{cwd}/stop')
        if_train = (dont_break or not_reached_goal)\
            and evaluator.total_step <= break_step\
            and stop_dir_absent
    # if_save_agent = True
    print(f'| UsedTime: {time.time() - evaluator.start_time:.0f} | SavedDir: {cwd}')
    if if_save_agent:
        agent.save_or_load_agent(cwd, if_save=if_save_agent)
        buffer.save_or_load_history(cwd, if_save=True) if agent.if_off_policy else None


def init_agent(args: Arguments, gpu_id, env=None) -> AgentBase:
    agent = args.agent(args.net_dim, args.state_dim, args.action_dim, gpu_id=gpu_id, args=args)
    agent.save_or_load_agent(args.cwd, if_save=False)

    if env is not None:
        '''assign `agent.states` for exploration'''
        if args.env_num == 1:
            states = [env.reset(), ]
            assert isinstance(states[0], np.ndarray)
            assert states[0].shape in {(args.state_dim,), args.state_dim}
        else:
            states = env.reset()
            assert isinstance(states, torch.Tensor)
            assert states.shape == (args.env_num, args.state_dim)
        agent.states = states
    return agent


def init_buffer(args, gpu_id):
    if args.if_off_policy:
        buffer = ReplayBuffer(
            gpu_id=gpu_id,
            max_len=args.max_memo,
            state_dim=args.state_dim,
            action_dim=1 if args.if_discrete else args.action_dim,
        )
        buffer.save_or_load_history(args.cwd, if_save=False)

    else:
        buffer = ReplayBufferList()
    return buffer


def init_evaluator(args, gpu_id):
    eval_env = build_env(args.env, args.env_func, args.env_args)
    return Evaluator(cwd=args.cwd, agent_id=gpu_id, eval_env=eval_env, args=args)


'''train multiple process'''


def train_and_evaluate_mp(args: Arguments):
    args.init_before_training()

    # force all the multiprocessing to 'spawn' methods
    mp.set_start_method(method='spawn', force=True)  

    evaluator_pipe = PipeEvaluator()
    process = [mp.Process(target=evaluator_pipe.run, args=(args,))]
    worker_pipe = PipeWorker(args.worker_num)
    process.extend(
        [
            mp.Process(target=worker_pipe.run, args=(args, worker_id))
                for worker_id in range(args.worker_num)
        ]
    )

    learner_pipe = PipeLearner()
    process.append(mp.Process(target=learner_pipe.run, args=(args, evaluator_pipe, worker_pipe)))

    [p.start() for p in process]
    
    # waiting for learner
    process[-1].join()
    safely_terminate_process(process)


class PipeWorker:
    def __init__(self, worker_num):
        self.worker_num = worker_num
        self.pipes = [mp.Pipe() for _ in range(worker_num)]
        self.pipe1s = [pipe[1] for pipe in self.pipes]

    def explore(self, agent):
        act_dict = agent.act.state_dict()

        for worker_id in range(self.worker_num):
            self.pipe1s[worker_id].send(act_dict)

        # traj_lists
        return [pipe1.recv() for pipe1 in self.pipe1s]

    def run(self, args, worker_id):
        torch.set_grad_enabled(False)
        gpu_id = args.learner_gpus

        '''init'''
        env = build_env(args.env, args.env_func, args.env_args)
        agent = init_agent(args, gpu_id, env)

        '''loop'''
        target_step = args.target_step
        if args.if_off_policy:
            trajectory = agent.explore_env(env, args.target_step)
            self.pipes[worker_id][0].send(trajectory)
        del args

        while True:
            act_dict = self.pipes[worker_id][0].recv()
            agent.act.load_state_dict(act_dict)
            trajectory = agent.explore_env(env, target_step)
            self.pipes[worker_id][0].send(trajectory)


class PipeLearner:
    def __init__(self):
        pass

    @staticmethod
    def run(args, comm_eva, comm_exp):
        torch.set_grad_enabled(False)
        gpu_id = args.learner_gpus

        '''init'''
        agent = init_agent(args, gpu_id)
        buffer = init_buffer(args, gpu_id)

        '''loop'''
        if_train = True
        while if_train:
            traj_list = comm_exp.explore(agent)
            steps, r_exp = buffer.update_buffer(traj_list)

            torch.set_grad_enabled(True)
            logging_tuple = agent.update_net(buffer)
            torch.set_grad_enabled(False)

            if_train, if_save = comm_eva.evaluate_and_save_mp(agent.act, steps, r_exp, logging_tuple)
        agent.save_or_load_agent(args.cwd, if_save=True)
        print(f'| Learner: Save in {args.cwd}')

        if hasattr(buffer, 'save_or_load_history'):
            print(f"| LearnerPipe.run: ReplayBuffer saving in {args.cwd}")
            buffer.save_or_load_history(args.cwd, if_save=True)


class PipeEvaluator:
    def __init__(self):
        self.pipe0, self.pipe1 = mp.Pipe()

    def evaluate_and_save_mp(self, act, steps, r_exp, logging_tuple):
        if self.pipe1.poll():  # if_evaluator_idle
            if_train, if_save_agent = self.pipe1.recv()
            act_state_dict = act.state_dict().copy()  # deepcopy(act.state_dict())
        else:
            if_train = True
            if_save_agent = False
            act_state_dict = None

        self.pipe1.send((act_state_dict, steps, r_exp, logging_tuple))
        return if_train, if_save_agent

    def run(self, args):
        torch.set_grad_enabled(False)
        gpu_id = args.learner_gpus

        '''init'''
        agent = init_agent(args, gpu_id)
        evaluator = init_evaluator(args, gpu_id)

        '''loop'''
        cwd = args.cwd
        act = agent.act
        break_step = args.break_step
        if_allow_break = args.if_allow_break
        del args

        if_train = True
        if_reach_goal = False
        if_save_agent = False
        while if_train:
            act_dict, steps, r_exp, logging_tuple = self.pipe0.recv()

            if act_dict:
                act.load_state_dict(act_dict)
                if_reach_goal, if_save_agent = evaluator.evaluate_save_and_plot(
                    act, steps, r_exp, logging_tuple
                )
            else:
                evaluator.total_step += steps
                
            dont_break = not if_allow_break
            not_reached_goal = not if_reach_goal
            stop_dir_absent = not os.path.exists(f'{cwd}/stop')
            if_train = (dont_break or not_reached_goal)\
                and evaluator.total_step <= break_step\
                and stop_dir_absent
            
            self.pipe0.send((if_train, if_save_agent))

        print(f'| UsedTime: {time.time() - evaluator.start_time:>7.0f} | SavedDir: {cwd}')

        while True:  # wait for the forced stop from main process
            time.sleep(1943)


def safely_terminate_process(process):
    for p in process:
        try:
            p.kill()
        except OSError as e:
            print(e)
