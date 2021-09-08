import os
import shutil
import os.path as osp
import pickle5 as pickle
import json
import numpy as np
import click
import torch
import csv
import random
import time

from rlkit.envs import ENVS
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.torch.sac.policies import TanhGaussianPolicy
from rlkit.torch.networks import FlattenMlp, MlpEncoder, RecurrentEncoder
from rlkit.torch.sac.agent import PEARLAgent
from configs.default import default_config
from launch_experiment import deep_update_dict
from rlkit.torch.sac.policies import MakeDeterministic
from rlkit.samplers.util_gp import rollout
from rlkit.torch.sac.sac import PEARLSoftActorCritic

from multiprocessing import Pool

class fake_agent():
    def __init__(self):
        with open('./gp_human_fast.pickle', 'rb') as handle:
            self.gp_predictions = pickle.load(handle)

def wraped_rollout(arg):
    variant, idx = arg
    env = NormalizedBoxEnv(ENVS[variant['env_name']](**variant['env_params']))
    env.reset_task(idx)
    agent = fake_agent()
    path = rollout( env, 
                    agent, 
                    max_path_length=variant['algo_params']['max_path_length'],
                    accum_context=True,
                    animated=False,
                    use_MPC=True,
                    test=True,
                    mpc_solver=None,
                    pred=None
                    )
    return path

def sim_policy(variant):
    # create multi-task environment and sample tasks
    env = NormalizedBoxEnv(ENVS[variant['env_name']](**variant['env_params']))
    tasks = env.get_all_task_idx()
    obs_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))
    eval_tasks=list(tasks[-variant['n_eval_tasks']:])

    gp_predictions = {i: {'mean':[], 'cov':[]} for i in tasks}
    agent = fake_agent()

    start = time.time()
    args = [(variant, idx) for idx in tasks]

    # multiprocessing
    with Pool(25) as p:
        res = p.map(wraped_rollout, args)

    for idx, path in enumerate(res):
        gp_predictions[idx]['mean'], gp_predictions[idx]['cov'] = path['prediction_mean'], path['prediction_cov']

    print('## testing time / task : ',(time.time() - start)/len(args))
    
    with open('gp_human_fast.pickle', 'wb') as handle:
        pickle.dump(gp_predictions, handle, protocol=pickle.HIGHEST_PROTOCOL)


@click.command()
@click.argument('config', default=None)
def main(config):
    np.random.seed(666)
    variant = default_config
    if config:
        with open(osp.join(config)) as f:
            exp_params = json.load(f)
        variant = deep_update_dict(exp_params, variant)
    sim_policy(variant)


if __name__ == "__main__":
    main()
