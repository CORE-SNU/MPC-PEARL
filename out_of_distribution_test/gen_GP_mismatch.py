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
from rlkit.torch.sac.policies import MakeDeterministic
from rlkit.samplers.util_gp import rollout
from rlkit.torch.sac.sac import PEARLSoftActorCritic

from multiprocessing import Pool

# from gp_mpc import InitGP, GP

class fake_agent():
    def __init__(self):
        self.gp_predictions = 0

def wraped_rollout(arg):
    variant, idx = arg
   
    env = NormalizedBoxEnv(ENVS[variant['env_name']](**variant['env_params']))
    env.reset_task_out_of_distribution(idx)
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
    # Rollout environment in parallel to generate GP
    env = NormalizedBoxEnv(ENVS[variant['env_name']](**variant['env_params']))
    tasks = list(range(180, 205))
    obs_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))
    eval_tasks=list(tasks[-variant['n_eval_tasks']:])

    gp_predictions = {i: {'mean':[], 'cov':[]} for i in tasks}
    agent = fake_agent()

    start = time.time()
    args = [(variant, idx) for idx in tasks]

    # multiprocessing to get GPR data
    with Pool(25) as p:
        res = p.map(wraped_rollout, args)

    for idx, path in enumerate(res):
        gp_predictions[idx+180]['mean'], gp_predictions[idx+180]['cov'] = path['prediction_mean'], path['prediction_cov']

    print('## testing time / task : ',(time.time() - start)/len(args))

    with open('gp_human_fast.pickle', 'wb') as handle:
        pickle.dump(gp_predictions, handle, protocol=pickle.HIGHEST_PROTOCOL)

def deep_update_dict(fr, to):
    ''' update dict of dicts with new values '''
    # merge default configuration setting into task-specific configuration setting
    # assume dicts have same keys
    for k, v in fr.items():
        if type(v) is dict:
            deep_update_dict(v, to[k])
        else:
            to[k] = v
    return to

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
