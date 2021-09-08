import numpy as np
from typing import List
from rlkit.samplers.util import rollout
from rlkit.torch.sac.policies import MakeDeterministic
import random


class InPlacePathSampler(object):
    """
    A sampler that does not serialization for sampling. Instead, it just uses
    the current policy and environment as-is.

    WARNING: This will affect the environment! So
    ```
    sampler = InPlacePathSampler(env, ...)
    sampler.obtain_samples  # this has side-effects: env will change!
    ```
    """
    def __init__(self, env, policy, max_path_length):
        self.env = env
        self.policy = policy

        self.max_path_length = max_path_length

    def start_worker(self):
        pass

    def shutdown_worker(self):
        pass

    def obtain_samples(self,
                       deterministic=False,
                       max_samples=np.inf,
                       max_trajs=np.inf,
                       accum_context=True,
                       train=True,
                       resample=1,
                       use_MPC=False,
                       mpc_solver=None,
                       ):
        """
        Obtains samples in the environment until either we reach either max_samples transitions or
        num_traj trajectories.
        The resample argument specifies how often (in trajectories) the agent will resample it's context.
        """
        assert max_samples < np.inf or max_trajs < np.inf, "either max_samples or max_trajs must be finite"
        policy = MakeDeterministic(self.policy) if deterministic else self.policy
        paths: List[dict] = []
        n_steps_total = 0
        n_trajs = 0
        # gp_human = None will build GP internally!
        gp_human = [None]
        idx = self.env.task_idx
        '''
        tasks = self.env.get_gp_task_idx()
        for gp_idx, val in enumerate(tasks):
            if val == idx:
                #print(algorithm.gp_human_list[gp_idx])
                gp_human = GP.load_dict(self.env.gp_human_list[gp_idx])
        '''
        gp_count, gp_fail = 0, 0
        while n_steps_total < max_samples and n_trajs < max_trajs:
            path = rollout(self.env,
                           policy,
                           max_path_length=self.max_path_length,
                           accum_context=accum_context,
                           train=train,
                           use_MPC=use_MPC,
                           mpc_solver=mpc_solver
                           )

            """ Below GP formualtion is not in use """
            """
            # Initialize GP to estimate dynamic obstacle
            # Do not update GP if already GP exists - need 1 GP for 1 task!
            gp_human = path['gp_human']
            if use_MPC and gp_human is None:
                solver_opts = {}
                solver_opts['expand'] = False
                X = path['observations']
                Y = path['next_observations']
                X = [x[2:] for x in X]
                Y = [y[2:] for y in Y]
                idx = random.sample(range(len(X)), 20)
                # idx = range(5,len(X))
                _idx = list(range(len(X)))
                for val in idx:
                    for i, val_compare in enumerate(_idx):
                        if val == val_compare:
                            _idx.pop(i)
                X_test = [X[i] for i in idx]
                Y_test = [Y[i] for i in idx]
                X_train = [X[i] for i in _idx]
                Y_train = [Y[i] for i in _idx]
                # gp_human = GP(np.array(X_train), np.array(Y_train), xlb=xlb, xub=xub, optimizer_opts=solver_opts, normalize=False)
                try:
                    gp_human = GP(np.array(X_train), np.array(Y_train), optimizer_opts=solver_opts, normalize=False)
                    SMSE, MNLP = gp_human.validate(np.array(X_test), np.array(Y_test))    # Need validation?
                except:
                    gp_human = None
                    print('GP error : cannot correctly estimate obstacle. Do not use GP')
            """
            # To check number of trajs that use GP / not use GP
            use_GP, no_GP = path['gp_human']
            gp_count += use_GP
            gp_fail += no_GP

            # Initialize gp_human to re-build GP
            gp_human = [None]

            # save the latent context that generated this trajectory
            path['context'] = policy.z.detach().cpu().numpy()
            paths.append(path)
            n_steps_total += len(path['observations'])
            n_trajs += 1
            # don't we also want the option to resample z over transition?
            if n_trajs % resample == 0:
                policy.sample_z()
        return paths, n_steps_total, gp_count, gp_fail

