import os
import shutil
import os.path as osp
import pickle
import json
import numpy as np
import click
import torch
import csv
import random
import time
from typing import List
import matplotlib.pyplot as plt


from rlkit.envs import ENVS
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.torch.sac.policies import TanhGaussianPolicy
from rlkit.torch.networks import FlattenMlp, MlpEncoder, RecurrentEncoder
from rlkit.torch.sac.agent import PEARLAgent
from configs.default import default_config
from launch_experiment import deep_update_dict
from rlkit.torch.sac.policies import MakeDeterministic
from rlkit.samplers.util import rollout
from rlkit.torch.sac.sac import PEARLSoftActorCritic

# from gp_mpc import InitGP, GP


def sim_policy(variant, path_to_exp, num_trajs=1, deterministic=False, save_video=False, mpc_only=False, name=''):
    '''
    simulate a trained policy adapting to a new task
    optionally save videos of the trajectories - requires ffmpeg

    :variant: experiment configuration dict
    :path_to_exp: path to exp folder
    :num_trajs: number of trajectories to simulate per task (default 1)
    :deterministic: if the policy is deterministic (default stochastic)
    :save_video: whether to generate and save a video (default False)
    '''

    # create multi-task environment and sample tasks
    env = NormalizedBoxEnv(ENVS[variant['env_name']](**variant['env_params']))
    tasks = env.get_all_task_idx()
    #tasks = env.get_gp_task_idx()
    obs_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))
    eval_tasks=list(tasks[-variant['n_eval_tasks']:])
    print('testing on {} test tasks, {} trajectories each'.format(len(eval_tasks), num_trajs))

    # instantiate networks
    latent_dim = variant['latent_size']
    context_encoder = latent_dim * 2 if variant['algo_params']['use_information_bottleneck'] else latent_dim
    reward_dim = 1
    net_size = variant['net_size']
    recurrent = variant['algo_params']['recurrent']
    encoder_model = RecurrentEncoder if recurrent else MlpEncoder

    qf1 = FlattenMlp(
        hidden_sizes=[net_size, net_size, net_size],
        input_size=obs_dim + action_dim + latent_dim,
        output_size=1,
    )
    qf2 = FlattenMlp(
        hidden_sizes=[net_size, net_size, net_size],
        input_size=obs_dim + action_dim + latent_dim,
        output_size=1,
    )
    vf = FlattenMlp(
        hidden_sizes=[net_size, net_size, net_size],
        input_size=obs_dim + latent_dim,
        output_size=1,
    )
    context_encoder = encoder_model(
        hidden_sizes=[200, 200, 200],
        input_size=2*obs_dim + action_dim + reward_dim,
        output_size=context_encoder,
    )
    policy = TanhGaussianPolicy(
        hidden_sizes=[net_size, net_size, net_size],
        obs_dim=obs_dim + latent_dim,
        latent_dim=latent_dim,
        action_dim=action_dim,
    )
    agent = PEARLAgent(
        latent_dim,
        context_encoder,
        policy,
        **variant['algo_params']
    )
    algorithm = PEARLSoftActorCritic(
        env=env,
        train_tasks=list(tasks[:variant['n_train_tasks']]),
        eval_tasks=list(tasks[-variant['n_eval_tasks']:]),
        nets=[agent, qf1, qf2, vf],
        latent_dim=latent_dim,
        **variant['algo_params']
    )
    
    # deterministic eval
    if deterministic:
        agent = MakeDeterministic(agent)

    # load trained weights (otherwise simulate random policy)
    context_encoder.load_state_dict(torch.load(os.path.join(path_to_exp, 'context_encoder.pth')))
    policy.load_state_dict(torch.load(os.path.join(path_to_exp, 'policy.pth')))
    computation_time_list = []

    tables = {}
    num_col_free = 0

    trajectories = {}

    # loop through tasks collecting rollouts
    dataset = {
        'rewards': [],
        'collisions_human': [],
        'collisions_table': [],
        'reaches': [],
        'navi_parts': [],
        'col_parts': [],
        'mpc_count': [],
        'pearl_count': []
    }

    video_frames = []
    algorithm.pretrain()

    gp_count, gp_fail, reaches = 0, 0, []
    # num_trajs = 5
    # eval_tasks = [eval_tasks[7]]
    # for idx in eval_tasks:
    total_time = 0
    start = time.time()

    for idx in eval_tasks:
        # print('task id :',idx)
        env.reset_task(idx)
        agent.clear_z()
        paths = []
        gp_human = [None]
        pred = None

        tables['task{}'.format(idx)] = env.restaurant.table_vector
        for n in range(num_trajs):
            
            if not variant['algo_params']['use_MPC']:
                env.eps = 0.
            '''
            # Call gp_human
            idx = env.task_idx
            for gp_idx, val in enumerate(eval_tasks):
                if val == idx:
                    #print(algorithm.gp_human_list[gp_idx])
                    gp_human = GP.load_dict(algorithm.gp_human_list[gp_idx])

            if gp_human is None:
                raise ValueError('No GP_human!')
            '''
            if n == num_trajs - 1:
                begin = time.time()

            path = rollout(env,
                           agent,
                           max_path_length=variant['algo_params']['max_path_length'],
                           accum_context=True,
                           animated=save_video,
                           use_MPC=variant['algo_params']['use_MPC'],
                           test=mpc_only,
                           mpc_solver=algorithm.mpc_forces,
                           pred=pred
                           )

            if n == num_trajs - 1:
                path_computation_time = time.time() - begin
                computation_time_list.append(path_computation_time)

            paths.append(path)

            pred = path['pred']
            # print('## %d iteration finished. ##'%(n))
            # print('Num_collisions : %.4f'%(np.sum(path['collisions_table'])+np.sum(path['collisions_human'])))
            # print('')
            
            # To check success rate
            if np.sum(path['reaches']):
                # print('yes!')
                reaches.append(1)
                   
            # print(path['reaches'])

            # raise ValueError ('Stop!')
            
            # To check number of trajs that use GP / not use GP
            use_GP, no_GP = path['gp_human']
            gp_count += use_GP
            gp_fail += no_GP

            gp_human = [None]

            if n >= variant['algo_params']['num_exp_traj_eval']:
                agent.infer_posterior(agent.context)

        trajectories['task{}'.format(idx)] = paths

        for key, val_list in dataset.items():
            val_list.append([sum(p[key]) for p in paths])

    """
    if save_video:
        # save frames to file temporarily
        temp_dir = os.path.join(path_to_exp, 'temp')
        frame_dir = os.path.join(path_to_exp, 'frame')
        os.makedirs(temp_dir, exist_ok=True)
        os.makedirs(frame_dir, exist_ok=True)
        for i, frm in enumerate(video_frames):
            frm.save(os.path.join(temp_dir, '%06d.jpg' % i))
            if i==len(video_frames):
                frm.save(os.path.join(frame_dir, '%06d.png' % i))

        video_filename=os.path.join(path_to_exp, 'video.mp4'.format(idx))
        # run ffmpeg to make the video
        os.system('ffmpeg -i {}/%06d.jpg -vcodec mpeg4 {}'.format(temp_dir, video_filename))
        # delete the frames
        shutil.rmtree(temp_dir)
    """
    print('Total testing time : ',time.time() - start)

    print('average computation time : ', np.mean(computation_time_list))
    # compute average returns across tasks
    n = min([len(a) for a in dataset['rewards']])

    rets = [a[:n] for a in dataset['rewards']]
    rets_mean = np.mean(np.stack(rets), axis=0)
    rets_std = np.std(np.stack(rets), axis=0)

    col_human = [a[:n] for a in dataset['collisions_human']]
    col_human_mean = np.mean(np.stack(col_human), axis=0)
    col_human_std = np.std(np.stack(col_human), axis=0)

    col_table = [a[:n] for a in dataset['collisions_table']]
    col_table_mean = np.mean(np.stack(col_table), axis=0)
    col_table_std = np.std(np.stack(col_table), axis=0)

    reaches = [a[:n] for a in dataset['reaches']]
    reaches_mean = np.mean(np.stack(reaches), axis=0)
    reaches_std = np.std(np.stack(reaches), axis=0)

    mpc_counts = [a[:n] for a in dataset['mpc_count']]
    mpc_counts_mean = np.mean(np.stack(mpc_counts), axis=0)
    mpc_counts_std = np.std(np.stack(mpc_counts), axis=0)

    pearl_counts = [a[:n] for a in dataset['pearl_count']]
    pearl_counts_mean = np.mean(np.stack(pearl_counts), axis=0)
    pearl_counts_std = np.std(np.stack(pearl_counts), axis=0)

    navi_parts = [a[:n] for a in dataset['navi_parts']]
    navi_parts_mean = np.mean(np.stack(navi_parts), axis=0)
    navi_parts_std = np.std(np.stack(navi_parts), axis=0)

    col_parts = [a[:n] for a in dataset['col_parts']]
    col_parts_mean = np.mean(np.stack(col_parts), axis=0)
    col_parts_std = np.std(np.stack(col_parts), axis=0)

    # ------------------------------------------------------------------------------------
    # ----------------------------- plot adapted trajectories ----------------------------
    num_people = env.num_people
    num_tables = env.num_tables
    table_radii = env.table_radii
    init_pos = env.init_state[:2]
    goal_pos = env._goal

    with open('./tables.pkl', 'wb') as f2:
        pickle.dump(tables, f2, pickle.HIGHEST_PROTOCOL)

    with open('./trajectories_{}.pkl'.format(name), 'wb') as f:
        pickle.dump(trajectories, f, pickle.HIGHEST_PROTOCOL)

    """
    for idx in eval_tasks:
        plt.clf()
        fig, ax = plt.subplots()

        plt.scatter(init_pos[0], init_pos[1], color='tab:red', marker='x')
        plt.annotate('start', tuple(init_pos))
        plt.scatter(goal_pos[0], goal_pos[1], color='tab:red', marker='x')
        plt.annotate('goal', tuple(goal_pos))

        adapted_before: List[np.ndarray] = obs_paths['task{}_{}'.format(idx, 0)]
        adapted_after: List[np.ndarray] = obs_paths['task{}_{}'.format(idx, num_trajs - 1)]
        table_pos: List[np.ndarray] = tables['task{}'.format(idx)]

        x_before = 5. * np.array(adapted_before)[:, 0]
        y_before = 5. * np.array(adapted_before)[:, 1]

        x_after = 5. * np.array(adapted_after)[:, 0]
        y_after = 5. * np.array(adapted_after)[:, 1]

        ax.plot(x_before, y_before, alpha=0.5, color='black')
        ax.plot(x_after, y_after, alpha=1., color='black')
        ax.set_xlim(-5., 5.)
        ax.set_ylim(-5., 5.)
        for i in range(num_people):
            x_human = 5. * np.array(adapted_before[:, 3 + 2 * i])
            y_human = 5. * np.array(adapted_before[:, 3 + 2 * i + 1])
            ax.plot(x_human, y_human, color='tab:red', linestyle='dashed')
        for i in range(num_tables):
            x_table = table_pos[2 * i]
            y_table = table_pos[2 * i + 1]
            obstacle = plt.Circle((x_table, y_table),
                                  table_radii,
                                  color='tab:blue'
                                  )
            ax.add_artist(obstacle)
        ax.grid(True)
        fig.savefig('task{}_{}.pdf'.format(idx, time.strftime("%m%d-%H%M%S")))
        """
    # ------------------------------------------------------------------------------------

    with open('results_total_{}.csv'.format(time.strftime("%m%d-%H%M%S")), 'w', newline='') as csvfile:
        fieldnames = ['rets_avg', 'rets_std', 'col_human_avg', 'col_human_std', 'col_table_avg', 'col_table_std', 'navi_parts_avg', 'navi_parts_std', 'col_parts_avg', 'col_parts_std', 'reaches_avg', 'reaches_std', 'mpc_counts_avg', 'mpc_counts_std', 'pearl_counts_avg', 'pearl_counts_std']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        dict_header = {name: name for name in fieldnames}
        writer.writerow(dict_header)
        
        for i, ret in enumerate(rets_mean):        
            # print('trajectory {}, avg return: {} \n'.format(i, ret))
            # print('trajectory {}, std return: {} \n'.format(i, rets_std[i]))
            writer.writerow({
                'rets_avg' : ret[0],
                'rets_std' : rets_std[i][0], 
                'col_human_avg' : col_human_mean[i],
                'col_human_std' : col_human_std[i], 
                'col_table_avg' : col_table_mean[i],
                'col_table_std' : col_table_std[i],
                'navi_parts_avg' : navi_parts_mean[i][0],
                'navi_parts_std' : navi_parts_std[i][0],
                'col_parts_avg' : col_parts_mean[i][0],
                'col_parts_std' : col_parts_std[i][0],
                'reaches_avg' : reaches_mean[i],
                'reaches_std' : reaches_std[i],
                'mpc_counts_avg' : mpc_counts_mean[i],
                'mpc_counts_std' : mpc_counts_std[i],
                'pearl_counts_avg' : pearl_counts_mean[i],
                'pearl_counts_std' : pearl_counts_std[i],
                })

    print('Successfully used GP :', gp_count)
    print('Failed to use GP :', gp_fail)
    return np.mean(computation_time_list)


@click.command()
@click.argument('config', default=None)
@click.argument('path', default=None)
@click.option('--num_trajs', default=10)
@click.option('--deterministic', is_flag=True, default=False)
@click.option('--video', is_flag=True, default=False)
@click.option('--mpc_only', is_flag=True, default=False)
@click.option('--name', is_flag=False, default='')
def main(config, path, num_trajs, deterministic, video, mpc_only, name):
    np.random.seed(666)
    variant = default_config
    if config:
        with open(osp.join(config)) as f:
            exp_params = json.load(f)
        variant = deep_update_dict(exp_params, variant)
    sim_policy(variant, path, num_trajs, deterministic, video, mpc_only, name)


if __name__ == "__main__":
    main()
