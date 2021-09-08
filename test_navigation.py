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


def sim_policy(variant, path_to_exp, num_trajs=1, deterministic=False, save_video=False, mpc_only=False, save_path=None):
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

    # loop through tasks collecting rollouts
    dataset = {
        'rewards':[],
        'collisions_human':[],
        'collisions_table':[],
        'reaches':[],
        'navi_parts':[],
        'col_parts':[],
        'reaches':[],
        'mpc_count':[],
        'pearl_count':[],
    }

    video_frames = []
    algorithm.pretrain()

    gp_count, gp_fail, reaches = 0, 0, []
    num_eval_tasks = len(eval_tasks)
    collision_free_count = np.zeros((num_eval_tasks, num_trajs))
    reach_count = np.zeros((num_eval_tasks, num_trajs))
    task_completion_count = np.zeros((num_eval_tasks, num_trajs))
    computation_time = np.zeros((num_eval_tasks, num_trajs))


    # eval_tasks = [eval_tasks[7]]
    # for idx in eval_tasks:
    total_time = 0
    start = time.time()
    elapsed_list = []
    for idx in eval_tasks:

        # print('task id :',idx)
        env.reset_task(idx)
        agent.clear_z()
        paths = []
        gp_human = [None]
        pred = None
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
            # computation time measurement
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

            computation_time[idx-180, n] = time.time() - begin

            if np.sum(path['collisions_table']) + np.sum(path['collisions_human']) == 0:
                collision_free_count[idx-180, n] = 1.
                if np.sum(path['reaches']):
                    task_completion_count[idx-180, n] = 1.
            if np.sum(path['reaches']):
                reach_count[idx-180, n] = 1.

            paths.append(path)
            pred = path['pred']
            # print('## %d iteration finished. ##'%(n))
            # print('Num_collisions : %.4f'%(np.sum(path['collisions_table'])+np.sum(path['collisions_human'])))
            # print('')
            
            # To check number of trajs that use GP / not use GP
            use_GP, no_GP = path['gp_human']
            gp_count += use_GP
            gp_fail += no_GP

            gp_human = [None]

            if n >= variant['algo_params']['num_exp_traj_eval']:
                agent.infer_posterior(agent.context)

        for key, val_list in dataset.items():
            val_list.append([sum(p[key]) for p in paths])

    '''
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
    '''
    print('Total testing time : ',time.time() - start)

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
    if save_path is None:
        save_path = 'results_total_{}.csv'.format(time.strftime("%m%d-%H%M%S"))

    with open(save_path, 'w', newline='') as csvfile:
        fieldnames = ['rets_avg',
                      'rets_std',
                      'col_human_avg',
                      'col_human_std',
                      'col_table_avg',
                      'col_table_std',
                      'navi_parts_avg',
                      'navi_parts_std',
                      'col_parts_avg',
                      'col_parts_std',
                      'reaches_avg',
                      'reaches_std',
                      'mpc_counts_avg',
                      'mpc_counts_std',
                      'pearl_counts_avg',
                      'pearl_counts_std',
                      'collision_free_ratio',
                      'reach_ratio',
                      'task_completion_ratio',
                      'computation_time_avg'
                      ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        dict_header = {name:name for name in fieldnames}
        writer.writerow(dict_header)
        
        for i, ret in enumerate(rets_mean):        
            # print('trajectory {}, avg return: {} \n'.format(i, ret))
            # print('trajectory {}, std return: {} \n'.format(i, rets_std[i]))
            writer.writerow({
                'rets_avg': ret[0],
                'rets_std': rets_std[i][0],
                'col_human_avg': col_human_mean[i],
                'col_human_std': col_human_std[i],
                'col_table_avg': col_table_mean[i],
                'col_table_std': col_table_std[i],
                'navi_parts_avg': navi_parts_mean[i][0],
                'navi_parts_std': navi_parts_std[i][0],
                'col_parts_avg': col_parts_mean[i][0],
                'col_parts_std': col_parts_std[i][0],
                'reaches_avg': reaches_mean[i],
                'reaches_std': reaches_std[i],
                'mpc_counts_avg': mpc_counts_mean[i],
                'mpc_counts_std': mpc_counts_std[i],
                'pearl_counts_avg': pearl_counts_mean[i],
                'pearl_counts_std': pearl_counts_std[i],
                'collision_free_ratio': np.sum(collision_free_count[:, i]) / num_eval_tasks,
                'reach_ratio': np.sum(reach_count[:, i]) / num_eval_tasks,
                'task_completion_ratio': np.sum(task_completion_count[:, i]) / num_eval_tasks,
                'computation_time_avg': np.sum(computation_time[:, i]) / num_eval_tasks
                })

    print('Successfully used GP :', gp_count)
    print('Failed to use GP :', gp_fail)
    # return np.mean(elapsed_list), col_free_ratio


@click.command()
@click.argument('config', default=None)
@click.argument('path', default=None)
@click.option('--num_trajs', default=10)
@click.option('--deterministic', is_flag=True, default=False)
@click.option('--video', is_flag=True, default=False)
@click.option('--mpc_only', is_flag=True, default=False)
@click.option('--save_path', is_flag=False, default=None)
def main(config, path, num_trajs, deterministic, video, mpc_only, save_path):
    np.random.seed(666)
    variant = default_config
    if config:
        with open(osp.join(config)) as f:
            exp_params = json.load(f)
        variant = deep_update_dict(exp_params, variant)
    sim_policy(variant, path, num_trajs, deterministic, video, mpc_only, save_path)


def main_external_call(config, path, num_trajs, deterministic, video, mpc_only, save_path):
    np.random.seed(666)
    variant = default_config
    if config:
        with open(osp.join(config)) as f:
            exp_params = json.load(f)
        variant = deep_update_dict(exp_params, variant)
    sim_policy(variant, path, num_trajs, deterministic, video, mpc_only, save_path)
    return


if __name__ == "__main__":
    main()
