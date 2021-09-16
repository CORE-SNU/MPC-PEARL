import abc
from collections import OrderedDict
import time
import pickle5 as pickle
from typing import List, Dict
import gtimer as gt
import numpy as np

from rlkit.core import logger, eval_util
from rlkit.data_management.env_replay_buffer import MultiTaskReplayBuffer
from rlkit.data_management.path_builder import PathBuilder
from rlkit.samplers.in_place import InPlacePathSampler
from rlkit.torch import pytorch_util as ptu


class MetaRLAlgorithm(metaclass=abc.ABCMeta):
    def __init__(
            self,
            env,
            agent,
            train_tasks,
            eval_tasks,
            meta_batch=64,
            num_iterations=100,
            num_train_steps_per_itr=1000,
            num_initial_steps=100,
            num_tasks_sample=100,
            num_steps_prior=100,
            num_steps_posterior=100,
            num_extra_rl_steps_posterior=100,
            num_evals=10,
            batch_size=1024,
            embedding_batch_size=1024,
            embedding_mini_batch_size=1024,
            max_path_length=1000,
            discount=0.99,
            replay_buffer_size=1000000,
            reward_scale=1,
            num_exp_traj_eval=1,
            update_post_train=1,
            eval_deterministic=True,
            render=False,
            num_steps_per_eval=600,
            save_replay_buffer=False,
            save_algorithm=False,
            save_environment=False,
            render_eval_paths=False,
            dump_eval_paths=False,
            plotter=None,
            use_MPC=False
        ):
        """
        :param env: training env
        :param agent: agent that is conditioned on a latent variable z that rl_algorithm is responsible for feeding in
        :param train_tasks: list of tasks used for training
        :param eval_tasks: list of tasks used for eval

        see default experiment config file for descriptions of the rest of the arguments
        """
        self.use_MPC = use_MPC
        self.env = env
        self.agent = agent
        # Can potentially use a different policy purely for exploration rather than also solving tasks
        # currently not being used
        self.exploration_agent = agent
        self.train_tasks = train_tasks
        self.eval_tasks = eval_tasks
        self.meta_batch = meta_batch
        self.num_iterations = num_iterations
        self.num_train_steps_per_itr = num_train_steps_per_itr
        self.num_initial_steps = num_initial_steps
        self.num_tasks_sample = num_tasks_sample
        self.num_steps_prior = num_steps_prior
        self.num_steps_posterior = num_steps_posterior
        self.num_extra_rl_steps_posterior = num_extra_rl_steps_posterior
        self.num_evals = num_evals
        self.num_steps_per_eval = num_steps_per_eval
        self.batch_size = batch_size
        self.embedding_batch_size = embedding_batch_size
        self.embedding_mini_batch_size = embedding_mini_batch_size
        self.max_path_length = max_path_length
        self.discount = discount
        self.replay_buffer_size = replay_buffer_size
        self.reward_scale = reward_scale
        self.update_post_train = update_post_train
        self.num_exp_traj_eval = num_exp_traj_eval
        self.eval_deterministic = eval_deterministic
        self.render = render
        self.save_replay_buffer = save_replay_buffer
        self.save_algorithm = save_algorithm
        self.save_environment = save_environment
        
        self.eval_statistics = None
        self.render_eval_paths = render_eval_paths
        self.dump_eval_paths = dump_eval_paths
        self.plotter = plotter

        self.sampler = InPlacePathSampler(
            env=env,
            policy=agent,
            max_path_length=self.max_path_length,
        )

        # separate replay buffers for
        # - training RL update
        # - training encoder update
        self.replay_buffer = MultiTaskReplayBuffer(
                self.replay_buffer_size,
                env,
                self.train_tasks,
            )

        self.enc_replay_buffer = MultiTaskReplayBuffer(
                self.replay_buffer_size,
                env,
                self.train_tasks,
        )

        self._n_env_steps_total = 0
        self._n_train_steps_total = 0
        self._n_rollouts_total = 0
        self._do_train_time = 0
        self._epoch_start_time = None
        self._algo_start_time = None
        self._old_table_keys = None
        self._current_path_builder = PathBuilder()
        self._exploration_paths = []
        self.mpc_forces = None
        self.gp = None
        
    def make_exploration_policy(self, policy):
         return policy

    def make_eval_policy(self, policy):
        return policy

    def sample_task(self, is_eval=False):
        '''
        sample task randomly
        '''
        if is_eval:
            idx = np.random.randint(len(self.eval_tasks))
        else:
            idx = np.random.randint(len(self.train_tasks))
        return idx

    def train(self):
        '''
        meta-training loop
        '''
        self.pretrain()
        params = self.get_epoch_snapshot(-1)
        logger.save_itr_params(-1, params)
        gt.reset()
        gt.set_def_unique(False)
        self._current_path_builder = PathBuilder()

        # at each iteration, we first collect data from tasks, perform meta-updates, then try to evaluate
        for it_ in gt.timed_for(
                range(self.num_iterations),
                save_itrs=True,
        ):
            self._start_epoch(it_)
            self.training_mode(True)
            
            # MPC rate scheduler
            # self.env.interval = int(self.env.interval_list[it_])

            print('MPC interval :', self.env.interval)

            if it_ == 0:
                print('collecting initial pool of data for train and eval')
                # temp for evaluating
                for idx in self.train_tasks:
                    # Before training, initialize each task buffer with a sufficient number of contexts.
                    self.task_idx = idx
                    self.env.reset_task(idx)
                    self.collect_data(self.num_initial_steps, 1, np.inf)

            # Sample data from train tasks.
            for i in range(self.num_tasks_sample):
                idx = np.random.randint(len(self.train_tasks))
                self.task_idx = idx
                self.env.reset_task(idx)
                self.enc_replay_buffer.task_buffers[idx].clear()
                # collect some trajectories with z ~ prior
                if self.num_steps_prior > 0:
                    self.collect_data(self.num_steps_prior, 1, np.inf)
                # collect some trajectories with z ~ posterior
                if self.num_steps_posterior > 0:
                    self.collect_data(self.num_steps_posterior, 1, self.update_post_train)
                # even if encoder is trained only on samples from the prior, the policy needs to learn to handle z ~ posterior
                if self.num_extra_rl_steps_posterior > 0:
                    self.collect_data(self.num_extra_rl_steps_posterior,
                                      1,
                                      self.update_post_train,
                                      add_to_enc_buffer=False
                                      )

            # Sample train tasks and compute gradient updates on parameters.
            for train_step in range(self.num_train_steps_per_itr):
                indices = np.random.choice(self.train_tasks, self.meta_batch)
                self._do_training(indices)
                self._n_train_steps_total += 1
            gt.stamp('train')

            self.training_mode(False)

            # eval
            if it_ % 3 == 0:
                self._try_to_eval(it_)
                gt.stamp('eval')

            self._end_epoch()

    def pretrain(self):
        """
        Do anything before the main training phase.
        """
        if self.use_MPC:
            self.mpc_forces = self.create_MPC()
            self.load_gp_human()

    def load_gp_human(self):
        with open('./gp_human_fast.pickle', 'rb') as handle:
            self.agent.gp_human = pickle.load(handle)

    def create_MPC(self):
        """ Dynamic Model options"""
        print('Creating MPC...')
        from rlkit.core.MPC_CVaR import MPC_forces
        Nu = self.env.action_space.shape[0]
        # Nx = 2 when we don't consider obstacle dynamics
        #Nx = self.env.observation_space.shape[0]
        Nx = 3
        Ny = 2

        normalize = False  # Option to normalize data in GP model

        """ Limits in the training data """
        #ulb = np.array([-1.0, -1.0])
        #uub = np.array([1.0, 1.0])
        #xlb = np.array([-1.3, -1.3])
        #xub = np.array([1.3, 1.3])
        ulb = self.env._wrapped_env.action_space.low.tolist()
        uub = self.env._wrapped_env.action_space.high.tolist()
        uub[0] = 5
        xlb = self.env.observation_space.low[:2].tolist()
        xub = self.env.observation_space.high[:2].tolist()
        
        # Penalty matrices
        Q = np.diag([1., 1.])
        R = np.diag([.2, .2])
        S =  0*R
        Qf = 20*Q
        
        mpc_forces = MPC_forces(nx=Nx, ny=Ny, nu=Nu, T=self.env.Nt,
                                Q=Q, Qf=Qf, R=R, S=S,
                                ulb=ulb, uub=uub, xlb=xlb, xub=xub, deltaulb=[-0.3, -np.pi / 5.], deltauub=[0.3, np.pi / 5.],
                                L=self.env.L, lr=self.env.lr,
                                N_table=self.env.num_tables, N_human=self.env.N_adapt, dt=self.env.dt,
                                safe_rad=self.env.safe_rad
                                )

        return mpc_forces

    def collect_data(self, num_samples, resample_z_rate, update_posterior_rate, add_to_enc_buffer=True):
        '''
        get trajectories from current env in batch mode with given policy
        collect complete trajectories until the number of collected transitions >= num_samples

        :param agent: policy to rollout
        :param num_samples: total number of transitions to sample
        :param resample_z_rate: how often to resample latent context z (in units of trajectories)
        :param update_posterior_rate: how often to update q(z | c) from which z is sampled (in units of trajectories)
        :param add_to_enc_buffer: whether to add collected data to encoder replay buffer
        '''
        # start from the prior
        self.agent.clear_z()

        # TODO: Check if task is fixed in this loop
        # TODO2: train GP along with z
        # TODO3: if clear_z is activated, we should re-train GP! - Add below loop for every clear_z()
        num_transitions = 0
        while num_transitions < num_samples:
            paths, n_samples, gp_count, gp_fail = self.sampler.obtain_samples(
                                                           max_samples=num_samples - num_transitions,
                                                           max_trajs=update_posterior_rate,
                                                           accum_context=False,
                                                           train=True,
                                                           resample=resample_z_rate,
                                                           use_MPC=self.use_MPC,
                                                           mpc_solver=self.mpc_forces	
                                                           )
            num_transitions += n_samples
            self.replay_buffer.add_paths(self.task_idx, paths)
            if add_to_enc_buffer:
                self.enc_replay_buffer.add_paths(self.task_idx, paths)
            if update_posterior_rate != np.inf:
                context = self.sample_context(self.task_idx)
                self.agent.infer_posterior(context)
        self._n_env_steps_total += num_transitions
        gt.stamp('sample')

    def _try_to_eval(self, epoch):
        logger.save_extra_data(self.get_extra_data_to_save(epoch))
        if self._can_evaluate():
            self.evaluate(epoch)

            params = self.get_epoch_snapshot(epoch)
            logger.save_itr_params(epoch, params)
            table_keys = logger.get_table_key_set()
            if self._old_table_keys is not None:
                assert table_keys == self._old_table_keys, (
                    "Table keys cannot change from iteration to iteration."
                )
            self._old_table_keys = table_keys

            logger.record_tabular(
                "Number of train steps total",
                self._n_train_steps_total,
            )
            logger.record_tabular(
                "Number of env steps total",
                self._n_env_steps_total,
            )
            logger.record_tabular(
                "Number of rollouts total",
                self._n_rollouts_total,
            )

            times_itrs = gt.get_times().stamps.itrs
            train_time = times_itrs['train'][-1]
            sample_time = times_itrs['sample'][-1]
            eval_time = times_itrs['eval'][-1] if epoch > 0 else 0
            epoch_time = train_time + sample_time + eval_time
            total_time = gt.get_times().total

            logger.record_tabular('Train Time (s)', train_time)
            logger.record_tabular('(Previous) Eval Time (s)', eval_time)
            logger.record_tabular('Sample Time (s)', sample_time)
            logger.record_tabular('Epoch Time (s)', epoch_time)
            logger.record_tabular('Total Train Time (s)', total_time)

            logger.record_tabular("Epoch", epoch)
            logger.dump_tabular(with_prefix=False, with_timestamp=False)
        else:
            logger.log("Skipping eval for now.")

    def _can_evaluate(self):
        """
        One annoying thing about the logger table is that the keys at each
        iteration need to be the exact same. So unless you can compute
        everything, skip evaluation.

        A common example for why you might want to skip evaluation is that at
        the beginning of training, you may not have enough data for a
        validation and training set.

        :return:
        """
        # eval collects its own context, so can eval any time
        return True

    def _can_train(self):
        return all([self.replay_buffer.num_steps_can_sample(idx) >= self.batch_size for idx in self.train_tasks])

    def _get_action_and_info(self, agent, observation):
        """
        Get an action to take in the environment.
        :param observation:
        :return:
        """
        agent.set_num_steps_total(self._n_env_steps_total)
        return agent.get_action(observation,)

    def _start_epoch(self, epoch):
        self._epoch_start_time = time.time()
        self._exploration_paths = []
        self._do_train_time = 0
        logger.push_prefix('Iteration #%d | ' % epoch)

    def _end_epoch(self):
        logger.log("Epoch Duration: {0}".format(
            time.time() - self._epoch_start_time
        ))
        logger.log("Started Training: {0}".format(self._can_train()))
        logger.pop_prefix()

    ##### Snapshotting utils #####
    def get_epoch_snapshot(self, epoch):
        data_to_save = dict(
            epoch=epoch,
            exploration_policy=self.exploration_policy,
        )
        if self.save_environment:
            data_to_save['env'] = self.training_env
        return data_to_save

    def get_extra_data_to_save(self, epoch):
        """
        Save things that shouldn't be saved every snapshot but rather
        overwritten every time.
        :param epoch:
        :return:
        """
        if self.render:
            self.training_env.render(close=True)
        data_to_save = dict(
            epoch=epoch,
        )
        if self.save_environment:
            data_to_save['env'] = self.training_env
        if self.save_replay_buffer:
            data_to_save['replay_buffer'] = self.replay_buffer
        if self.save_algorithm:
            data_to_save['algorithm'] = self
        return data_to_save

    def collect_paths(self, idx, epoch, run, use_mpc):
        # collect paths
        self.task_idx = idx
        self.env.reset_task(idx)

        self.agent.clear_z()
        paths: List[Dict] = []      # 1 dictionary for 1 path
        num_transitions = 0
        num_trajs = 0
        while num_transitions < self.num_steps_per_eval:
            # collect a single episode
            # max_trajs = 1 and episode length is small => just forget about max_samples
            path, num, gp_count, gp_fail = self.sampler.obtain_samples(deterministic=self.eval_deterministic,
                                                                       max_samples=self.num_steps_per_eval - num_transitions,
                                                                       max_trajs=1,
                                                                       accum_context=True,
                                                                       train=True,
                                                                       use_MPC=use_mpc,
                                                                       mpc_solver=self.mpc_forces
                                                                       )
            # path : list which contains a single dictionary
            paths += path
            num_transitions += num
            num_trajs += 1
            if num_trajs >= self.num_exp_traj_eval:
                # Bayesian update of latent variable distribution
                self.agent.infer_posterior(self.agent.context)
        """
        goal = self.env._goal
        for path in paths:
            path['goal'] = goal     # goal
        """

        """
        # save the paths for visualization, only useful for point mass
        if self.dump_eval_paths:
            logger.save_extra_data(paths, path='eval_trajectories/task{}-epoch{}-run{}'.format(idx, epoch, run))
        """
        # paths : [dict for path1, dict for path2]
        p = paths[-1]   # <------- dictionary containing information of the final path

        # some performance measures evaluated on the final path
        col_free = (np.sum(p['collisions_table']) + np.sum(p['collisions_human']) == 0)
        success = (np.sum(p['reaches']) > 0)
        return paths, col_free, success

    def _do_eval(self, indices, epoch):
        final_returns = []
        online_returns = []
        for idx in indices:
            all_rets = []
            for r in range(self.num_evals):
                paths = self.collect_paths(idx, epoch, r, use_mpc=self.use_MPC)
                all_rets.append([eval_util.get_average_returns([p]) for p in paths])
            final_returns.append(np.mean([a[-1] for a in all_rets]))
            # record online returns for the first n trajectories
            n = min([len(a) for a in all_rets])
            all_rets = [a[:n] for a in all_rets]
            all_rets = np.mean(np.stack(all_rets), axis=0) # avg return per nth rollout
            online_returns.append(all_rets)
        n = min([len(t) for t in online_returns])
        online_returns = [t[:n] for t in online_returns]
        return final_returns, online_returns

    def _do_extended_eval(self, indices, epoch, use_mpc):
        final_returns = []
        online_returns = []

        final_cols_human = []
        online_cols_human = []

        final_cols_table = []
        online_cols_table = []

        final_reaches = []
        online_reaches = []

        final_navi_parts = []
        online_navi_parts = []

        final_col_parts = []
        online_col_parts = []

        conditional_table_collision_count = []
        conditional_human_collision_count = []

        final_alerts_human = []
        final_alerts_table = []

        final_mpc_count = []
        final_pearl_count = []

        final_returns_col_free = []
        online_returns_col_free = []

        final_cols_human_col_free = []
        online_cols_human_col_free = []

        final_cols_table_col_free = []
        online_cols_table_col_free = []

        final_reaches_col_free = []
        online_reaches_col_free = []

        final_navi_parts_col_free = []
        online_navi_parts_col_free = []

        final_col_parts_col_free = []
        online_col_parts_col_free = []

        conditional_table_collision_count_col_free = []
        conditional_human_collision_count_col_free = []

        final_alerts_human_col_free = []
        final_alerts_table_col_free = []

        final_mpc_count_col_free = []
        final_pearl_count_col_free = []

        final_returns_col_free_success = []
        online_returns_col_free_success = []

        final_cols_human_col_free_success = []
        online_cols_human_col_free_success = []

        final_cols_table_col_free_success = []
        online_cols_table_col_free_success = []

        final_reaches_col_free_success = []
        online_reaches_col_free_success = []

        final_navi_parts_col_free_success = []
        online_navi_parts_col_free_success = []

        final_col_parts_col_free_success = []
        online_col_parts_col_free_success = []

        conditional_table_collision_count_col_free_success = []
        conditional_human_collision_count_col_free_success = []

        final_alerts_human_col_free_success = []
        final_alerts_table_col_free_success = []

        final_mpc_count_col_free_success = []
        final_pearl_count_col_free_success = []

        is_valid = False        # check whether the robot successfully reaches the goal at least 1 episode.

        free_count_sum = 0
        free_success_count_sum = 0

        for idx in indices:
            free_count = 0
            free_success_count = 0

            all_rets = []
            all_cols_human = []
            all_cols_table = []
            all_reaches = []
            all_navi_parts = []
            all_col_parts = []
            all_alerts_human = []
            all_alerts_table = []
            all_mpc_count = []
            all_pearl_count = []

            all_rets_col_free = []
            all_cols_human_col_free = []
            all_cols_table_col_free = []
            all_reaches_col_free = []
            all_navi_parts_col_free = []
            all_col_parts_col_free = []
            all_alerts_human_col_free = []
            all_alerts_table_col_free = []
            all_mpc_count_col_free = []
            all_pearl_count_col_free = []

            all_rets_col_free_success = []
            all_cols_human_col_free_success = []
            all_cols_table_col_free_success = []
            all_reaches_col_free_success = []
            all_navi_parts_col_free_success = []
            all_col_parts_col_free_success = []
            all_alerts_human_col_free_success = []
            all_alerts_table_col_free_success = []
            all_mpc_count_col_free_success = []
            all_pearl_count_col_free_success = []

            for r in range(self.num_evals):
                # set of adaptation paths
                # if number of evaluations = 2, then we have
                # r = 0 : path^0_1 => ... => path^0_N
                # r = 1 : path^1_1 => ... => path^1_N
                paths, col_free, success = self.collect_paths(idx, epoch, r, use_mpc=use_mpc)

                all_rets.append([eval_util.get_average_returns([p]) for p in paths])
                all_cols_human.append([eval_util.get_average_collisions_human([p]) for p in paths])
                all_cols_table.append([eval_util.get_average_collisions_table([p]) for p in paths])
                all_reaches.append([eval_util.get_average_reaches([p]) for p in paths])
                all_navi_parts.append([eval_util.get_average_navi_parts([p]) for p in paths])
                all_col_parts.append([eval_util.get_average_col_parts([p]) for p in paths])
                all_alerts_human.append([eval_util.get_average_collisions_human([p]) / eval_util.get_average_alerts_human([p]) for p in paths])
                all_alerts_table.append([eval_util.get_average_collisions_table([p]) / eval_util.get_average_alerts_table([p]) for p in paths])
                all_mpc_count.append([eval_util.get_average_mpc_count([p]) for p in paths])
                all_pearl_count.append([eval_util.get_average_pearl_count([p]) for p in paths])

                if col_free:
                    free_count += 1
                    free_count_sum += 1

                    all_rets_col_free.append([eval_util.get_average_returns([p]) for p in paths])
                    all_cols_human_col_free.append([eval_util.get_average_collisions_human([p]) for p in paths])
                    all_cols_table_col_free.append([eval_util.get_average_collisions_table([p]) for p in paths])
                    all_reaches_col_free.append([eval_util.get_average_reaches([p]) for p in paths])
                    all_navi_parts_col_free.append([eval_util.get_average_navi_parts([p]) for p in paths])
                    all_col_parts_col_free.append([eval_util.get_average_col_parts([p]) for p in paths])
                    all_alerts_human_col_free.append(
                        [eval_util.get_average_collisions_human([p]) / eval_util.get_average_alerts_human([p]) for p in
                         paths])
                    all_alerts_table_col_free.append(
                        [eval_util.get_average_collisions_table([p]) / eval_util.get_average_alerts_table([p]) for p in
                         paths])
                    all_mpc_count_col_free.append([eval_util.get_average_mpc_count([p]) for p in paths])
                    all_pearl_count_col_free.append([eval_util.get_average_pearl_count([p]) for p in paths])

                    if success:
                        free_success_count += 1
                        free_success_count_sum += 1

                        all_rets_col_free_success.append([eval_util.get_average_returns([p]) for p in paths])
                        all_cols_human_col_free_success.append([eval_util.get_average_collisions_human([p]) for p in paths])
                        all_cols_table_col_free_success.append([eval_util.get_average_collisions_table([p]) for p in paths])
                        all_reaches_col_free_success.append([eval_util.get_average_reaches([p]) for p in paths])
                        all_navi_parts_col_free_success.append([eval_util.get_average_navi_parts([p]) for p in paths])
                        all_col_parts_col_free_success.append([eval_util.get_average_col_parts([p]) for p in paths])
                        all_alerts_human_col_free_success.append(
                            [eval_util.get_average_collisions_human([p]) / eval_util.get_average_alerts_human([p]) for p
                             in
                             paths])
                        all_alerts_table_col_free_success.append(
                            [eval_util.get_average_collisions_table([p]) / eval_util.get_average_alerts_table([p]) for p
                             in
                             paths])
                        all_mpc_count_col_free_success.append([eval_util.get_average_mpc_count([p]) for p in paths])
                        all_pearl_count_col_free_success.append([eval_util.get_average_pearl_count([p]) for p in paths])

                p = paths[-1]   # final result of adaptation
                if eval_util.get_average_reaches([p]) > 0:
                    # If the robot succeeds to reach the goal, then save the number of collisions.
                    is_valid = True
                    conditional_table_collision_count.append(eval_util.get_average_collisions_table([p]))
                    conditional_human_collision_count.append(eval_util.get_average_collisions_human([p]))
                else:
                    # If robot fails to reach the goal, then save a dummy value.
                    conditional_table_collision_count.append(np.nan)
                    conditional_human_collision_count.append(np.nan)

            final_returns.append(np.mean([a[-1] for a in all_rets]))
            final_cols_human.append(np.mean([a[-1] for a in all_cols_human]))
            final_cols_table.append(np.mean([a[-1] for a in all_cols_table]))
            final_reaches.append(np.mean([a[-1] for a in all_reaches]))
            final_navi_parts.append(np.mean([a[-1] for a in all_navi_parts]))
            final_col_parts.append(np.mean([a[-1] for a in all_col_parts]))
            final_alerts_human.append(np.mean([a[-1] for a in all_alerts_human]))
            final_alerts_table.append(np.mean([a[-1] for a in all_alerts_table]))
            final_mpc_count.append(np.mean([a[-1] for a in all_mpc_count]))
            final_pearl_count.append(np.mean([a[-1] for a in all_pearl_count]))

            if free_count:
                final_returns_col_free.append(np.mean([a[-1] for a in all_rets_col_free]))
                final_cols_human_col_free.append(np.mean([a[-1] for a in all_cols_human_col_free]))
                final_cols_table_col_free.append(np.mean([a[-1] for a in all_cols_table_col_free]))
                final_reaches_col_free.append(np.mean([a[-1] for a in all_reaches_col_free]))
                final_navi_parts_col_free.append(np.mean([a[-1] for a in all_navi_parts_col_free]))
                final_col_parts_col_free.append(np.mean([a[-1] for a in all_col_parts_col_free]))
                final_alerts_human_col_free.append(np.mean([a[-1] for a in all_alerts_human_col_free]))
                final_alerts_table_col_free.append(np.mean([a[-1] for a in all_alerts_table_col_free]))
                final_mpc_count_col_free.append(np.mean([a[-1] for a in all_mpc_count_col_free]))
                final_pearl_count_col_free.append(np.mean([a[-1] for a in all_pearl_count_col_free]))

            if free_success_count:
                final_returns_col_free_success.append(np.mean([a[-1] for a in all_rets_col_free_success]))
                final_cols_human_col_free_success.append(np.mean([a[-1] for a in all_cols_human_col_free_success]))
                final_cols_table_col_free_success.append(np.mean([a[-1] for a in all_cols_table_col_free_success]))
                final_reaches_col_free_success.append(np.mean([a[-1] for a in all_reaches_col_free_success]))
                final_navi_parts_col_free_success.append(np.mean([a[-1] for a in all_navi_parts_col_free_success]))
                final_col_parts_col_free_success.append(np.mean([a[-1] for a in all_col_parts_col_free_success]))
                final_alerts_human_col_free_success.append(np.mean([a[-1] for a in all_alerts_human_col_free_success]))
                final_alerts_table_col_free_success.append(np.mean([a[-1] for a in all_alerts_table_col_free_success]))
                final_mpc_count_col_free_success.append(np.mean([a[-1] for a in all_mpc_count_col_free_success]))
                final_pearl_count_col_free_success.append(np.mean([a[-1] for a in all_pearl_count_col_free_success]))

            # record online returns for the first n trajectories
            n = min([len(a) for a in all_rets])
            all_rets = [a[:n] for a in all_rets]
            all_rets = np.mean(np.stack(all_rets), axis=0)  # avg return per nth rollout
            online_returns.append(all_rets)

        assert len(conditional_table_collision_count) == len(indices) * self.num_evals
        if is_valid:
            # At least one of the entries is not NaN.
            mean_conditional_table_collision_count = np.nanmean(conditional_table_collision_count)
            mean_conditional_human_collision_count = np.nanmean(conditional_human_collision_count)
            std_conditional_table_collision_count = np.nanstd(conditional_table_collision_count)
            std_conditional_human_collision_count = np.nanstd(conditional_human_collision_count)
        else:
            mean_conditional_table_collision_count = np.nan
            mean_conditional_human_collision_count = np.nan
            std_conditional_table_collision_count = np.nan
            std_conditional_human_collision_count = np.nan

        col_free_ratio = free_count_sum / (len(indices) * self.num_evals)
        col_free_success_ratio = free_success_count_sum / (len(indices) * self.num_evals)

        n = min([len(t) for t in online_returns])
        online_returns = [t[:n] for t in online_returns]
        return (final_returns,
                online_returns,
                final_cols_human,
                final_cols_table,
                final_reaches,
                final_navi_parts,
                final_col_parts,
                mean_conditional_table_collision_count,
                mean_conditional_human_collision_count,
                std_conditional_table_collision_count,
                std_conditional_human_collision_count,
                final_alerts_human,
                final_alerts_table,
                final_mpc_count,
                final_pearl_count,
                col_free_ratio,
                final_returns_col_free,
                final_cols_human_col_free,
                final_cols_table_col_free,
                final_reaches_col_free,
                final_navi_parts_col_free,
                final_col_parts_col_free,
                final_alerts_human_col_free,
                final_alerts_table_col_free,
                final_mpc_count_col_free,
                final_pearl_count_col_free,
                col_free_success_ratio,
                final_returns_col_free_success,
                final_cols_human_col_free_success,
                final_cols_table_col_free_success,
                final_reaches_col_free_success,
                final_navi_parts_col_free_success,
                final_col_parts_col_free_success,
                final_alerts_human_col_free_success,
                final_alerts_table_col_free_success,
                final_mpc_count_col_free_success,
                final_pearl_count_col_free_success
                )

    def evaluate(self, epoch):
        # meta-testing
        if self.eval_statistics is None:
            self.eval_statistics = OrderedDict()

        # sample trajectories from prior for debugging / visualization
        """
        if self.dump_eval_paths:
            # 100 arbitrarily chosen for visualizations of point_robot trajectories
            # just want stochasticity of z, not the policy
            # not used for PEARL + GP-MPC
            self.agent.clear_z()
            prior_paths, _, gp_count, gp_fail = self.sampler.obtain_samples(deterministic=self.eval_deterministic,
                                                                            max_samples=self.max_path_length * 20,
                                                                            accum_context=False,
                                                                            train=True,
                                                                            resample=1,
                                                                            use_MPC=self.use_MPC,
                                                                            mpc_solver=self.mpc_forces
                                                                            )
            logger.save_extra_data(prior_paths, path='eval_trajectories/prior-epoch{}'.format(epoch))
        """
        ### train tasks
        # eval on a subset of train tasks for speed
        """
        indices = np.random.choice(self.train_tasks, len(self.eval_tasks))
        eval_util.dprint('evaluating on {} train tasks'.format(len(indices)))
        ### eval train tasks with posterior sampled from the training replay buffer
        train_returns = []
        gp_count, gp_fail = 0, 0
        for idx in indices:
            self.task_idx = idx
            self.env.reset_task(idx)
            paths = []
            for _ in range(self.num_steps_per_eval // self.max_path_length):
                context = self.sample_context(idx)
                self.agent.infer_posterior(context)
                p, _, _gp_count, _gp_fail = self.sampler.obtain_samples(deterministic=self.eval_deterministic,
                                                                        max_samples=self.max_path_length,
                                                                        accum_context=False,
                                                                        train=True,
                                                                        max_trajs=1,
                                                                        resample=np.inf,
                                                                        use_MPC=self.use_MPC,
                                                                        mpc_solver=self.mpc_forces
                                                                        )
                paths += p
                gp_count += _gp_count
                gp_fail += _gp_fail
                
            if self.sparse_rewards:
                for p in paths:
                    sparse_rewards = np.stack(e['sparse_reward'] for e in p['env_infos']).reshape(-1, 1)
                    p['rewards'] = sparse_rewards

            train_returns.append(eval_util.get_average_returns(paths))
        train_returns = np.mean(train_returns)
        ### eval train tasks with on-policy data to match eval of test tasks
        # train_final_returns, train_online_returns = self._do_eval(indices, epoch)
        res = self._do_extended_eval(indices, epoch)


        train_final_returns = res[0]
        train_online_returns = res[1]
        train_final_cols_human = res[2]
        train_final_cols_table = res[3]
        train_final_reaches = res[4]
        train_final_navi_parts = res[5]
        train_final_col_parts = res[6]
        train_mean_conditional_table_collision_count = res[7]
        train_mean_conditional_human_collision_count = res[8]
        train_std_conditional_table_collision_count = res[9]
        train_std_conditional_human_collision_count = res[10]
        train_final_alerts_human = res[11]
        train_final_alerts_table = res[12]
        train_final_mpc_count = res[13]
        train_final_pearl_count = res[14]

        eval_util.dprint('train online returns')
        eval_util.dprint(train_online_returns)
        # train_final_collisions, train_online_collisions = self._eval_collisions(indices, epoch)
        # train_final_reaches, train_online_reaches = self._eval_reaches(indices, epoch)
        # train_final_navi_parts, train_online_navi_parts = self._eval_navi_parts(indices, epoch)
        # train_final_col_parts, train_online_col_parts = self._eval_col_parts(indices, epoch)
        """
        ### test tasks
        eval_util.dprint('evaluating on {} test tasks'.format(len(self.eval_tasks)))
        # test_final_returns, test_online_returns = self._do_eval(self.eval_tasks, epoch)

        eval_res = self._do_extended_eval(self.eval_tasks, epoch, use_mpc=self.use_MPC)
        test_final_returns = eval_res[0]
        test_online_returns = eval_res[1]
        test_final_cols_human = eval_res[2]
        test_final_cols_table = eval_res[3]
        test_final_reaches = eval_res[4]
        test_final_navi_parts = eval_res[5]
        test_final_col_parts = eval_res[6]
        test_mean_conditional_table_collision_count = eval_res[7]
        test_mean_conditional_human_collision_count = eval_res[8]
        test_std_conditional_table_collision_count = eval_res[9]
        test_std_conditional_human_collision_count = eval_res[10]
        test_final_alerts_human = eval_res[11]
        test_final_alerts_table = eval_res[12]
        test_final_mpc_count = eval_res[13]
        test_final_pearl_count = eval_res[14]
        test_col_free_ratio = eval_res[15]
        test_final_returns_col_free = eval_res[16]
        test_final_cols_human_col_free = eval_res[17]
        test_final_cols_table_col_free = eval_res[18]
        test_final_reaches_col_free = eval_res[19]
        test_final_navi_parts_col_free = eval_res[20]
        test_final_col_parts_col_free = eval_res[21]
        test_final_alerts_human_col_free = eval_res[22]
        test_final_alerts_table_col_free = eval_res[23]
        test_final_mpc_count_col_free = eval_res[24]
        test_final_pearl_count_col_free = eval_res[25]
        test_col_free_success_ratio = eval_res[26]
        test_final_returns_col_free_success = eval_res[27]
        test_final_cols_human_col_free_success = eval_res[28]
        test_final_cols_table_col_free_success = eval_res[29]
        test_final_reaches_col_free_success = eval_res[30]
        test_final_navi_parts_col_free_success = eval_res[31]
        test_final_col_parts_col_free_success = eval_res[32]
        test_final_alerts_human_col_free_success = eval_res[33]
        test_final_alerts_table_col_free_success = eval_res[34]
        test_final_mpc_count_col_free_success = eval_res[35]
        test_final_pearl_count_col_free_success = eval_res[36]

        eval_util.dprint('test online returns')
        eval_util.dprint(test_online_returns)
        # test_final_collisions, test_online_collisions = self._eval_collisions(self.eval_tasks, epoch)
        # test_final_reaches, test_online_reaches = self._eval_reaches(self.eval_tasks, epoch)
        # test_final_navi_parts, test_online_navi_parts = self._eval_navi_parts(self.eval_tasks, epoch)
        # test_final_col_parts, test_online_col_parts = self._eval_col_parts(self.eval_tasks, epoch)

        # save the final posterior
        self.agent.log_diagnostics(self.eval_statistics)

        # if hasattr(self.env, "log_diagnostics"):
        #     self.env.log_diagnostics(paths, prefix=None)

        # avg_train_return = np.mean(train_final_returns)
        # std_train_return = np.std(train_final_returns)
        avg_test_return = np.mean(test_final_returns)
        std_test_return = np.std(test_final_returns)
        # avg_train_online_return = np.mean(np.stack(train_online_returns), axis=0)
        avg_test_online_return = np.mean(np.stack(test_online_returns), axis=0)
        avg_test_return_col_free = np.mean(test_final_returns_col_free)
        std_test_return_col_free = np.std(test_final_returns_col_free)

        avg_test_return_col_free_success = np.mean(test_final_returns_col_free_success)
        std_test_return_col_free_success = np.std(test_final_returns_col_free_success)


        # self.eval_statistics['AverageTrainReturn_all_train_tasks'] = train_returns
        # self.eval_statistics['AverageReturn_all_train_tasks'] = avg_train_return
        self.eval_statistics['AverageReturn_all_test_tasks'] = avg_test_return
        # self.eval_statistics['StdReturn_all_train_tasks'] = std_train_return
        self.eval_statistics['StdReturn_all_test_tasks'] = std_test_return

        # Collision and goal reach count
        # self.eval_statistics['AverageCollisionHuman_all_train_tasks'] = np.mean(train_final_cols_human)
        self.eval_statistics['AverageCollisionHuman_all_test_tasks'] = np.mean(test_final_cols_human)
        # self.eval_statistics['StdCollisionHuman_all_train_tasks'] = np.std(train_final_cols_human)
        self.eval_statistics['StdCollisionHuman_all_test_tasks'] = np.std(test_final_cols_human)

        # self.eval_statistics['AverageCollisionTable_all_train_tasks'] = np.mean(train_final_cols_table)
        self.eval_statistics['AverageCollisionTable_all_test_tasks'] = np.mean(test_final_cols_table)
        # self.eval_statistics['StdCollisionTable_all_train_tasks'] = np.std(train_final_cols_table)
        self.eval_statistics['StdCollisionTable_all_test_tasks'] = np.std(test_final_cols_table)

        # self.eval_statistics['AverageConditionalCollisionTable_all_train_tasks'] = train_mean_conditional_table_collision_count
        self.eval_statistics['AverageConditionalCollisionTable_all_test_tasks'] = test_mean_conditional_table_collision_count
        # self.eval_statistics['StdConditionalCollisionTable_all_train_tasks'] = train_std_conditional_table_collision_count
        self.eval_statistics['StdConditionalCollisionTable_all_test_tasks'] = test_std_conditional_table_collision_count

        # self.eval_statistics['AverageConditionalCollisionHuman_all_train_tasks'] = train_mean_conditional_human_collision_count
        self.eval_statistics['AverageConditionalCollisionHuman_all_test_tasks'] = test_mean_conditional_human_collision_count
        # self.eval_statistics['StdConditionalCollisionHuman_all_train_tasks'] = train_std_conditional_human_collision_count
        self.eval_statistics['StdConditionalCollisionHuman_all_test_tasks'] = test_std_conditional_human_collision_count

        # self.eval_statistics['AverageReaches_all_train_tasks'] = np.mean(train_final_reaches)
        self.eval_statistics['AverageReaches_all_test_tasks'] = np.mean(test_final_reaches)
        # self.eval_statistics['StdReaches_all_train_tasks'] = np.std(train_final_reaches)
        self.eval_statistics['StdReaches_all_test_tasks'] = np.std(test_final_reaches)

        # self.eval_statistics['AverageNaviParts_all_train_tasks'] = np.mean(train_final_navi_parts)
        self.eval_statistics['AverageNaviParts_all_test_tasks'] = np.mean(test_final_navi_parts)
        # self.eval_statistics['StdNaviParts_all_train_tasks'] = np.std(train_final_navi_parts)
        self.eval_statistics['StdNaviParts_all_test_tasks'] = np.std(test_final_navi_parts)

        # self.eval_statistics['AverageColParts_all_train_tasks'] = np.mean(train_final_col_parts)
        self.eval_statistics['AverageColParts_all_test_tasks'] = np.mean(test_final_col_parts)
        # self.eval_statistics['StdColParts_all_train_tasks'] = np.std(train_final_col_parts)
        self.eval_statistics['StdColParts_all_test_tasks'] = np.std(test_final_col_parts)

        # self.eval_statistics['AverageCollisionHumanProbability_all_train_tasks'] = np.mean(train_final_alerts_human)
        self.eval_statistics['AverageCollisionHumanProbability_all_test_tasks'] = np.mean(test_final_alerts_human)
        # self.eval_statistics['StdCollisionHumanProbability_all_train_tasks'] = np.std(train_final_alerts_human)
        self.eval_statistics['StdCollisionHumanProbability_all_test_tasks'] = np.std(test_final_alerts_human)

        # self.eval_statistics['AverageCollisionTableProbability_all_train_tasks'] = np.mean(train_final_alerts_table)
        self.eval_statistics['AverageCollisionTableProbability_all_test_tasks'] = np.mean(test_final_alerts_table)
        # self.eval_statistics['StdCollisionTableProbability_all_train_tasks'] = np.std(train_final_alerts_table)
        self.eval_statistics['StdCollisionTableProbability_all_test_tasks'] = np.std(test_final_alerts_table)

        # self.eval_statistics['AverageMPCCount_all_train_tasks'] = np.mean(train_final_mpc_count)
        self.eval_statistics['AverageMPCCount_all_test_tasks'] = np.mean(test_final_mpc_count)
        # self.eval_statistics['StdMPCCount_all_train_tasks'] = np.std(train_final_mpc_count)
        self.eval_statistics['StdMPCCount_all_test_tasks'] = np.std(test_final_mpc_count)

        # self.eval_statistics['AveragePEARLCount_all_train_tasks'] = np.mean(train_final_pearl_count)
        self.eval_statistics['AveragePEARLCount_all_test_tasks'] = np.mean(test_final_pearl_count)
        # self.eval_statistics['StdPEARLCount_all_train_tasks'] = np.std(train_final_pearl_count)
        self.eval_statistics['StdPEARLCount_all_test_tasks'] = np.std(test_final_pearl_count)

        # collision-free statistics
        self.eval_statistics['Collision_free_ratio'] = test_col_free_ratio
        self.eval_statistics['AverageReturn_all_test_tasks_col_free'] = avg_test_return_col_free
        # self.eval_statistics['StdReturn_all_train_tasks'] = std_train_return
        self.eval_statistics['StdReturn_all_test_tasks_col_free'] = std_test_return_col_free

        # Collision and goal reach count
        # self.eval_statistics['AverageCollisionHuman_all_train_tasks'] = np.mean(train_final_cols_human)
        self.eval_statistics['AverageCollisionHuman_all_test_tasks_col_free'] = np.mean(test_final_cols_human_col_free)
        # self.eval_statistics['StdCollisionHuman_all_train_tasks'] = np.std(train_final_cols_human)
        self.eval_statistics['StdCollisionHuman_all_test_tasks_col_free'] = np.std(test_final_cols_human_col_free)

        # self.eval_statistics['AverageCollisionTable_all_train_tasks'] = np.mean(train_final_cols_table)
        self.eval_statistics['AverageCollisionTable_all_test_tasks_col_free'] = np.mean(test_final_cols_table_col_free)
        # self.eval_statistics['StdCollisionTable_all_train_tasks'] = np.std(train_final_cols_table)
        self.eval_statistics['StdCollisionTable_all_test_tasks_col_free'] = np.std(test_final_cols_table_col_free)

        # self.eval_statistics['AverageReaches_all_train_tasks'] = np.mean(train_final_reaches)
        self.eval_statistics['AverageReaches_all_test_tasks_col_free'] = np.mean(test_final_reaches_col_free)
        # self.eval_statistics['StdReaches_all_train_tasks'] = np.std(train_final_reaches)
        self.eval_statistics['StdReaches_all_test_tasks_col_free'] = np.std(test_final_reaches_col_free)

        # self.eval_statistics['AverageNaviParts_all_train_tasks'] = np.mean(train_final_navi_parts)
        self.eval_statistics['AverageNaviParts_all_test_tasks_col_free'] = np.mean(test_final_navi_parts_col_free)
        # self.eval_statistics['StdNaviParts_all_train_tasks'] = np.std(train_final_navi_parts)
        self.eval_statistics['StdNaviParts_all_test_tasks_col_free'] = np.std(test_final_navi_parts_col_free)

        # self.eval_statistics['AverageColParts_all_train_tasks'] = np.mean(train_final_col_parts)
        self.eval_statistics['AverageColParts_all_test_tasks_col_free'] = np.mean(test_final_col_parts_col_free)
        # self.eval_statistics['StdColParts_all_train_tasks'] = np.std(train_final_col_parts)
        self.eval_statistics['StdColParts_all_test_tasks_col_free'] = np.std(test_final_col_parts_col_free)

        # self.eval_statistics['AverageCollisionHumanProbability_all_train_tasks'] = np.mean(train_final_alerts_human)
        self.eval_statistics['AverageCollisionHumanProbability_all_test_tasks_col_free'] = np.mean(test_final_alerts_human_col_free)
        # self.eval_statistics['StdCollisionHumanProbability_all_train_tasks'] = np.std(train_final_alerts_human)
        self.eval_statistics['StdCollisionHumanProbability_all_test_tasks_col_free'] = np.std(test_final_alerts_human_col_free)

        # self.eval_statistics['AverageCollisionTableProbability_all_train_tasks'] = np.mean(train_final_alerts_table)
        self.eval_statistics['AverageCollisionTableProbability_all_test_tasks_col_free'] = np.mean(test_final_alerts_table_col_free)
        # self.eval_statistics['StdCollisionTableProbability_all_train_tasks'] = np.std(train_final_alerts_table)
        self.eval_statistics['StdCollisionTableProbability_all_test_tasks_col_free'] = np.std(test_final_alerts_table_col_free)

        # self.eval_statistics['AverageMPCCount_all_train_tasks'] = np.mean(train_final_mpc_count)
        self.eval_statistics['AverageMPCCount_all_test_tasks_col_free'] = np.mean(test_final_mpc_count_col_free)
        # self.eval_statistics['StdMPCCount_all_train_tasks'] = np.std(train_final_mpc_count)
        self.eval_statistics['StdMPCCount_all_test_tasks_col_free'] = np.std(test_final_mpc_count_col_free)

        # self.eval_statistics['AveragePEARLCount_all_train_tasks'] = np.mean(train_final_pearl_count)
        self.eval_statistics['AveragePEARLCount_all_test_tasks_col_free'] = np.mean(test_final_pearl_count_col_free)
        # self.eval_statistics['StdPEARLCount_all_train_tasks'] = np.std(train_final_pearl_count)
        self.eval_statistics['StdPEARLCount_all_test_tasks_col_free'] = np.std(test_final_pearl_count_col_free)


        self.eval_statistics['Collision_free_success_ratio'] = test_col_free_success_ratio
        self.eval_statistics['AverageReturn_all_test_tasks_col_free_success'] = avg_test_return_col_free_success
        # self.eval_statistics['StdReturn_all_train_tasks'] = std_train_return
        self.eval_statistics['StdReturn_all_test_tasks_col_free_success'] = std_test_return_col_free_success

        # Collision and goal reach count
        # self.eval_statistics['AverageCollisionHuman_all_train_tasks'] = np.mean(train_final_cols_human)
        self.eval_statistics['AverageCollisionHuman_all_test_tasks_col_free_success'] = np.mean(test_final_cols_human_col_free_success)
        # self.eval_statistics['StdCollisionHuman_all_train_tasks'] = np.std(train_final_cols_human)
        self.eval_statistics['StdCollisionHuman_all_test_tasks_col_free_success'] = np.std(test_final_cols_human_col_free_success)

        # self.eval_statistics['AverageCollisionTable_all_train_tasks'] = np.mean(train_final_cols_table)
        self.eval_statistics['AverageCollisionTable_all_test_tasks_col_free_success'] = np.mean(test_final_cols_table_col_free_success)
        # self.eval_statistics['StdCollisionTable_all_train_tasks'] = np.std(train_final_cols_table)
        self.eval_statistics['StdCollisionTable_all_test_tasks_col_free_success'] = np.std(test_final_cols_table_col_free_success)

        # self.eval_statistics['AverageReaches_all_train_tasks'] = np.mean(train_final_reaches)
        self.eval_statistics['AverageReaches_all_test_tasks_col_free_success'] = np.mean(test_final_reaches_col_free_success)
        # self.eval_statistics['StdReaches_all_train_tasks'] = np.std(train_final_reaches)
        self.eval_statistics['StdReaches_all_test_tasks_col_free_success'] = np.std(test_final_reaches_col_free_success)

        # self.eval_statistics['AverageNaviParts_all_train_tasks'] = np.mean(train_final_navi_parts)
        self.eval_statistics['AverageNaviParts_all_test_tasks_col_free_success'] = np.mean(test_final_navi_parts_col_free_success)
        # self.eval_statistics['StdNaviParts_all_train_tasks'] = np.std(train_final_navi_parts)
        self.eval_statistics['StdNaviParts_all_test_tasks_col_free_success'] = np.std(test_final_navi_parts_col_free_success)

        # self.eval_statistics['AverageColParts_all_train_tasks'] = np.mean(train_final_col_parts)
        self.eval_statistics['AverageColParts_all_test_tasks_col_free_success'] = np.mean(test_final_col_parts_col_free_success)
        # self.eval_statistics['StdColParts_all_train_tasks'] = np.std(train_final_col_parts)
        self.eval_statistics['StdColParts_all_test_tasks_col_free_success'] = np.std(test_final_col_parts_col_free_success)

        # self.eval_statistics['AverageCollisionHumanProbability_all_train_tasks'] = np.mean(train_final_alerts_human)
        self.eval_statistics['AverageCollisionHumanProbability_all_test_tasks_col_free_success'] = np.mean(
            test_final_alerts_human_col_free_success)
        # self.eval_statistics['StdCollisionHumanProbability_all_train_tasks'] = np.std(train_final_alerts_human)
        self.eval_statistics['StdCollisionHumanProbability_all_test_tasks_col_free_success'] = np.std(
            test_final_alerts_human_col_free_success)

        # self.eval_statistics['AverageCollisionTableProbability_all_train_tasks'] = np.mean(train_final_alerts_table)
        self.eval_statistics['AverageCollisionTableProbability_all_test_tasks_col_free_success'] = np.mean(
            test_final_alerts_table_col_free_success)
        # self.eval_statistics['StdCollisionTableProbability_all_train_tasks'] = np.std(train_final_alerts_table)
        self.eval_statistics['StdCollisionTableProbability_all_test_tasks_col_free_success'] = np.std(
            test_final_alerts_table_col_free_success)

        # self.eval_statistics['AverageMPCCount_all_train_tasks'] = np.mean(train_final_mpc_count)
        self.eval_statistics['AverageMPCCount_all_test_tasks_col_free_success'] = np.mean(test_final_mpc_count_col_free_success)
        # self.eval_statistics['StdMPCCount_all_train_tasks'] = np.std(train_final_mpc_count)
        self.eval_statistics['StdMPCCount_all_test_tasks_col_free_success'] = np.std(test_final_mpc_count_col_free_success)

        # self.eval_statistics['AveragePEARLCount_all_train_tasks'] = np.mean(train_final_pearl_count)
        self.eval_statistics['AveragePEARLCount_all_test_tasks_col_free_success'] = np.mean(test_final_pearl_count_col_free_success)
        # self.eval_statistics['StdPEARLCount_all_train_tasks'] = np.std(train_final_pearl_count)
        self.eval_statistics['StdPEARLCount_all_test_tasks_col_free_success'] = np.std(test_final_pearl_count_col_free_success)

        if self.use_MPC:
            # If MPC is infused, we report evaluation result with MPC turned off
            eval_res_wo_mpc = self._do_extended_eval(self.eval_tasks, epoch, use_mpc=False)
            test_final_returns_wo_mpc = eval_res_wo_mpc[0]
            test_online_returns_wo_mpc = eval_res_wo_mpc[1]
            test_final_cols_human_wo_mpc = eval_res_wo_mpc[2]
            test_final_cols_table_wo_mpc = eval_res_wo_mpc[3]
            test_final_reaches_wo_mpc = eval_res_wo_mpc[4]
            test_final_navi_parts_wo_mpc = eval_res_wo_mpc[5]
            test_final_col_parts_wo_mpc = eval_res_wo_mpc[6]
            test_mean_conditional_table_collision_count_wo_mpc = eval_res_wo_mpc[7]
            test_mean_conditional_human_collision_count_wo_mpc = eval_res_wo_mpc[8]
            test_std_conditional_table_collision_count_wo_mpc = eval_res_wo_mpc[9]
            test_std_conditional_human_collision_count_wo_mpc = eval_res_wo_mpc[10]
            test_final_alerts_human_wo_mpc = eval_res_wo_mpc[11]
            test_final_alerts_table_wo_mpc = eval_res_wo_mpc[12]
            test_final_mpc_count_wo_mpc = eval_res_wo_mpc[13]
            test_final_pearl_count_wo_mpc = eval_res_wo_mpc[14]
            test_col_free_ratio_wo_mpc = eval_res_wo_mpc[15]
            test_final_returns_col_free_wo_mpc = eval_res_wo_mpc[16]
            test_final_cols_human_col_free_wo_mpc = eval_res_wo_mpc[17]
            test_final_cols_table_col_free_wo_mpc = eval_res_wo_mpc[18]
            test_final_reaches_col_free_wo_mpc = eval_res_wo_mpc[19]
            test_final_navi_parts_col_free_wo_mpc = eval_res_wo_mpc[20]
            test_final_col_parts_col_free_wo_mpc = eval_res_wo_mpc[21]
            test_final_alerts_human_col_free_wo_mpc = eval_res_wo_mpc[22]
            test_final_alerts_table_col_free_wo_mpc = eval_res_wo_mpc[23]
            test_final_mpc_count_col_free_wo_mpc = eval_res_wo_mpc[24]
            test_final_pearl_count_col_free_wo_mpc = eval_res_wo_mpc[25]
            test_col_free_success_ratio_wo_mpc = eval_res_wo_mpc[26]
            test_final_returns_col_free_success_wo_mpc = eval_res_wo_mpc[27]
            test_final_cols_human_col_free_success_wo_mpc = eval_res_wo_mpc[28]
            test_final_cols_table_col_free_success_wo_mpc = eval_res_wo_mpc[29]
            test_final_reaches_col_free_success_wo_mpc = eval_res_wo_mpc[30]
            test_final_navi_parts_col_free_success_wo_mpc = eval_res_wo_mpc[31]
            test_final_col_parts_col_free_success_wo_mpc = eval_res_wo_mpc[32]
            test_final_alerts_human_col_free_success_wo_mpc = eval_res_wo_mpc[33]
            test_final_alerts_table_col_free_success_wo_mpc = eval_res_wo_mpc[34]
            test_final_mpc_count_col_free_success_wo_mpc = eval_res_wo_mpc[35]
            test_final_pearl_count_col_free_success_wo_mpc = eval_res_wo_mpc[36]

            # eval_util.dprint('test online returns')
            # eval_util.dprint(test_online_returns)
            # test_final_collisions, test_online_collisions = self._eval_collisions(self.eval_tasks, epoch)
            # test_final_reaches, test_online_reaches = self._eval_reaches(self.eval_tasks, epoch)
            # test_final_navi_parts, test_online_navi_parts = self._eval_navi_parts(self.eval_tasks, epoch)
            # test_final_col_parts, test_online_col_parts = self._eval_col_parts(self.eval_tasks, epoch)

            # save the final posterior
            self.agent.log_diagnostics(self.eval_statistics)

            # if hasattr(self.env, "log_diagnostics"):
            #     self.env.log_diagnostics(paths, prefix=None)
            avg_test_return_wo_mpc = np.mean(test_final_returns_wo_mpc)
            std_test_return_wo_mpc = np.std(test_final_returns_wo_mpc)
            # self.eval_statistics['AverageTrainReturn_all_train_tasks'] = train_returns
            # self.eval_statistics['AverageReturn_all_train_tasks'] = avg_train_return
            self.eval_statistics['AverageReturn_all_test_tasks_wo_mpc'] = avg_test_return_wo_mpc
            # self.eval_statistics['StdReturn_all_train_tasks'] = std_train_return
            self.eval_statistics['StdReturn_all_test_tasks_wo_mpc'] = std_test_return_wo_mpc

            # Collision and goal reach count
            # self.eval_statistics['AverageCollisionHuman_all_train_tasks'] = np.mean(train_final_cols_human)
            self.eval_statistics['AverageCollisionHuman_all_test_tasks_wo_mpc'] = np.mean(test_final_cols_human_wo_mpc)
            # self.eval_statistics['StdCollisionHuman_all_train_tasks'] = np.std(train_final_cols_human)
            self.eval_statistics['StdCollisionHuman_all_test_tasks_wo_mpc'] = np.std(test_final_cols_human_wo_mpc)

            # self.eval_statistics['AverageCollisionTable_all_train_tasks'] = np.mean(train_final_cols_table)
            self.eval_statistics['AverageCollisionTable_all_test_tasks_wo_mpc'] = np.mean(test_final_cols_table_wo_mpc)
            # self.eval_statistics['StdCollisionTable_all_train_tasks'] = np.std(train_final_cols_table)
            self.eval_statistics['StdCollisionTable_all_test_tasks_wo_mpc'] = np.std(test_final_cols_table_wo_mpc)

            # self.eval_statistics['AverageConditionalCollisionTable_all_train_tasks'] = train_mean_conditional_table_collision_count
            self.eval_statistics[
                'AverageConditionalCollisionTable_all_test_tasks_wo_mpc'] = test_mean_conditional_table_collision_count_wo_mpc
            # self.eval_statistics['StdConditionalCollisionTable_all_train_tasks'] = train_std_conditional_table_collision_count
            self.eval_statistics[
                'StdConditionalCollisionTable_all_test_tasks_wo_mpc'] = test_std_conditional_table_collision_count_wo_mpc

            # self.eval_statistics['AverageConditionalCollisionHuman_all_train_tasks'] = train_mean_conditional_human_collision_count
            self.eval_statistics[
                'AverageConditionalCollisionHuman_all_test_tasks_wo_mpc'] = test_mean_conditional_human_collision_count_wo_mpc
            # self.eval_statistics['StdConditionalCollisionHuman_all_train_tasks'] = train_std_conditional_human_collision_count
            self.eval_statistics[
                'StdConditionalCollisionHuman_all_test_tasks_wo_mpc'] = test_std_conditional_human_collision_count_wo_mpc

            # self.eval_statistics['AverageReaches_all_train_tasks'] = np.mean(train_final_reaches)
            self.eval_statistics['AverageReaches_all_test_tasks_wo_mpc'] = np.mean(test_final_reaches_wo_mpc)
            # self.eval_statistics['StdReaches_all_train_tasks'] = np.std(train_final_reaches)
            self.eval_statistics['StdReaches_all_test_tasks_wo_mpc'] = np.std(test_final_reaches_wo_mpc)

            # self.eval_statistics['AverageNaviParts_all_train_tasks'] = np.mean(train_final_navi_parts)
            self.eval_statistics['AverageNaviParts_all_test_tasks_wo_mpc'] = np.mean(test_final_navi_parts_wo_mpc)
            # self.eval_statistics['StdNaviParts_all_train_tasks'] = np.std(train_final_navi_parts)
            self.eval_statistics['StdNaviParts_all_test_tasks_wo_mpc'] = np.std(test_final_navi_parts_wo_mpc)

            # self.eval_statistics['AverageColParts_all_train_tasks'] = np.mean(train_final_col_parts)
            self.eval_statistics['AverageColParts_all_test_tasks_wo_mpc'] = np.mean(test_final_col_parts_wo_mpc)
            # self.eval_statistics['StdColParts_all_train_tasks'] = np.std(train_final_col_parts)
            self.eval_statistics['StdColParts_all_test_tasks_wo_mpc'] = np.std(test_final_col_parts_wo_mpc)

            # self.eval_statistics['AverageCollisionHumanProbability_all_train_tasks'] = np.mean(train_final_alerts_human)
            self.eval_statistics['AverageCollisionHumanProbability_all_test_tasks_wo_mpc'] = np.mean(test_final_alerts_human_wo_mpc)
            # self.eval_statistics['StdCollisionHumanProbability_all_train_tasks'] = np.std(train_final_alerts_human)
            self.eval_statistics['StdCollisionHumanProbability_all_test_tasks_wo_mpc'] = np.std(test_final_alerts_human_wo_mpc)

            # self.eval_statistics['AverageCollisionTableProbability_all_train_tasks'] = np.mean(train_final_alerts_table)
            self.eval_statistics['AverageCollisionTableProbability_all_test_tasks_wo_mpc'] = np.mean(test_final_alerts_table_wo_mpc)
            # self.eval_statistics['StdCollisionTableProbability_all_train_tasks'] = np.std(train_final_alerts_table)
            self.eval_statistics['StdCollisionTableProbability_all_test_tasks_wo_mpc'] = np.std(test_final_alerts_table_wo_mpc)

            # self.eval_statistics['AverageMPCCount_all_train_tasks'] = np.mean(train_final_mpc_count)
            self.eval_statistics['AverageMPCCount_all_test_tasks_wo_mpc'] = np.mean(test_final_mpc_count_wo_mpc)
            # self.eval_statistics['StdMPCCount_all_train_tasks'] = np.std(train_final_mpc_count)
            self.eval_statistics['StdMPCCount_all_test_tasks_wo_mpc'] = np.std(test_final_mpc_count_wo_mpc)

            # self.eval_statistics['AveragePEARLCount_all_train_tasks'] = np.mean(train_final_pearl_count)
            self.eval_statistics['AveragePEARLCount_all_test_tasks_wo_mpc'] = np.mean(test_final_pearl_count_wo_mpc)
            # self.eval_statistics['StdPEARLCount_all_train_tasks'] = np.std(train_final_pearl_count)
            self.eval_statistics['StdPEARLCount_all_test_tasks_wo_mpc'] = np.std(test_final_pearl_count_wo_mpc)


            # avg_train_return = np.mean(train_final_returns)
            # std_train_return = np.std(train_final_returns)
            avg_test_return_col_free_wo_mpc = np.mean(test_final_returns_col_free_wo_mpc)
            std_test_return_col_free_wo_mpc = np.std(test_final_returns_col_free_wo_mpc)
            # avg_train_online_return = np.mean(np.stack(train_online_returns), axis=0)
            avg_test_online_return_wo_mpc = np.mean(np.stack(test_online_returns_wo_mpc), axis=0)
            self.eval_statistics['Collision_free_ratio_wo_mpc'] = test_col_free_ratio_wo_mpc
            # self.eval_statistics['AverageTrainReturn_all_train_tasks'] = train_returns
            # self.eval_statistics['AverageReturn_all_train_tasks'] = avg_train_return
            self.eval_statistics['AverageReturn_all_test_tasks_col_free_wo_mpc'] = avg_test_return_col_free_wo_mpc
            # self.eval_statistics['StdReturn_all_train_tasks'] = std_train_return
            self.eval_statistics['StdReturn_all_test_tasks_col_free_wo_mpc'] = std_test_return_col_free_wo_mpc

            # Collision and goal reach count
            # self.eval_statistics['AverageCollisionHuman_all_train_tasks'] = np.mean(train_final_cols_human)
            self.eval_statistics['AverageCollisionHuman_all_test_tasks_col_free_wo_mpc'] = np.mean(test_final_cols_human_col_free_wo_mpc)
            # self.eval_statistics['StdCollisionHuman_all_train_tasks'] = np.std(train_final_cols_human)
            self.eval_statistics['StdCollisionHuman_all_test_tasks_col_free_wo_mpc'] = np.std(test_final_cols_human_col_free_wo_mpc)

            # self.eval_statistics['AverageCollisionTable_all_train_tasks'] = np.mean(train_final_cols_table)
            self.eval_statistics['AverageCollisionTable_all_test_tasks_col_free_wo_mpc'] = np.mean(test_final_cols_table_col_free_wo_mpc)
            # self.eval_statistics['StdCollisionTable_all_train_tasks'] = np.std(train_final_cols_table)
            self.eval_statistics['StdCollisionTable_all_test_tasks_col_free_wo_mpc'] = np.std(test_final_cols_table_col_free_wo_mpc)

            # self.eval_statistics['AverageConditionalCollisionTable_all_train_tasks'] = train_mean_conditional_table_collision_count

            # self.eval_statistics['AverageReaches_all_train_tasks'] = np.mean(train_final_reaches)
            self.eval_statistics['AverageReaches_all_test_tasks_col_free_wo_mpc'] = np.mean(test_final_reaches_col_free_wo_mpc)
            # self.eval_statistics['StdReaches_all_train_tasks'] = np.std(train_final_reaches)
            self.eval_statistics['StdReaches_all_test_tasks_col_free_wo_mpc'] = np.std(test_final_reaches_col_free_wo_mpc)

            # self.eval_statistics['AverageNaviParts_all_train_tasks'] = np.mean(train_final_navi_parts)
            self.eval_statistics['AverageNaviParts_all_test_tasks_col_free_wo_mpc'] = np.mean(test_final_navi_parts_col_free_wo_mpc)
            # self.eval_statistics['StdNaviParts_all_train_tasks'] = np.std(train_final_navi_parts)
            self.eval_statistics['StdNaviParts_all_test_tasks_col_free_wo_mpc'] = np.std(test_final_navi_parts_col_free_wo_mpc)

            # self.eval_statistics['AverageColParts_all_train_tasks'] = np.mean(train_final_col_parts)
            self.eval_statistics['AverageColParts_all_test_tasks_col_free_wo_mpc'] = np.mean(test_final_col_parts_col_free_wo_mpc)
            # self.eval_statistics['StdColParts_all_train_tasks'] = np.std(train_final_col_parts)
            self.eval_statistics['StdColParts_all_test_tasks_col_free_wo_mpc'] = np.std(test_final_col_parts_col_free_wo_mpc)

            # self.eval_statistics['AverageCollisionHumanProbability_all_train_tasks'] = np.mean(train_final_alerts_human)
            self.eval_statistics['AverageCollisionHumanProbability_all_test_tasks_col_free_wo_mpc'] = np.mean(test_final_alerts_human_col_free_wo_mpc)
            # self.eval_statistics['StdCollisionHumanProbability_all_train_tasks'] = np.std(train_final_alerts_human)
            self.eval_statistics['StdCollisionHumanProbability_all_test_tasks_col_free_wo_mpc'] = np.std(test_final_alerts_human_col_free_wo_mpc)

            # self.eval_statistics['AverageCollisionTableProbability_all_train_tasks'] = np.mean(train_final_alerts_table)
            self.eval_statistics['AverageCollisionTableProbability_all_test_tasks_col_free_wo_mpc'] = np.mean(test_final_alerts_table_col_free_wo_mpc)
            # self.eval_statistics['StdCollisionTableProbability_all_train_tasks'] = np.std(train_final_alerts_table)
            self.eval_statistics['StdCollisionTableProbability_all_test_tasks_col_free_wo_mpc'] = np.std(test_final_alerts_table_col_free_wo_mpc)

            # self.eval_statistics['AverageMPCCount_all_train_tasks'] = np.mean(train_final_mpc_count)
            self.eval_statistics['AverageMPCCount_all_test_tasks_col_free_wo_mpc'] = np.mean(test_final_mpc_count_col_free_wo_mpc)
            # self.eval_statistics['StdMPCCount_all_train_tasks'] = np.std(train_final_mpc_count)
            self.eval_statistics['StdMPCCount_all_test_tasks_col_free_wo_mpc'] = np.std(test_final_mpc_count_col_free_wo_mpc)

            # self.eval_statistics['AveragePEARLCount_all_train_tasks'] = np.mean(train_final_pearl_count)
            self.eval_statistics['AveragePEARLCount_all_test_tasks_col_free_wo_mpc'] = np.mean(test_final_pearl_count_col_free_wo_mpc)
            # self.eval_statistics['StdPEARLCount_all_train_tasks'] = np.std(train_final_pearl_count)
            self.eval_statistics['StdPEARLCount_all_test_tasks_col_free_wo_mpc'] = np.std(test_final_pearl_count_col_free_wo_mpc)

            avg_test_return_col_free_success_wo_mpc = np.mean(test_final_returns_col_free_success_wo_mpc)
            std_test_return_col_free_success_wo_mpc = np.std(test_final_returns_col_free_success_wo_mpc)
            # avg_train_online_return = np.mean(np.stack(train_online_returns), axis=0)
            # avg_test_online_return_wo_mpc = np.mean(np.stack(test_online_returns_wo_mpc), axis=0)
            self.eval_statistics['Collision_free_success_ratio_wo_mpc'] = test_col_free_success_ratio_wo_mpc
            # self.eval_statistics['AverageTrainReturn_all_train_tasks'] = train_returns
            # self.eval_statistics['AverageReturn_all_train_tasks'] = avg_train_return
            self.eval_statistics['AverageReturn_all_test_tasks_col_free_success_wo_mpc'] = avg_test_return_col_free_success_wo_mpc
            # self.eval_statistics['StdReturn_all_train_tasks'] = std_train_return
            self.eval_statistics['StdReturn_all_test_tasks_col_free_success_wo_mpc'] = std_test_return_col_free_success_wo_mpc

            # Collision and goal reach count
            # self.eval_statistics['AverageCollisionHuman_all_train_tasks'] = np.mean(train_final_cols_human)
            self.eval_statistics['AverageCollisionHuman_all_test_tasks_col_free_success_wo_mpc'] = np.mean(
                test_final_cols_human_col_free_success_wo_mpc)
            # self.eval_statistics['StdCollisionHuman_all_train_tasks'] = np.std(train_final_cols_human)
            self.eval_statistics['StdCollisionHuman_all_test_tasks_col_free_success_wo_mpc'] = np.std(
                test_final_cols_human_col_free_success_wo_mpc)

            # self.eval_statistics['AverageCollisionTable_all_train_tasks'] = np.mean(train_final_cols_table)
            self.eval_statistics['AverageCollisionTable_all_test_tasks_col_free_success_wo_mpc'] = np.mean(
                test_final_cols_table_col_free_success_wo_mpc)
            # self.eval_statistics['StdCollisionTable_all_train_tasks'] = np.std(train_final_cols_table)
            self.eval_statistics['StdCollisionTable_all_test_tasks_col_free_success_wo_mpc'] = np.std(
                test_final_cols_table_col_free_success_wo_mpc)

            # self.eval_statistics['AverageConditionalCollisionTable_all_train_tasks'] = train_mean_conditional_table_collision_count

            # self.eval_statistics['AverageReaches_all_train_tasks'] = np.mean(train_final_reaches)
            self.eval_statistics['AverageReaches_all_test_tasks_col_free_success_wo_mpc'] = np.mean(
                test_final_reaches_col_free_success_wo_mpc)
            # self.eval_statistics['StdReaches_all_train_tasks'] = np.std(train_final_reaches)
            self.eval_statistics['StdReaches_all_test_tasks_col_free_success_wo_mpc'] = np.std(
                test_final_reaches_col_free_success_wo_mpc)

            # self.eval_statistics['AverageNaviParts_all_train_tasks'] = np.mean(train_final_navi_parts)
            self.eval_statistics['AverageNaviParts_all_test_tasks_col_free_success_wo_mpc'] = np.mean(
                test_final_navi_parts_col_free_success_wo_mpc)
            # self.eval_statistics['StdNaviParts_all_train_tasks'] = np.std(train_final_navi_parts)
            self.eval_statistics['StdNaviParts_all_test_tasks_col_free_success_wo_mpc'] = np.std(
                test_final_navi_parts_col_free_success_wo_mpc)

            # self.eval_statistics['AverageColParts_all_train_tasks'] = np.mean(train_final_col_parts)
            self.eval_statistics['AverageColParts_all_test_tasks_col_free_success_wo_mpc'] = np.mean(
                test_final_col_parts_col_free_success_wo_mpc)
            # self.eval_statistics['StdColParts_all_train_tasks'] = np.std(train_final_col_parts)
            self.eval_statistics['StdColParts_all_test_tasks_col_free_success_wo_mpc'] = np.std(
                test_final_col_parts_col_free_success_wo_mpc)

            # self.eval_statistics['AverageCollisionHumanProbability_all_train_tasks'] = np.mean(train_final_alerts_human)
            self.eval_statistics['AverageCollisionHumanProbability_all_test_tasks_col_free_success_wo_mpc'] = np.mean(
                test_final_alerts_human_col_free_success_wo_mpc)
            # self.eval_statistics['StdCollisionHumanProbability_all_train_tasks'] = np.std(train_final_alerts_human)
            self.eval_statistics['StdCollisionHumanProbability_all_test_tasks_col_free_success_wo_mpc'] = np.std(
                test_final_alerts_human_col_free_success_wo_mpc)

            # self.eval_statistics['AverageCollisionTableProbability_all_train_tasks'] = np.mean(train_final_alerts_table)
            self.eval_statistics['AverageCollisionTableProbability_all_test_tasks_col_free_success_wo_mpc'] = np.mean(
                test_final_alerts_table_col_free_success_wo_mpc)
            # self.eval_statistics['StdCollisionTableProbability_all_train_tasks'] = np.std(train_final_alerts_table)
            self.eval_statistics['StdCollisionTableProbability_all_test_tasks_col_free_success_wo_mpc'] = np.std(
                test_final_alerts_table_col_free_success_wo_mpc)

            # self.eval_statistics['AverageMPCCount_all_train_tasks'] = np.mean(train_final_mpc_count)
            self.eval_statistics['AverageMPCCount_all_test_tasks_col_free_success_wo_mpc'] = np.mean(
                test_final_mpc_count_col_free_success_wo_mpc)
            # self.eval_statistics['StdMPCCount_all_train_tasks'] = np.std(train_final_mpc_count)
            self.eval_statistics['StdMPCCount_all_test_tasks_col_free_success_wo_mpc'] = np.std(
                test_final_mpc_count_col_free_success_wo_mpc)

            # self.eval_statistics['AveragePEARLCount_all_train_tasks'] = np.mean(train_final_pearl_count)
            self.eval_statistics['AveragePEARLCount_all_test_tasks_col_free_success_wo_mpc'] = np.mean(
                test_final_pearl_count_col_free_success_wo_mpc)
            # self.eval_statistics['StdPEARLCount_all_train_tasks'] = np.std(train_final_pearl_count)
            self.eval_statistics['StdPEARLCount_all_test_tasks_col_free_success_wo_mpc'] = np.std(
                test_final_pearl_count_col_free_success_wo_mpc)

        # self.eval_statistics['Attempts_GP'] = gp_count
        # self.eval_statistics['Fail_GP'] = gp_fail
        
        # logger.save_extra_data(avg_train_online_return, path='online-train-epoch{}'.format(epoch))
        # logger.save_extra_data(avg_test_online_return, path='online-test-epoch{}'.format(epoch))

        for key, value in self.eval_statistics.items():
            logger.record_tabular(key, value)
        self.eval_statistics = None

        # if self.render_eval_paths:
        #     self.env.render_paths(paths)

        # if self.plotter:
        #     self.plotter.draw()

    @abc.abstractmethod
    def training_mode(self, mode):
        """
        Set training mode to `mode`.
        :param mode: If True, training will happen (e.g. set the dropout
        probabilities to not all ones).
        """
        pass

    @abc.abstractmethod
    def _do_training(self):
        """
        Perform some update, e.g. perform one gradient step.
        :return:
        """
        pass

