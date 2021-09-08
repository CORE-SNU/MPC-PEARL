import numpy as np
import warnings
import time
from multiprocessing import Pool

# Another repo for GP!
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel, ConstantKernel, RBF
import casadi as ca
import GPy



def rollout(
        env,
        agent,
        max_path_length=np.inf,
        accum_context=True,
        train=True,
        animated=False,

        use_MPC=False,
        mpc_solver=None,
        gp_human=None,
        test=False,
        pred=None,
):
    """
    The following value for the following keys will be a 2D array, with the
    first dimension corresponding to the time dimension.
     - observations
     - actions
     - rewards
     - next_observations
     - terminals

    The next two elements will be lists of dictionaries, with the index into
    the list being the index into the time
     - agent_infos
     - env_infos

    :param env:
    :param agent:
    :param max_path_length:
    :param accum_context: if True, accumulate the collected context
    :param animated:
    :return:
    """

    observations = []
    next_observations = []

    unnormalized_observations = []
    unnormalized_next_observations = []

    actions = []
    rewards = []
    terminals = []
    agent_infos = []
    env_infos = []
    o, normalized_o = env.reset()
    next_o = None
    path_length = 0
    pred_horizon = env.Nt
    N_adapt = env.N_adapt

    # Custom information
    collisions_human = []
    collisions_table = []
    alerts_human = []
    alerts_table = []
    reaches = []
    navi_parts = []
    col_parts = []

    if animated:
        env.render(pred=pred)

    # Data for obstacle GP
    prediction = []
    max_data_size = env.Nt
    min_data_size = env.Nt // 4
    solver_opts = {}
    solver_opts['expand'] = False
    human = []
    human_next = []
    dynamic = []
    use_GP, no_GP = 0, 0
    mpc_count, pearl_count = 0, 0

    # Collision metric
    alert_radii = 2.
    
    # For periodic use of ther interval
    collision_history = 0
    goal_history = 0

    # To use only human data for GP
    # obs = env.restaurant.table_list

    while path_length < max_path_length:
        cond1 = env.goal_event
        '''
        if cond1:
            # cond1 = env.goal_event and (goal_history % env.goal_interval == env.goal_interval - 1)
            cond1 = env.goal_event and (np.random.rand() < env.eps)
            goal_history += 1
        '''
        # cond2 = (path_length % env.interval == env.interval - 1)
        cond2 = False
        cond3 = env.collision_event
        '''
        if cond3:
            # cond3 = env.collision_event and (collision_history % env.col_interval == env.col_interval - 1)
            cond3 = env.collision_event and (np.random.rand() < env.eps)  
            collision_history += 1
        '''
        turn_on_mpc = (cond1 or cond2 or cond3) and use_MPC
        turn_on_mpc = turn_on_mpc and (np.random.rand() < env.eps)
        turn_on_mpc = turn_on_mpc or test
        # turn_on_mpc = turn_on_mpc and env.task_idx < 180
        # build prediction matrix and initialize
        pred_cov = 0* np.ones((env.Nt, env.num_people, 2))
        pred = np.zeros((env.Nt, env.num_people, 2))
        for t in range(pred.shape[0]):
            pred[t, :, :] = o[3:3+2*env.num_people].reshape(-1, pred.shape[2])

        # Build distance from table for safety metric
        table_dist, alert_table = np.zeros(env.num_tables), np.zeros(env.num_tables)
        for i, table in enumerate(env.restaurant.table_list):
            table_dist[i] = np.linalg.norm(table.center - o[:2])
            if table_dist[i] < alert_radii:
                alert_table[i] = 1
        # Adaptive set of table to use (closest ones)
        adapt_table_idxs = np.argsort(table_dist)[:N_adapt]        # [0, 1, 5, 4, 2]
        adapt_table_list = [env.restaurant.table_list[i] for i in adapt_table_idxs]

        # Build distance from human for safety metric
        human_dist, alert_human = np.zeros(env.num_people), np.zeros(env.num_people)
        for i, human in enumerate(env.restaurant.human_list):
            human_dist[i] = np.linalg.norm(human.center - o[:2])
            if human_dist[i] < alert_radii:
                alert_human[i] = 1
        # Adaptive set of dynamic obstacle to use (closest ones)
        adapt_human_idxs = np.argsort(human_dist)[:N_adapt]        # [0, 1, 5, 4, 2]

        # use GP-MPC if it is turned on

        if env.landed:
            # If the robot lands on the goal, then do not give a control anymore.
            # normalized zero control : lower bound of linear velocity = 0
            a = np.array([0., 0.]) if turn_on_mpc else np.array([-1., 0.])
            agent_info = {}
            Xout = None

        else:
            if turn_on_mpc:
                # Use GP to build constraints
                gp_effective = np.zeros(env.num_people)
                if len(observations) > min_data_size:
                    '''
                    pred, pred_cov, gp_effective = fit_GP_adaptive(unnormalized_observations,
                                                                   unnormalized_next_observations,
                                                                   pred,
                                                                   pred_cov,
                                                                   max_data_size=max_data_size,
                                                                   min_data_size=min_data_size,
                                                                   num_people=env.num_people,
                                                                   pred_horizon=env.Nt,
                                                                   N_adapt=N_adapt
                                                                   )
                    '''
                    pred, pred_cov, gp_effective = _load_GP_predictions(unnormalized_observations,
                                                                        unnormalized_next_observations,
                                                                        pred,
                                                                        pred_cov,
                                                                        env,
                                                                        agent,
                                                                        max_data_size=10,
                                                                        range_adapt=alert_radii,
                                                                        N_adapt=N_adapt
                                                                        )
                    
                # if (rand_var <= env.eps and env.event) or (col and use_MPC and test):
                # Caution! You must pass only the list of static obstacles as an argument to GP-MPC solver
                # TODO2: Pass GP-generated K-step obstacle prediction to MPC as constraint
                # print([o.center for o in table_list_new])
                Xout, a_MPC, flag = mpc_solver.plan_pol(
                    o[:3],
                    env._goal,  # Goal position given to MPC
                    *env.cost_weight,
                    env.restaurant.table_radii,
                    env.restaurant.human_radii,
                    env.restaurant.table_list,
                    # adapt_table_list,
                    pred[:, adapt_human_idxs, :],
                    pred_cov[:, adapt_human_idxs, :],
                    env._counter,
                    gp_effective[adapt_human_idxs]
                )
                
                #a_MPC = np.array([0, 0])
                agent_info = {}
                if flag == 1 or flag == 0:
                    a = a_MPC
                    mpc_count += 1

                else:
                    #backup controller (PEARL)
                    # a, agent_info = agent.get_action(o)
                    #backup controller (0)
                    # a, agent_info = agent.get_action(normalized_o)
                    a = a_MPC
                    Xout = None
                    pearl_count += 1

            else:
                Xout = None
                a, agent_info = agent.get_action(normalized_o)

        prediction.append(pred)
        
        # episode roll-out
        next_o, r, d, a, _, info_custom, env_info, normalized_next_o = env.step(a, train, Xout=Xout, use_MPC=turn_on_mpc)
  
        # append custom info
        collision_human, collision_table, reach, navi_part, col_part = info_custom
        collisions_human.append(collision_human)
        collisions_table.append(collision_table)
        reaches.append(reach)
        navi_parts.append(navi_part)
        col_parts.append(col_part)
        alerts_human.append(np.sum(alert_table))
        alerts_table.append(np.sum(alert_table))
        # update the agent's current context
        human_next.append(next_o[3:3+2*env.num_people].reshape(-1, 2))
        if accum_context:
            agent.update_context([normalized_o, a, r, normalized_next_o, d, env_info])
        # observations.append(o)

        # for GP-MPC
        unnormalized_observations.append(o)
        unnormalized_next_observations.append(next_o)
        # next_observations.append(next_o)

        # for PEARL
        observations.append(normalized_o)
        next_observations.append(normalized_next_o)
        rewards.append(r)
        terminals.append(d)
        actions.append(a)
        agent_infos.append(agent_info)
        path_length += 1
        o = next_o
        normalized_o = normalized_next_o
        if animated:
            if use_MPC:
                env.render(pred=prediction)
            else:
                env.render(pred=None)

        env_infos.append(env_info)
        if d:
            break

    """ Summarize results """
    actions = np.array(actions)
    if len(actions.shape) == 1:
        actions = np.expand_dims(actions, 1)
    observations = np.array(observations)
    if len(observations.shape) == 1:
        observations = np.expand_dims(observations, 1)
        normalized_next_o = np.array([normalized_next_o])
    next_observations = np.vstack(
        (
            observations[1:, :],
            np.expand_dims(normalized_next_o, 0)
        )
    )
    return dict(
        observations=observations,
        actions=actions,
        rewards=np.array(rewards).reshape(-1, 1),
        next_observations=next_observations,
        terminals=np.array(terminals).reshape(-1, 1),
        agent_infos=agent_infos,
        env_infos=env_infos,
        collisions_human=collisions_human,
        collisions_table=collisions_table,
        reaches=reaches,
        navi_parts=np.array(navi_parts).reshape(-1, 1),
        col_parts=np.array(col_parts).reshape(-1, 1),
        gp_human=(use_GP, no_GP),
        pred=pred,
        alerts_human=alerts_human,
        alerts_table=alerts_table,
        mpc_count=[mpc_count],
        pearl_count=[pearl_count],
    )


def split_paths(paths):
    """
    Stack multiples obs/actions/etc. from different paths
    :param paths: List of paths, where one path is something returned from
    the rollout functino above.
    :return: Tuple. Every element will have shape batch_size X DIM, including
    the rewards and terminal flags.
    """
    rewards = [path["rewards"].reshape(-1, 1) for path in paths]
    terminals = [path["terminals"].reshape(-1, 1) for path in paths]
    actions = [path["actions"] for path in paths]
    obs = [path["observations"] for path in paths]
    next_obs = [path["next_observations"] for path in paths]
    rewards = np.vstack(rewards)
    terminals = np.vstack(terminals)
    obs = np.vstack(obs)
    actions = np.vstack(actions)
    next_obs = np.vstack(next_obs)
    assert len(rewards.shape) == 2
    assert len(terminals.shape) == 2
    assert len(obs.shape) == 2
    assert len(actions.shape) == 2
    assert len(next_obs.shape) == 2
    return rewards, terminals, obs, actions, next_obs


def split_paths_to_dict(paths):
    rewards, terminals, obs, actions, next_obs = split_paths(paths)
    return dict(
        rewards=rewards,
        terminals=terminals,
        observations=obs,
        actions=actions,
        next_observations=next_obs,
    )
'''
def get_stat_in_paths(paths, dict_name, scalar_name):
    if len(paths) == 0:
        return np.array([[]])

    if type(paths[0][dict_name]) == dict:
        # Support rllab interface
        return [path[dict_name][scalar_name] for path in paths]

    return [
        [info[scalar_name] for info in path[dict_name]]
        for path in paths
    ]
'''
def fit_GP_adaptive(obs, next_obs, pred, pred_cov, max_data_size=10, min_data_size=2, num_people=5, pred_horizon=10, N_adapt=2):
    # Adaptively use closest obstacle to fit GP

    # Maximum data length
    obs, next_obs = obs[-max_data_size:], next_obs[-max_data_size:]

    # Current robot position
    robot_pos = next_obs[-1][:2]

    # Calculate velocity and distance from agent to select data to fit GP
    v, dist = {i:[] for i in range(num_people)}, {i:[] for i in range(num_people)}
    human, next_human = [], []
    for human_pos, next_human_pos in zip(obs, next_obs):
        human_pos = np.array(human_pos[3:3+2*num_people]).reshape(-1,2)
        next_human_pos = np.array(next_human_pos[3:3+2*num_people]).reshape(-1,2)
        human.append(human_pos)
        next_human.append(next_human_pos)
    for i in range(num_people):
        v[i] = np.linalg.norm( next_human[-1][i, :] - human[-1][i, :] )
        dist[i] = np.linalg.norm( next_human[-1][i, :] - robot_pos )

    # Adapatively select obstacle to fit GP
    adaptive_set, gp_effective = [], np.zeros(num_people)
    min_v = 1e-3    # Don't use fixed obstacle for speed-up
    range_adapt = 2.
    for i in range(num_people):
        if np.max(v[i]) > min_v:
            if np.min(dist[i]) < range_adapt:   # See if obstacle is in range
                adaptive_set.append(i)
                gp_effective[i] = 1

    if len(adaptive_set) < 1:
        return pred, pred_cov, gp_effective

    # No multiprocessing version
    for i in adaptive_set:
        X = [pos[i, :].reshape(1,-1)[0] for pos in human]
        y = [pos[i, :].reshape(1,-1)[0] for pos in next_human]
        # y0 = [pos[i, 0].reshape(1,-1)[0] for pos in next_human]
        # y1 = [pos[i, 1].reshape(1,-1)[0] for pos in next_human]
        #y_diff = [y[i] - X[i] for i in range(len(X))]
        y_diff0 = [y[i][0] - X[i][0] + np.random.normal(0, 0.001) for i in range(len(X))]
        y_diff1 = [y[i][1] - X[i][1] + np.random.normal(0, 0.001) for i in range(len(X))]
        #y_diff0 = [y[i][0] - X[i][0] for i in range(len(X))]
        #y_diff1 = [y[i][1] - X[i][1] for i in range(len(X))]
        #kernel = 1.0 + ConstantKernel()*RBF()

        data_len = max_data_size
        while data_len >= min_data_size:

            kernel = GPy.kern.RBF(input_dim=2, variance=1., lengthscale=1.)
                    #+ GPy.kern.Bias(input_dim=2)
            kernel1 = GPy.kern.RBF(input_dim=2, variance=1., lengthscale=1.)
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                X_train = np.array(X[-data_len:])
                Y_train1 = np.array([np.array(y_diff0[-data_len:])]).T
                Y_train2 = np.array([np.array(y_diff1[-data_len:])]).T
                gp_human0 = GPy.models.GPRegression(X_train, Y_train1, kernel)
                gp_human0.optimize(messages=False)

                gp_human1 = GPy.models.GPRegression(X_train, Y_train2, kernel1)
                gp_human1.optimize(messages=False)

                gp_human1.optimize_restarts(num_restarts=5, verbose=False)
                gp_human0.optimize_restarts(num_restarts=5, verbose=False)
            # gp_human0.optimize_restarts(num_restarts=10)
            break
            '''
            gp_human1 = GPy.models.GPRegression(np.array(X[-data_len:]), np.array(y_diff1[-data_len:]), kernel)
            gp_human1.optimize(messages=True, num_restarts=10)

            raise ValueError ('Good enough?')
            '''
            '''
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
            
                gp_human0 = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=20)
                # gp_human0.fit(X[-data_len:], y0[-data_len:])
                gp_human0.fit(X[-data_len:], y_diff0[-data_len:])

                gp_human1 = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=20)
                # gp_human1.fit(X[-data_len:], y1[-data_len:])
                gp_human1.fit(X[-data_len:], y_diff1[-data_len:])
            
                if len(w) == 0:
                    print('Successfully fitted GP!')
                    break
            '''
            data_len -= 1

        cur_state = y[-1]
        cur_cov = [0, 0]
        cov_v = [0, 0]
        cov_xv = [0, 0]

        for t in range(pred_horizon):
            # Return prediction!
            # print(cur_state)
            pred[t, i, :] = np.array(cur_state.copy()).reshape(-1,2)
            pred_cov[t, i, :] = np.array(cur_cov)
            #print(np.array([np.array(cur_state)]).shape)
            vel_0, cur_cov0 = gp_human0.predict(np.array([np.array(cur_state)]), include_likelihood=False)
            vel_1, cur_cov1 = gp_human1.predict(np.array([np.array(cur_state)]), include_likelihood=False)

            #Our implementation of GP prediction
            #vel_0, cur_cov0 = predict(np.array([np.array(cur_state)]), X_train, Y_train1, gp_human0.param_array, kernel)
            #vel_1, cur_cov1 = predict(np.array([np.array(cur_state)]), X_train, Y_train2, gp_human1.param_array, kernel1)
            
            #Taylor expansion
            #dmu1_dX, _ = gp_human0.predictive_gradients(np.array([np.array(cur_state)]))
            #dmu2_dX, _ = gp_human1.predictive_gradients(np.array([np.array(cur_state)]))
            #cov_v[0] = cur_cov0[0][0] + (dmu1_dX[:,:,0] @ np.diag(cur_cov) @ dmu1_dX[:,:,0].T)[0,0]
            #cov_v[1] = cur_cov1[0][0] + (dmu2_dX[:,:,0] @ np.diag(cur_cov) @ dmu2_dX[:,:,0].T)[0,0]
            #cov_xv[0] = (np.diag(cur_cov) @ dmu1_dX[:,:,0].T)[0,0]
            #cov_xv[1] = (np.diag(cur_cov) @ dmu2_dX[:,:,0].T)[0,0]
            #cur_cov[0] = cur_cov[0] + cov_v[0] + 2*cov_xv[0]
            #cur_cov[1] = cur_cov[1] + cov_v[1] + 2*cov_xv[1]
            
            #Mean Equivalent
            cur_cov[0] = cur_cov[0] + cur_cov0[0][0]
            cur_cov[1] = cur_cov[1] + cur_cov1[0][0]
            
            cur_state0 = cur_state[0] + vel_0[0]
            cur_state1 = cur_state[1] + vel_1[0]
            cur_state = [cur_state0[0], cur_state1[0]]
            #cur_cov = [np.max([cur_cov0[0][0], 1e-5]), np.max([cur_cov1[0][0], 1e-5])]
            
            #cur_cov = [1e-5, 1e-5]
            #print('t: {}, mean: {}, cov: {}'.format(t, cur_state, cur_cov))
    """
            cur_state = y[-1]
        cur_cov = [0, 0]
        for t in range(pred_horizon):
            # Return prediction!
            # print(cur_state)
            pred[t, i, :] = np.array(cur_state.copy()).reshape(-1,2)
            pred_cov[t, i, :] = np.array(cur_cov)
            vel_0, cur_cov0 = gp_human0.predict([cur_state], return_cov = True)
            vel_1, cur_cov1 = gp_human1.predict([cur_state], return_cov = True)
            cur_state0 = cur_state[0] + vel_0[0]
            cur_state1 = cur_state[1] + vel_1[0]
            cur_state = [cur_state0, cur_state1]
            cur_cov = [cur_cov0[0][0], cur_cov1[0][0]]

    data_len = max_data_size
    while data_len >= min_data_size:
        args_mp = []
        
        for i in adaptive_set:
            X = [pos[i, :].reshape(1,-1)[0] for pos in human]
            y = [pos[i, :].reshape(1,-1)[0] for pos in next_human]
            y_diff0 = [y[i][0] - X[i][0] for i in range(len(X))]
            y_diff1 = [y[i][1] - X[i][1] for i in range(len(X))]
            args_mp.append((i, X, y_diff0, y_diff1, data_len))

        with Pool(10) as p:
            res = p.map(_fit_GP, args_mp)

        for i, gp_human0, gp_human1, success in res:
            if success:
                print('Successfully fitted GP!')
                cur_state = y[-1]
                for t in range(pred_horizon):
                    # Return prediction!
                    pred[t, i, :] = np.array(cur_state.copy()).reshape(-1,2)
                    cur_state0 = cur_state[0] + gp_human0.predict([cur_state])[0]
                    cur_state1 = cur_state[1] + gp_human1.predict([cur_state])[0]
                    cur_state = [cur_state0, cur_state1]
                adaptive_set.remove(i)

        if len(adaptive_set) == 0:
            break

        data_len -= 1
    """

    #print('')
    #print(f"Time for GPy {timer1:0.4f} seconds")
    #print(f"Time for My {timer2:0.4f} seconds")
    return pred, pred_cov, gp_effective

def _fit_GP(args):
    i, X, y_diff0, y_diff1, data_len = args
    kernel = 1.0 + ConstantKernel() * RBF()
    success = False
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        gp_human0 = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=20)
        # gp_human0.fit(X[-data_len:], y0[-data_len:])
        gp_human0.fit(X[-data_len:], y_diff0[-data_len:])

        gp_human1 = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=20)
        # gp_human1.fit(X[-data_len:], y1[-data_len:])
        gp_human1.fit(X[-data_len:], y_diff1[-data_len:])

        if len(w) == 0:
            success = True

    return i, gp_human0, gp_human1, success

def _load_GP_predictions(obs, next_obs, pred_mean, pred_cov, env, agent, max_data_size=10, range_adapt=2., N_adapt=2):
    # Load pretrained GP
    pred_mean_offline, pred_cov_offline = agent.gp_human[env.task_idx]['mean'], agent.gp_human[env.task_idx]['cov']

    # Adaptively use closest obstacle to fit GP
    num_people = env.num_people

    # Maximum data length
    obs, next_obs = obs[-max_data_size:], next_obs[-max_data_size:]

    # Current robot position
    robot_pos = next_obs[-1][:2]

    # Calculate velocity and distance from agent to select data to fit GP
    v, dist = {i:[] for i in range(num_people)}, {i:[] for i in range(num_people)}
    human, next_human = [], []
    for human_pos, next_human_pos in zip(obs, next_obs):
        human_pos = np.array(human_pos[3:3+2*num_people]).reshape(-1,2)
        next_human_pos = np.array(next_human_pos[3:3+2*num_people]).reshape(-1,2)
        human.append(human_pos)
        next_human.append(next_human_pos)
    for i in range(num_people):
        v[i] = np.linalg.norm( next_human[-1][i, :] - human[-1][i, :] )
        dist[i] = np.linalg.norm( next_human[-1][i, :] - robot_pos )

    # Adapatively select obstacle to fit GP
    adaptive_set, gp_effective = [], np.zeros(num_people)
    min_v = 1e-3	# Don't use fixed obstacle for speed-up
    for i in range(num_people):
        if np.max(v[i]) > min_v:
            if np.min(dist[i]) < range_adapt:	# See if obstacle is in range
                adaptive_set.append(i)
                gp_effective[i] = 1

    #adaptive_set = [i for i in range(num_people)]
    #print('obstacle in use :',len(adaptive_set))

    if len(adaptive_set) < 1:
        return pred_mean, pred_cov, gp_effective

    # No multiprocessing version
    for i in adaptive_set:
        pred_mean[:, i, :], pred_cov[:, i, :] =  pred_mean_offline[env._counter][:env.Nt, i, :], pred_cov_offline[env._counter][:env.Nt, i, :]
    # pred_cov = np.clip(pred_cov, 1e-5, np.inf)

    return pred_mean, pred_cov, gp_effective

def predict(X, X_train, Y_train, params, kern):
    sigma_n = params[2]
    #Le = params[1]
    n = X_train.shape[0]
    K = kern.K(X_train)
    k = kern.K(X_train, X)
    L = np.linalg.cholesky(K+sigma_n*np.eye(n))
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, Y_train))
    mu = k.T @ alpha
    v = np.linalg.solve(L, k)
    sigma = kern.K(X, X) - v.T @ v
    #dmu_dx = (k * (X_train - X)/(Le**2)).T @ alpha
    return mu, sigma

def RBF(x,y,params):
    sigma = params[0]
    L = params[1]
    dim1 = x.shape[0]
    dim2 = y.shape[0]
    K = np.zeros((dim1, dim2))
    for i in range(dim1):
        for j in range(dim2):
            K[i,j] = sigma*np.exp(0.5*np.dot(x[i]-y[j], x[i]-y[j])/(L**2))
    return K

