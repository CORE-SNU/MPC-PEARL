import numpy as np
from os import path
import time
from gym import Env
import gym.spaces as spaces
from . import register_env
from rlkit.envs.utils.restaurant import Restaurant
from rlkit.envs.utils.obstacles import Human
import gc
# from rlkit.envs.planner.MPC_planning import updatePlots_pol, plan_pol

import matplotlib
# matplotlib.use('Agg')

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, writers, MovieWriter
import matplotlib.animation as animation
from matplotlib.patches import Wedge
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

import pandas as pd


@register_env('Navigation_MPC')
class NavigationToyEnv(Env):
    # Robot navigation in restaurant environment with some static & dynamic obstacles
    # We assume that the spawn point & goal point of the robot is fixed throughout tasks
    # Set of moving obstacle parameters defines a task distribution
    # State variables of the robot includes its own position & the position of each dynamic obstacle
    # However, we assume that the static obstacle information are unknown to the robot
    def __init__(self, randomize_tasks=True, n_tasks=2, use_MPC=0, ep_len=100, Nt=10):
        
        # ------------------------------ Hyperparameter setup ---------------------------------
        # Main hyperparameters
        self._obs_reward = 20.
        self._goal_reward = 10.
        self.eps = .2
         
        # Initial and goal state
        self._goal = np.array([3.5, -3.5])
        self.init_state = np.array([-4.5, 4.5, -np.pi / 8.])
        # --------------------------------------------------------------------------------------
        
        # For Periodic version
        self.interval = 5
        self.col_interval = 5
        self.goal_interval = 5
        
        self.n_tasks = n_tasks
        self.num_tasks_out_of_distribution = 10

        self.task_idx = None
        self.gp_human_list = None
        
        # self.interval_list = np.linspace(2,40,1000)
        self.interval_list = 5*np.ones(1000)
        # self.interval_list = np.concatenate((np.linspace(4, 80, 800), 80*np.ones(200)))
        
        self.max_dist = np.sqrt(((self.init_state[0:2]-self._goal)**2).sum())
        
        # -------------------------------- Simulation Setup ----------------------------------
        self.ep_len = ep_len
        self.dt = 0.05  # sampling interval
        self.N_adapt = 2
        self.goal_tresh = 2.
        # current step within episode
        # initialized once the episode is reset
        self._counter = 0
        # ------------------------------------------------------------------------------------

        # ----------------------------------- Robot Setup ------------------------------------
        # for simplicity, we use a single integrator model to describe robot dynamics
        # TODO : use more realistic vehicle dynamics
        self._buffer = np.zeros((self.ep_len, 3))
        # maximum velocity along each axis
        self.max_robot_vel = 7.
        self.max_robot_steering = np.pi / 2.
        self.L = 0.5               # radius of robot
        self.lr = self.L / 2.
        self.safe_rad = self.lr / 2.
        self.robot_state = None     # robot state variable
        self.goal_distance = None   # distance between robot & goal
        # buffer for rendering the robot agent
        self._buffer = np.zeros((self.ep_len, 3))
        self.prev_robot = None
        self.collision_history = None
        self.heading_angle_indicator = None
        # ------------------------------------------------------------------------------------

        # ------------------------------ Static Obstacle Setup -------------------------------
        # table positions remain unchanged even when a task is newly generated
        self.num_tables = 5
        self.table_radii = 0.5
        # ------------------------------------------------------------------------------------

        np.random.seed(2021)  # seeding for task distribution

        # ------------------------------ Dynamic Obstacle Setup ------------------------------
        # parameters of the system dynamics governing the motion of each moving obstacle
        # here we assume that each obstacle exhibits cyclic motion
        self.num_people = 6
        self.human_radii = 0.25

        self.buffer_for_people = None   # buffer for rendering moving obstacles
        self.prev_people = None         # again, just for rendering
        # -------------------------------------------------------------------------------------

        # for MPC
        self.use_MPC = use_MPC
        self.Nt = Nt
        self.Xout_buffer = []   # for MPC open-loop rendering

        # definition of observation space & action space
        obs_high = np.array([5., 5., np.pi] + [5.] * (2 * self.num_people) + [5.] * (2 * self.num_tables))
        obs_low = -obs_high
        ctrl_high = np.array([self.max_robot_vel, self.max_robot_steering])
        ctrl_low = np.array([0, -self.max_robot_steering])

        self.observation_space = spaces.Box(low=obs_low, high=obs_high)
        self.action_space = spaces.Box(low=ctrl_low, high=ctrl_high)

        # create a restaurant instance
        self.restaurant = Restaurant(
            num_tables=self.num_tables,
            num_tasks=n_tasks,
            num_people=self.num_people,
            table_radii=self.table_radii,
            human_radii=self.human_radii,
            ep_len=self.ep_len
        )

    def reset_task(self, idx):
        # sample a new task (i.e., task-level reset)
        # invocation of the method rearranges the restaurant environment
        # specifically, dynamics of each moving obstacles varies from task to task
        self.restaurant.reset_task(idx)
        self.task_idx = idx
        self.reset()

    def reset_task_out_of_distribution(self, idx):
        # sample a new task (i.e., task-level reset)
        # invocation of the method rearranges the restaurant environment
        # specifically, dynamics of each moving obstacles varies from task to task
        self.restaurant.reset_task_out_of_distribution(idx)
        self.task_idx = idx
        self.reset()

    def get_all_task_idx(self):
        return range(self.n_tasks)

    def reset_model(self):
        # episode-level reset
        self._counter = 0
        # initialization of restaurant
        # by calling reset method of the restaurant, we set the angular position of each moving obstacle to 0
        self.restaurant.reset()
        # initial position of the robot : upper-left corner of the restaurant
        self.robot_state = self.init_state
        #self.robot_state = np.array([0.75, 0, -np.pi / 2])
        self.goal_distance = ((self.robot_state[:2] - self._goal) ** 2).sum()

        # for rendering
        self.prev_people = []
        self.buffer_for_people = np.zeros((self.ep_len, 2 * self.num_people))
        self.Xout_buffer = []
        self.collision_history = []
        return self._get_obs(), self._get_normalized_obs()

    def reset(self):
        return self.reset_model()

    def _get_obs(self):
        # compute state/observation
        # Since table positions vary from task to task, it must be given as parts of state variables.
        return np.concatenate([self.robot_state, self.restaurant.human_vector, self.restaurant.table_vector])

    def _get_normalized_obs(self):
        # compute state/observation
        # Since table positions vary from task to task, it must be given as parts of state variables.

        return np.concatenate([np.array([self.robot_state[0] / 5.,
                               self.robot_state[1] / 5.,
                               self.robot_state[2] / np.pi]),
                               self.restaurant.human_vector / 5.,
                               self.restaurant.table_vector / 5.
                               ])

    def step(self, action: np.ndarray, train=True, Xout=None, init_state=None):
        # print('action :', action)
        # The following 4 lines are just for rendering purpose.
        if init_state is not None:
            self.robot_state = init_state
        if Xout is None:
            Xout = np.zeros((3, self.Nt))
        # print('robot! : ', self.robot_state)
        # ------------------------------ Logging for Rendering ------------------------------
        self.Xout_buffer.append(Xout)
        self._buffer[self._counter] = self.robot_state
        for i in range(self.num_people):
            self.buffer_for_people[self._counter, 2 * i: 2 * i + 2] = np.copy(self.restaurant.People[i].center)
        # -----------------------------------------------------------------------------------
        # -------------------------------- Vehicle Dynamics ---------------------------------
        beta = np.arctan(self.lr * np.tan(action[1]) / self.L)
        vdt = self.dt * action[0]
        next_robot_state = self.robot_state + vdt * np.array([np.cos(beta + self.robot_state[2]),
                                                              np.sin(beta + self.robot_state[2]),
                                                              np.tan(action[1]) * np.cos(beta) / self.L]
                                                             )
        '''
        next_robot_state = self.robot_state + vdt * np.array([1 - (beta + self.robot_state[2])**2 / 2,
                                                      (self.robot_state[2] + beta),
                                                      (self.robot_state[2] * np.tan(beta) * action[1]) / self.L]
                                                     )
        '''
        # state clipping : The robot must stay within the space.
        next_robot_state[2] = (next_robot_state[2] + np.pi) % (2. * np.pi) - np.pi     # \theta \in [-\pi, \pi)
        next_robot_state[:2] = np.clip(next_robot_state[:2], -5. + self.lr, 5. - self.lr)

        # ------------------------------- Reward Computation --------------------------------
        num_collision_table, num_collision_human = 0, 0
        self.restaurant.sim()
        """ Reward Setting """

        # reward weight parameters definition
        # must be synchronized with MPC cost function
        
        
        #--------------- Ratio!
        ratio = 3
        #----------------
        
        
        pos_weight, ctrl_weight, _ = self.cost_weight
       
        obs_weight = ratio * pos_weight[0]

        # <1> state & control cost
        ctrl_cost = (ctrl_weight * action ** 2.).sum()  # $\ell^2$-norm penalty assigned to control input
        navi_part = -(pos_weight[0] * self.goal_distance + ctrl_cost) / 25.
        # <2> obstacle collision penalty
        obs_cost = 0.
        for obstacle in self.restaurant.obstacle_list:

            d = np.sqrt(((next_robot_state[:2] - obstacle.center) ** 2).sum())
            r = obstacle.radii + self.lr + self.safe_rad
            d_pred = np.sqrt(((Xout[:2, 1] - obstacle.center) ** 2).sum())

            if d < r:
                if d < r - self.safe_rad:
                    if isinstance(obstacle, Human):
                        num_collision_human += 1
                    else:
                        num_collision_table += 1

                '''
                print('predicted distance : ', d_pred)
                print('real distance : ', d)
                print('current robot state : ', self.robot_state[:2])

                print('MPC current state : ', Xout[:2, 0])
                print('real next state : ', next_robot_state[:2])
                print('predicted next state : ', Xout[:2, 1])
                '''
                #-------- fixed
                #obs_cost = 50 / obs_weight
                #navi_part = 0
                #break
                #-------- incremental
                obs_cost += self._obs_reward / obs_weight
                #obs_cost += obs_weight * np.exp(-1e-5 / ((d - r) ** 2.))

            else:
                obs_cost += 0.
     
        col_part = -obs_weight * obs_cost

        reward = navi_part + col_part
        # -----------------------------------------------------------------------------------

        # ------------------------------ One-step Simulation --------------------------------
        #if not num_collision_table:
        if not num_collision_table + num_collision_human:
            self.robot_state = next_robot_state

        
        self._counter += 1
        ob = self._get_obs()
        # -----------------------------------------------------------------------------------

        # update distance between robot & goal
        self.goal_distance = ((self.robot_state[:2] - self._goal) ** 2).sum()

        if self.goal_distance <= 4. * self.lr**2:
            # if the next position of the robot lands on the goal point, return 1
            reach = 1
            reward += self._goal_reward
        else:
            reach = 0

        if num_collision_human + num_collision_table > 0:
            self.collision_history.append(True)
        else:
            self.collision_history.append(False)

        done = False

        return ob, reward, done, action, 0, (num_collision_human, num_collision_table, reach, navi_part, col_part), {}

    @property
    def collision_event(self) -> bool:
        # user-defined event for GP-MPC activation
        # If an event occur, then turn off GP-MPC completely.
        alert_radius = 2 * (self.human_radii + self.lr + self.safe_rad)
        human_centers = np.reshape(self.restaurant.human_vector, newshape=(self.num_people, 2))
        obstacle_distance_vector = np.sum((self.robot_state[0:2] - human_centers) ** 2, axis=1) ** .5
        if np.any(obstacle_distance_vector <= alert_radius):
            return True
        else:
            return False

    @property
    def reached(self) -> bool:
        return True if self.goal_distance <= 4. * self.lr**2 else False

    @property
    def goal_event(self) -> bool:
        # user-defined event for GP-MPC activation
        # If an event occur, then call GP-MPC with high probability(mostly 1).
        if self.goal_distance <= self.goal_tresh ** 2:
            return True
        else:
            return False

    @property
    def landed(self) -> bool:
        # check whether robot successfully lands on the goal point
        # triggers termination of control if true so that we save computational resource
        return True if self.goal_distance <= self.lr ** 2 else False

    @property
    def cost_weight(self):
        dist_to_goal = np.sqrt(self.goal_distance)        
        # Q = np.array([1., 1.]) * (1 + np.log(1+self.max_dist-dist_to_goal))
        Q = np.array([1., 1.])
        R = np.array([.2, .2])
        Qf = 20. * Q
        return Q, R, Qf

    def render(self, mode='human', pred=None):
        # TODO : use Gazebo for rendering

        if self._counter == self.ep_len - 1:
            plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'
            # writer = writers['ffmpeg'](fps=5)
            fig = plt.gcf()
            ax = fig.gca()
            x = []
            y = []
            line, = ax.plot(x, y,
                            label='trajectory',
                            linestyle='dotted',
                            marker='o',
                            color='tab:purple')
            line2, = ax.plot(x, y, 'k-')
            line_pred = []
            for i in range(self.num_people):
                line3, = ax.plot(x, y, 'ro')
                line_pred.append(line3)
            
            line_traj = []
            for i in range(self.num_people):
                line4, = ax.plot(x, y, 'g.')
                line_traj.append(line4)
            for o in self.restaurant.Tables:
                obstacle = plt.Circle(o.center,
                                      o.radii,
                                      color='tab:blue',
                                      )
                ax.add_artist(obstacle)
            image_path = path.join(path.dirname(__file__), "assets/robot.png")
            img = OffsetImage(plt.imread(image_path), zoom=.05)

            ax.text(x=self.restaurant.door1[0], y=self.restaurant.door1[1], s='door1')
            ax.text(x=self.restaurant.door2[0], y=self.restaurant.door2[1], s='door2')
            ax.text(x=self.restaurant.kitchen[0], y=self.restaurant.kitchen[1], s='kitchen')
            ax.text(x=self.restaurant.toilet[0], y= self.restaurant.toilet[1], s='toilet')
            ax.scatter([self.restaurant.door1[0], self.restaurant.door2[0], self.restaurant.kitchen[0], self.restaurant.toilet[0]],
                       [self.restaurant.door1[1], self.restaurant.door2[1], self.restaurant.kitchen[1], self.restaurant.toilet[1]],
                       marker='x', color='black')

            ax.add_artist(plt.Circle((self._goal[0], self._goal[1]), self.goal_tresh, color='tab:red', alpha=0.2))
            timestep = ax.text(-5.5, -5.7, '', fontsize=15)
            def func(t):
                if self.prev_robot is not None:
                    self.prev_robot.remove()
                    self.heading_angle_indicator.remove()
                for person_fig in list(self.prev_people):
                    person_fig.remove()
                    self.prev_people.remove(person_fig)
                x_traj = self._buffer[:t, 0]
                y_traj = self._buffer[:t, 1]
                xpos = self._buffer[t, 0]
                ypos = self._buffer[t, 1]

                timestep.set_text('t = {}'.format(t))

                # robot rendering
                if self.collision_history[t]:
                    color = 'tab:gray'
                else:
                    color = 'tab:purple'

                self.prev_robot = ax.add_artist(plt.Circle((xpos, ypos), self.lr, color=color))

                heading_angle = self._buffer[t, 2]
                dx = self.L * np.cos(heading_angle)
                dy = self.L * np.sin(heading_angle)
                self.heading_angle_indicator = ax.arrow(xpos, ypos,
                                                        dx, dy,
                                                        head_width=0.05, head_length=0.1, fc=color, ec=color)
                # Estimated obstacle position
                if pred is not None:
                    obs_dynamic = pred[t]
                    
                
                for i in range(self.num_people):
                    # human object rendering
                    px = self.buffer_for_people[t, 2 * i]
                    py = self.buffer_for_people[t, 2 * i + 1]
                    p_fig = ax.add_artist(plt.Circle((px, py), self.human_radii, color='tab:red'))
                    self.prev_people.append(p_fig)
                    px_ = self.buffer_for_people[t:, 2 * i]
                    py_ = self.buffer_for_people[t:, 2 * i + 1]
                    line_traj[i].set_data(px_, py_)

                    
                    if pred is not None:
                        pred_x, pred_y = [], []
                        pred_horizon = obs_dynamic.shape[0]
                        for j in range(pred_horizon):
                            pred_x.append(obs_dynamic[j,i,0])
                            pred_y.append(obs_dynamic[j,i,1])
                        line_pred[i].set_data(pred_x, pred_y)

                
                line.set_data(x_traj, y_traj)
                line2.set_data(self.Xout_buffer[t][0, :], self.Xout_buffer[t][1, :])

            '''
            # Save trajectory to render
            traj, scaler = {'x_pos': [], 'y_pos': []}, 5
            traj['x_pos'] = scaler * self._buffer[:-1, 0]
            traj['y_pos'] = scaler * self._buffer[:-1, 1]
            traj['heading'] = self._buffer[:-1,2]
            for i in range(self.num_people):
                traj['x_obstacle_%d'%(i)] = scaler * self.buffer_for_people[:-1, 2 * i]
                traj['y_obstacle_%d'%(i)] = scaler * self.buffer_for_people[:-1, 2 * i + 1]
                traj['x_heading_obstacle_%d'%(i)] = np.diff(self.buffer_for_people[:, 2 * i])
                traj['y_heading_obstacle_%d'%(i)] = np.diff(self.buffer_for_people[:, 2 * i + 1])
                normalizer = np.array(traj['x_heading_obstacle_%d'%(i)]) ** 2 + np.array(traj['y_heading_obstacle_%d'%(i)]) ** 2
                normalizer = np.sqrt(normalizer)
                traj['x_heading_obstacle_%d'%(i)] = traj['x_heading_obstacle_%d'%(i)] / normalizer
                traj['y_heading_obstacle_%d'%(i)] = traj['y_heading_obstacle_%d'%(i)] / normalizer
            for i, table in enumerate(self.restaurant.Tables):
                traj['x_table_%d'%(i)] = scaler * table.center[0]
                traj['y_table_%d'%(i)] = scaler * table.center[1]
            traj = pd.DataFrame(traj)
            traj.to_csv('traj_{}.csv'.format(time.strftime("%m%d-%H%M%S")))
            '''
            # Animate
            ani = FuncAnimation(fig=fig, func=func, frames=self.ep_len - 1)
            plt.grid()

            plt.scatter(self._buffer[0, 0], self._buffer[0, 1])
            plt.annotate('starting position', tuple(self._buffer[0, :2]))
            plt.scatter(self._goal[0], self._goal[1], color='tab:red', marker='x')
            plt.annotate('goal position', tuple(self._goal))
            plt.legend()
            plt.xlim(-5., 5.)
            plt.ylim(-5., 5.)
            fig.tight_layout()
            #plt.show(block=False)
            filename = 'vid_{}.gif'.format(time.strftime("%m%d-%H%M%S"))
            # if not self.use_MPC:
            ani.save(filename, writer='pillow', fps=5, dpi=100)
            # ani.save('vid_{}.mp4'.format(time.strftime("%m%d-%H%M%S")), writer=writer)

            #plt.pause(10)
            plt.close('all')
            return

