import numpy as np
from typing import List
from rlkit.envs.utils.obstacles import Obstacle, CircularObstacle, Human, Table, intersect
import os
from itertools import permutations
from tqdm import tqdm


class Restaurant:
    def __init__(self,
                 num_tasks,
                 num_tables,
                 num_people,
                 table_radii,
                 human_radii,
                 ep_len=200,
                 ):

        self.dt = 0.05   # sampling interval
        # number of tasks to generate
        # to reduce the computational burden, we generate multiple scenarios in advance, and sample tasks from these
        self.num_tasks = num_tasks

        self.ep_len = ep_len
        # restaurant map configuration
        # here we assume that configuration space is simply a 2-dim cube [-1, 1]^2
        self.upper_bound = np.array([5., 5.])
        self.lower_bound = np.array([-5., -5.])

        self.kitchen = np.array([-4.5, 3.])
        self.door1 = np.array([4.5, 0.5])

        self.door2 = np.array([-2., 4.5])
        self.toilet = np.array([4., 3.5])

        # ------------------------------ Tables Setup ------------------------------
        self.Tables = []
        self.num_tables = num_tables
        self.table_radii = table_radii

        possible_pos = [
            5*np.array([-0.5, 0.5]),
            5*np.array([0., 0.5]),
            5*np.array([0.5, 0.5]),
            5*np.array([-0.5, 0.]),
            5*np.array([0., 0.]),
            5*np.array([0.5, 0.]),
            5*np.array([-0.5, -0.5]),
            5*np.array([0., -0.5]),
            5*np.array([0.5, -0.5])
        ]
        #############
        # 0 # 1 # 2 #
        #############
        # 3 # 4 # 5 #
        #############
        # 6 # 7 # 8 #
        #############
        self.table_configs = [
            [possible_pos[0], possible_pos[2], possible_pos[4], possible_pos[5], possible_pos[7]],
            [possible_pos[0], possible_pos[4], possible_pos[5], possible_pos[6], possible_pos[8]],
            [possible_pos[1], possible_pos[3], possible_pos[4], possible_pos[5], possible_pos[8]],
            [possible_pos[1], possible_pos[2], possible_pos[3], possible_pos[4], possible_pos[7]],
            [possible_pos[2], possible_pos[3], possible_pos[4], possible_pos[6], possible_pos[7]]
        ]
        self.num_configs = len(self.table_configs)

        self.tables_in_task = [
            [Table(center=center, radii=table_radii) for center in centers]
            for centers in self.table_configs]
        self.Tables = []
        # --------------------------------------------------------------------------

        # ------------------------------ People Setup ------------------------------

        self.num_people = num_people
        self.human_radii = human_radii

        # self.cycle_radii = cycle_radii      # shape = (number of tasks to be generated, number of people)
        # self.cycle_vel = cycle_vel          # same shape as above

        # self.people_in_task = [[Human(center=None, radii=self.human_radii)] * self.num_people] * self.num_tasks
        # Caution! Don't use multiplication operator as above, since it does not behave as you expect...
        self.people_in_task = [
            [Human(center=None, radii=self.human_radii) for _ in range(self.num_people)]
            for _ in range(self.num_tasks)]
        # offline computation of the trajectories of the moving obstacles
        # this is crucial for computationally light simulation
        # TODO : more complex & interesting obstacle motion
        self.generate_scenarios()

        self.People = []        # placeholder for human objects
        # --------------------------------------------------------------------------
        self.out_of_distribution_configs = [
            [np.array([center[0]-(0.5+0.5*np.random.rand()), center[1]]) for center in centers]
            for centers in self.table_configs
        ]
        self.tables_in_task_out_of_distribution = [
            [Table(center=center, radii=table_radii) for center in centers]
            for centers in self.out_of_distribution_configs]
        self.num_tasks_out_of_distribution = 25
        self.people_in_task_out_of_distribution = [
            [Human(center=None, radii=self.human_radii) for _ in range(self.num_people)]
            for _ in range(self.num_tasks_out_of_distribution)]
        self.generate_out_of_distribution_scenarios()
        return

    def generate_scenarios(self):
        # toilet -> door
        """
        method for pre-computation of dynamic obstacle trajectories
        strongly recommended for complex motion simulation to generate all tasks in advance
        """
        # print('generating {} scenarios offline...'.format(self.num_tasks))
        os.makedirs('scenarios', exist_ok=True)
        if os.path.exists('./scenarios/X.npy') and os.path.exists('./scenarios/U.npy'):
            # load precomputed trajectories
            # TODO: set this part as a separate module
            load_existing_data = True
            X_total = np.load('./scenarios/X.npy')
            U_total = np.load('./scenarios/U.npy')
        else:
            from rlkit.envs.planner.Human_Planning import plan
            X_total = np.zeros((self.num_tasks, 2, self.ep_len + 1, self.num_people))
            U_total = np.zeros((self.num_tasks, 2, self.ep_len, self.num_people))
            load_existing_data = False

        idx = 0
        while idx < self.num_tasks:
            if not load_existing_data and idx % 10 == 0:
                print('{} scenarios generated'.format(idx + 1))
            for perm in permutations(range(self.num_tables)):
                # if (# of dynamic obstacles) = N, then there are N! configurations in total
                # if larger # of tasks need to be generated, iterate over these configurations repeatedly
                if idx >= self.num_tasks:
                    break
                people = self.people_in_task[idx]
                if load_existing_data:
                    X = X_total[idx]
                    U = U_total[idx]
                else:
                    # 1st-time generation of trajectories
                    r = self.table_radii + self.human_radii * np.random.rand(self.num_tables)
                    th = 2. * np.pi * np.random.rand(self.num_tables)
                    th[3] = np.pi * (0.25 + np.random.rand())
                    polar = r * np.vstack([np.cos(th), np.sin(th)])     # shape = (2, # of tables)
                    # table_pos = np.array(self.init_centers) + polar.T
                    table_pos = np.array(self.table_configs[idx % self.num_configs])
                    perturbed_pos = table_pos + polar.T
                    spawn_points = np.array([self.door1,
                                             self.toilet,
                                             self.kitchen,
                                             perturbed_pos[perm[3]],
                                             perturbed_pos[perm[4]],
                                             self.door2
                                             ]).T
                    goal_points = np.array([perturbed_pos[perm[0]],
                                            perturbed_pos[perm[1]],
                                            perturbed_pos[perm[2]],
                                            self.door1,
                                            self.kitchen,
                                            self.toilet
                                            ])

                    # velocity adjustment using travel distance
                    distances = [((spawn_points[:, i] - goal_points[i]) ** 2.).sum() ** .5 for i in range(self.num_people)]
                    u_lb = []
                    u_ub = []
                    for i in range(self.num_people):
                        lim = max(distances[i] / (np.sqrt(2.) * self.ep_len), 0.15)
                        u_ub += [lim, lim]
                        u_lb += [-lim, -lim]
                    # MPC-based motion generation
                    X, U = plan(table_pos, spawn_points, goal_points, self.ep_len, u_lb, u_ub)
                    X_total[idx] = X
                    U_total[idx] = U
                    print("Task {} Done!".format(idx))
                for i in range(self.num_people):
                    # set offline trajectory to every human object of a task
                    people[i].set_path(X[:, :, i])
                idx += 1

        if not load_existing_data:
            # first time generation
            np.save('./scenarios/X.npy', X_total)
            np.save('./scenarios/U.npy', U_total)

    def reset(self):
        # bring back people to their spawn point
        for human in self.People:
            human.init()
        # self.human_center_list = [[-0.9, 0.], [0., 0.9]]

    def reset_task(self, idx):
        self.People = self.people_in_task[idx]
        self.Tables = self.tables_in_task[idx % 5]

    def reset_task_out_of_distribution(self, idx):
        self.People = self.people_in_task_out_of_distribution[idx - 180]
        self.Tables = self.tables_in_task_out_of_distribution[idx % len(self.tables_in_task_out_of_distribution)]

    def sim(self):
        # simulate motions of the dynamic obstacles
        for human in self.People:
            human.sim()
        #     self.human_center_list[i] = human.center      # update position of each guest
        return

    @property
    def obstacle_list(self) -> List[CircularObstacle]:
        # list of every existing obstacles
        return (self.Tables+self.People)[:]

    @property
    def table_list(self) -> List[CircularObstacle]:
        # list of every existing obstacles
        return self.Tables[:]

    @property
    def human_list(self) -> List[CircularObstacle]:
        # list of every existing obstacles
        return self.People[:]
    
    @property
    def human_vector(self) -> np.ndarray:
        # return the real-time positions of all people as a flat numpy array
        human_center_list = [human.center for human in self.People]
        return np.reshape(np.array(human_center_list), self.num_people * 2)

    @property
    def human_vel_vector(self) -> np.ndarray:
        vel_array = np.array([human.vel for human in self.People])
        return np.reshape(vel_array, self.num_people * 2)


    @property
    def table_vector(self) -> np.ndarray:
        # return the center positions of all tables as a flat numpy array
        table_center_list = [table.center for table in self.Tables]
        return np.reshape(np.array(table_center_list), self.num_tables * 2)

    def generate_out_of_distribution_scenarios(self):
        # toilet -> door
        """
        method for pre-computation of dynamic obstacle trajectories
        strongly recommended for complex motion simulation to generate all tasks in advance
        """
        # print('generating {} scenarios offline...'.format(self.num_tasks))
        os.makedirs('scenarios_out_of_distribution', exist_ok=True)
        if os.path.exists('./scenarios_out_of_distribution/X.npy') and os.path.exists('./scenarios_out_of_distribution/U.npy'):
            # load precomputed trajectories
            # TODO: set this part as a separate module
            load_existing_data = True
            X_total = np.load('./scenarios_out_of_distribution/X.npy')
            U_total = np.load('./scenarios_out_of_distribution/U.npy')
        else:
            from rlkit.envs.planner.Human_Planning import plan
            X_total = np.zeros((self.num_tasks_out_of_distribution, 2, self.ep_len + 1, self.num_people))
            U_total = np.zeros((self.num_tasks_out_of_distribution, 2, self.ep_len, self.num_people))
            load_existing_data = False

        idx = 0
        while idx < self.num_tasks:
            if not load_existing_data and idx % 10 == 0:
                print('{} scenarios generated'.format(idx + 1))
            for perm in permutations(range(self.num_tables)):
                # if (# of dynamic obstacles) = N, then there are N! configurations in total
                # if larger # of tasks need to be generated, iterate over these configurations repeatedly
                if idx < 180:
                    idx += 1
                    break

                if idx >= self.num_tasks:
                    break
                people = self.people_in_task_out_of_distribution[idx - 180]
                if load_existing_data:
                    X = X_total[idx - 180]
                    U = U_total[idx - 180]
                else:
                    # 1st-time generation of trajectories
                    r = self.table_radii + self.human_radii * np.random.rand(self.num_tables)
                    th = 2. * np.pi * np.random.rand(self.num_tables)
                    th[3] = np.pi * (0.25 + np.random.rand())
                    polar = r * np.vstack([np.cos(th), np.sin(th)])     # shape = (2, # of tables)
                    # table_pos = np.array(self.init_centers) + polar.T
                    table_pos = np.array(self.out_of_distribution_configs[idx % 5])
                    perturbed_pos = table_pos + polar.T
                    spawn_points = np.array([self.door1,
                                             self.toilet,
                                             self.kitchen,
                                             perturbed_pos[perm[3]],
                                             perturbed_pos[perm[4]],
                                             self.door2
                                             ]).T
                    goal_points = np.array([perturbed_pos[perm[0]],
                                            perturbed_pos[perm[1]],
                                            perturbed_pos[perm[2]],
                                            self.door1,
                                            self.kitchen,
                                            self.toilet
                                            ])

                    # velocity adjustment using travel distance
                    distances = [((spawn_points[:, i] - goal_points[i]) ** 2.).sum() ** .5 for i in range(self.num_people)]
                    u_lb = []
                    u_ub = []
                    for i in range(self.num_people):
                        lim = max(distances[i] / (np.sqrt(2.) * self.ep_len), 0.15)
                        u_ub += [lim, lim]
                        u_lb += [-lim, -lim]
                    # MPC-based motion generation
                    X, U = plan(table_pos, spawn_points, goal_points, self.ep_len, u_lb, u_ub)
                    X_total[idx - 180] = X
                    U_total[idx - 180] = U
                    print("Task {} Done!".format(idx))
                for i in range(self.num_people):
                    # set offline trajectory to every human object of a task
                    people[i].set_path(X[:, :, i])
                idx += 1

        if not load_existing_data:
            # first time generation
            np.save('./scenarios_out_of_distribution/X.npy', X_total)
            np.save('./scenarios_out_of_distribution/U.npy', U_total)

