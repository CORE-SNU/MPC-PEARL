MPC-PEARL
====================================================

This repository includes an official python implementation of **MPC-PEARL** algorithm that is presented in **[Infusing Model Predictive Control into Meta Reinforcement Learning for Mobile Robots in Dynamic Environments][paper_link]**


## 1. Requirements
Our implementation is based on official repository of [PEARL][PEARL], and followings must be installed to run our implementation:
- **[PyTorch][Pytorch]**
- **[Gym][Gym]**
- **[GPy][GPy]**
- **[pickle5][pickle5]**
- **[FORCESPRO][FORCESPRO]**

To install FORCESPRO, you first need to get license. If you are currently working as a researcher you can request academic license. For detailed installation guide refer to [official manual][FORCESPRO_manual].

In addition, to run our result analysis tool you need to install additional packages such as [pandas][pandas]. For convenience, we summarized our test environment into `result_eval.yml` so that result can be easily reproduced.

Our code is tested on both Ubuntu 18.04 and Ubuntu 20.04.


## 2. Quick Start
First, clone our repository with:
```
$ git clone https://github.com/CORE-SNU/MPC-PEARL.git && cd mpc_pearl
```
As mentioned above, you can easily import requirements if you are working on conda environment:
```
$ conda env create -f mpc_pearl.yml && conda activate mpc_pearl
```
We provide our best test result along with network weights under `./output/Navigation_MPC/eps08`. You can visualize trained agent navigating in the modeled restaurant with:
```
$ python test_navigation.py --video --num_trajs=3
```
Navigation video over 25 distinct tasks will be generated under base directory as `gif` format along with summary of performance metric in `./results_total_[TEST_TIME].csv`.


## 3. Solve/Evaluate
### 3.1 Run Experiment
With our default setting, you can run experiment as follows:
```
$ python launch_experiment.py ./configs/Navigation_MPC.json
```
You can modify hyperparameters of PEARL from `./configs/Navigation_MPC.json` and `./confings/default.py`. For details of each component refer to official implementation of [PEARL][PEARL].

Additional parameters introduced in our algorithm should be modified manually in `./rlkit/envs/navi_toy.py`.
```python
# ------------------------------ Hyperparameter setup ---------------------------------
# Main hyperparameters
self._obs_reward = 20.
self._goal_reward = 10.
self.eps = .8
         
# Initial and goal state
self._goal = np.array([3.5, -3.5])
self.init_state = np.array([-4.5, 4.5, -np.pi / 8.])
# --------------------------------------------------------------------------------------
```
- `self._obs_reward` : collision penalty when the agent collides with static or dynamic obstacles
- `self._goal_reward` : reaching bonus when the agent is nearby goal
We are motivated to further aggregate these into `.json` file, and will be updated very soon.

In addition, vanilla PEARL algorithm can be simply trained by using `./configs/Navivation_WO_MPC.json`

### 3.2 Experiment Summary
During experiment, summary of each training epoch and network weights will be saved under `./output/Navigation_MPC/[EXP_START_TIME]` directory. To see training curve for various peformance metrics, run the following:
```
$ cd output && python plot_summary.py ./Navigation_MPC/[EXP_WANT_TO_SEE]
```

You can append more results and you will see the plot of mulitple experiments, such as:
```
$ cd output && python plot_summary.py ./Navigation_MPC/[EXP_WANT_TO_SEE_1], ./Navigation_MPC/[EXP_WANT_TO_SEE_2], ...
```

Plots will contain following performance metrics:
- Total return : undiscounted version
- Arrival time : required time to travel (80 if goal is not reached)
- Navigation reward : distance-based reward component
- Collisions : number of collisions

### 3.3 Evaluation and Visualization
To see how trained agent navigates among dynamic environment, run following line:
```
$ python test_navigation.py ./configs/Navigation_MPC.json ./output/Navigation_MPC/[EXP_WANT_TO_VISUALIZE]
```
Note that configuration file should match between trainig and testing. 
Navigation video and performance metric will be saved in `.gif` foramt and `./results_total_[TEST_TIME].csv` respectively.

To test trained agent on out-of-distribution task, go to `./out_of_distribution_test` and run above line again.
Make sure to copy trained weights under `./out_of_distribution_test/output/Navigation_MPC`


## 4. Build New Tasks

Our environment runs with pre-computed path of each dynamic obstacles. 
You can re-generated these scenarios based on our setups by deleting `U.npy` and `X.npy` under `./scenarios` and `./scenarios_out_of_distribution`.

To promote faster traing, we trained GPR offline and thus GPR should be re-trained with following command:
```
$ python gen_GP.py ./configs/Navigation_MPC.json
```

Now it is ready to run algorithm on new tasks.


[paper_link]: add_arxiv_address
[Pytorch]: https://pytorch.org/
[Gym]: https://github.com/openai/gym
[GPy]: https://github.com/SheffieldML/GPy
[pickle5]: https://pypi.org/project/pickle5/
[FORCESPRO]: https://www.embotech.com/products/forcespro/overview/
[FORCESPRO_manual]: https://forces.embotech.com/Documentation/
[pandas]: https://pandas.pydata.org/docs/getting_started/install.html
[PEARL]: https://github.com/katerakelly/oyster