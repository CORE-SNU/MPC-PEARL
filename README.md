MPC-PEARL
====================================================

This repository includes an official python implementation of **MPC-PEARL** algorithm presented in **[Infusing Model Predictive Control into Meta-Reinforcement Learning
for Mobile Robots in Dynamic Environments][paper_link]**.



## 1. Requirements
Our implementation is based on the official repository of [PEARL][PEARL], and the following must be installed to run our implementation:
- **[PyTorch][Pytorch]**
- **[Gym][Gym]**
- **[GPy][GPy]**
- **[pickle5][pickle5]**
- **[FORCESPRO][FORCESPRO]**

For FORCESPRO, you first need to activate a FORCESPRO license, which can be done [here][FORCESPRO]. For detailed installation guide, please refer to [official manual][FORCESPRO_manual].

In order to analyze evaluation results, install additional packages such as [pandas][pandas]. For convenience, we have summarized our test environment into `mpc_pearl.yml` so that the results presented in our paper may be easily reproduced.

Our code has been successfully tested on both Ubuntu 18.04 and Ubuntu 20.04.



## 2. Quick Start
First, clone our repository with:
```
git clone https://github.com/CORE-SNU/MPC-PEARL.git && cd mpc_pearl
```
As mentioned above, you can easily import requirements if you are working on conda environment:
```
conda env create -f mpc_pearl.yml && conda activate mpc_pearl
```
We provide the network weights of best-performing agent under `./output/Navigation_MPC/eps02`. You can visualize the trained agent navigating in a simulated indoor environment with:
```
python test_navigation.py ./configs/Navigation_MPC ./output/Navigation_MPC/eps02 --video --num_trajs=3
```
The videos on 25 distinct tasks will be generated under the project root directory in `.gif` format, along with the summary of performance metrics in `./results_total_[TEST_TIME].csv`.



## 3. Solve/Evaluate
### 3.1 Run Experiment
With our default setting, you can run experiment as follows:
```
python launch_experiment.py ./configs/Navigation_MPC.json
```
You can modify the hyperparameters of PEARL from `./configs/Navigation_MPC.json` and `./confings/default.py`. For details of each component, please refer to the official implementation of [PEARL][PEARL].

Additional parameters introduced in our algorithm should be modified manually in `./rlkit/envs/navi_toy.py`.
```python
# ------------------------------ Hyperparameter setup ---------------------------------
# Main hyperparameters
self._obs_reward = 20.
self._goal_reward = 10.
self.eps = .2
         
# Initial and goal state
self._goal = np.array([3.5, -3.5])
self.init_state = np.array([-4.5, 4.5, -np.pi / 8.])
# --------------------------------------------------------------------------------------
```
- `self._obs_reward` : collision penalty when the agent collides with static or dynamic obstacles
- `self._goal_reward` : reaching bonus when the agent is nearby goal

As a baseline, vanilla PEARL algorithm may be simply trained by running `launch_experiment.py` with `./configs/Navivation_WO_MPC.json`.


### 3.2 Experiment Summary
When training, the summary of each training epoch and network weights will be saved under `./output/Navigation_MPC/[EXP_START_TIME]` directory. To see the learning curves for various peformance metrics, run the following:
```
cd output && python plot_summary.py ./Navigation_MPC/[EXP_WANT_TO_SEE]
```

You can also visualize the results of mulitple experiments by running
```
cd output && python plot_summary.py ./Navigation_MPC/[EXP_WANT_TO_SEE_1], ./Navigation_MPC/[EXP_WANT_TO_SEE_2], ...
```

Plots will contain following performance metrics:
- Total return : undiscounted version
- Arrival time : required time to travel (80 if goal is not reached)
- Navigation reward : distance-based reward component
- Collisions : number of collisions


### 3.3 Evaluation and Visualization
To see how the trained agent navigates among dynamic, cluttered environments, run the following line:
```
python test_navigation.py ./configs/Navigation_MPC.json ./output/Navigation_MPC/[EXP_WANT_TO_VISUALIZE]
```
Note that the configuration file given as an argument must be the same for both trainig and testing. The summary of the result appears in `./results_total_[TEST_TIME].csv`.

The following options can be given for `test_navigation.py`:
- `--video` : Navigation video will be saved in `.gif` foramt. Default = False
- `--mpc_only`: If true, the agent will only use MPC for navigation. Default = False
- `--num_trajs`: Number of adaptation steps to use. Default = 10



## 4. Build New Tasks

Our environment runs with pre-computed paths of each dynamic obstacles. 
You can re-generated these scenarios based on our setups by deleting `U.npy` and `X.npy` under `./scenarios`.

To promote faster traing, you may pretrain GPR offline. This may be done by running
```
python gen_GP.py ./configs/Navigation_MPC.json
```



## 5. UCY Dataset

We also demonstrate the proposed method in [**UCY**][UCY] sidwalk environment.
To use the environment, first unzip `ucy.zip` and move into `./ucy` which has the same structure as MPC-PEARL directory.

To run an experiment, first build the offline GP dataset by running
```
python gen_GP.py ./configs/UCY_MPC.json
```
Then, use the same code with another configuration file:
```
python launch_experiment.py ./configs/UCY_MPC.json
```



## 6. Troubleshooting

Most commonly reported problems we have found so far are follows:


### 6.1 Memory error when running `test_navigation.py` with `--video` option
```
OSError: [Errno 12] Cannot allocate memory
```
This is known to be an error from gym, whose solution can be found from [official repo][issue1]:
```
sudo bash -c "echo vm.overcommit_memory=1 >> /etc/sysctl.conf"

sudo sysctl -p
```


[paper_link]: https://arxiv.org/abs/2109.07120
[Pytorch]: https://pytorch.org/
[Gym]: https://github.com/openai/gym
[GPy]: https://github.com/SheffieldML/GPy
[pickle5]: https://pypi.org/project/pickle5/
[FORCESPRO]: https://www.embotech.com/products/forcespro/overview/
[FORCESPRO_manual]: https://forces.embotech.com/Documentation/
[pandas]: https://pandas.pydata.org/docs/getting_started/install.html
[PEARL]: https://github.com/katerakelly/oyster
[UCY]: https://github.com/Habiba-Amroune/ETH-UCY-Preprocessing
[issue1]: github.com/openai/gym/issues/110#issuecomment-220672405
