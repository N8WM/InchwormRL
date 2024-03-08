# InchwormRL

By Nathan McCutchen
_Forked from [gwholland3/InchwormRL](https://github.com/gwholland3/InchwormRL) by Schuyler Fenton, Grant Holland, Nathan McCutchen, and Ishan Meher_

This project is rebooted for CSC 580 at Cal Poly, taught by Dr. Franz Kurfess

<img src="./assets/inchworm.gif" alt="Inchworm" style="width: 90%;">

## External Resources

We used the following external resources in our project.

MuJoCo physics engine: https://mujoco.readthedocs.io/en/stable/overview.html
Gymnasium RL framework: https://gymnasium.farama.org/
Stable Baselines3 RL algorithms: https://stable-baselines3.readthedocs.io/en/master/index.html

### Important Notice

At the time of writing this, there is a bug in the mujoco rendering file. If you encounter the following error, you need to navigate to `mujoco_rendering.py` in the gymnasium package on your local file system and replace the instance of `solver_iter` with `nsolver_iter` as the error suggests, and save your change.

This is the error:

```
AttributeError: 'mujoco._structs.MjData' object has no attribute 'solver_iter'. Did you mean: 'solver_niter'?
```

Since I use a virtual environment, the relative path to my `mujoco_rendering.py` file is `InchwormRL/.venv/lib/python3.10/site-packages/gymnasium/envs/mujoco/mujoco_rendering.py`.

## Setup

Follow these instructions to set up the codebase locally.

### 1. Clone the Repo

Run your favorite version of the git clone command on this repo. One version:

`git clone https://github.com/N8WM/InchwormRL.git`

### 2. Install Python

This code was developed and run on Python `3.10.10`, but most likely any version of Python `3.10` will do. Make sure you have an appropriate version installed locally.

### 3. Install Requirements

We recommend doing this in a fresh Python virtual environment. Cd into the repo and run:

`pip install -r requirements.txt`

## Interacting With the Project

**Working Model**

As of 4/7/24, there is one successful model trained with the SAC algorithm in the `saved_models` directory. You can run this model with the following command:

`python3 run_environment.py -rsem inchworm4.2_sac -a sac`

**Other Options**

Our whole project has a single entry point, `run_environment.py`. You can control the functionality via command-line arguments to Python script.

### Examples

Run an existing trained model (inchworm4.2_sac) and print evaluation data:

`python3 run_environment.py -rsem inchworm4.2_sac -a sac`

Run an existing trained model (inchworm2.1_td3) on the old Inchworm environment and print evaluation data:

`python3 run_environment.py -rsoem inchworm2.1_td3`

Train a new model (inchworm5.0_td3) with 10,000,000 timesteps:

`python3 run_environment.py -tm inchworm5.0_td3 -T 10000000`

```
usage: run_environment.py [-h] (-t | -r | -R | -c) [-m MODEL_NAME] [-a ALGORITHM] [-s] [-e] [-o]
                          [-T TOTAL_TIMESTEPS] [-l LEARNING_RATE]

Run or train an agent to control an inchworm robot

options:
  -h, --help            show this help message and exit

Functional arguments (mutually exclusive):
  -t, --train           train a new/existing model in test_models/ with the TD3 algorithm
  -r, --run             run a model with the TD3 algorithm
  -R, --random          run the environment with random actions
  -c, --control         run the environment with user control

Training and running arguments:
  -m MODEL_NAME, --model-name MODEL_NAME
                        name of the model to run (minus the .zip extension)
  -a ALGORITHM, --algorithm ALGORITHM
                        algorithm to use for training/running model, either sac or td3 (default: td3)

Running arguments:
  -s, --saved-dir       whether the model will be/is in the saved_models/ directory (otherwise test_models/)
  -e, --eval            whether to print out evaluation data while running the simulation
  -o, --old-model       whether the model was trained with the old version of the Inchworm environment

Training arguments:
  -T TOTAL_TIMESTEPS, --total-timesteps TOTAL_TIMESTEPS
                        total number of timesteps to train the model for (default: 1,000,000)
  -l LEARNING_RATE, --learning-rate LEARNING_RATE
                        learning rate for training the model (default: 0.0003)
```
