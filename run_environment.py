import os
import sys
import time
import argparse
import numpy as np

from pynput import keyboard
from stable_baselines3 import TD3, SAC
from stable_baselines3.common.env_checker import check_env

from inchworm import InchwormEnv


def train_with_sb3_agent(
    model_name="inchworm_td3",
    algorithm="td3",
    total_timesteps=30000,
    learning_rate=0.0003,
    render=False,
):
    """
    Trains the provided saved agent (or initializes a new agent if the provided model name doesn't exist)
    within the Inchworm environment for `total_timesteps` time steps. Saves the trained model to a test
    directory once it finishes, or if it receives a KeyboardInterrupt.

    Parameters
    ----------
    - `model_name`: name of the zip file (minus the .zip extension) contained in `model_dir` that represents a saved pretrained agent
    - `algorithm`: the algorithm to use for training. Must be one of "td3" or "sac"
    - `total_timesteps`: the number of time steps to train the agent for
    - `learning_rate`: the learning rate to apply to the training
    - `render`: whether to render the simulation after the agent is done training
    """
    model_path = f"test_models/{model_name}.zip"

    algorithm_class = {"td3": TD3, "sac": SAC}.get(algorithm.lower())
    assert algorithm_class is not None, f"Invalid algorithm: {algorithm}"

    env = InchwormEnv(render_mode=("human" if render else "rgb_array"))
    check_env(
        env
    )  # Make sure our env is compatible with the interface that stable-baselines3 agents expect

    try:
        model = algorithm_class.load(model_path, env)
        print("Continuing training of saved model")
    except FileNotFoundError:
        print("No saved model found, training new model")
        model = algorithm_class(
            "MlpPolicy", env, verbose=1, learning_rate=learning_rate
        )

    model.set_random_seed(time.time_ns() % 2**32)  # Set random seed to current time

    try:
        model.learn(total_timesteps, progress_bar=True)
    except KeyboardInterrupt:
        print("Interrupted by user, saving model")
    finally:
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        model.save(model_path)

    if render:
        input("Done. Press enter to view trained agent")
        run_simulation_with_sb3_agent(model_name=model_name, model_dir="test_models")
    else:
        print("Done.")

    env.close()


def run_simulation_with_sb3_agent(
    model_name="inchworm_td3",
    model_dir="saved_models",
    algorithm="td3",
    old_model=False,
    evals=False,
):
    """
    Runs the Inchworm environment using a provided saved agent, and applies the agent's actions
    to the environment without having the agent learn. For demonstration/testing purposes.

    Parameters
    ----------
    - `model_name`: name of the zip file (minus the .zip extension) contained in `model_dir` that represents a saved pretrained agent
    - `model_dir`: directory path where the model is stored (with no trailing slash)
    - `algorithm`: the algorithm to use for training. Must be one of "td3" or "sac"
    - `old_model`: whether the model was trained with the old version of the Inchworm environment
    - `evals`: whether to calculate evaluation metrics while running the agent
    """
    saved_model_path = f"{model_dir}/{model_name}.zip"

    algorithm_class = {"td3": TD3, "sac": SAC}.get(algorithm.lower())
    assert algorithm_class is not None, f"Invalid algorithm: {algorithm}"

    env = InchwormEnv(render_mode="human", old_model=old_model, evals=evals)
    check_env(
        env
    )  # Make sure our env is compatible with the interface that stable-baselines3 agents expect

    try:
        model = algorithm_class.load(saved_model_path, env)
        print("Using specified model")
    except FileNotFoundError:
        print("Specified model not found")
        sys.exit(1)

    model.set_random_seed(time.time_ns() % 2**32)  # Set random seed to current time

    vec_env = model.get_env()
    assert vec_env is not None
    obs = vec_env.reset()

    while True:
        try:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = vec_env.step(action)
            vec_env.render("human")
        except KeyboardInterrupt:
            if evals:
                InchwormEnv.print_evals(info[0]["evals"], "Session Evaluation")
            break


def run_simulation_random():
    """
    Runs the Inchworm environment while providing random actions from the action space
    at each time step
    """
    env = InchwormEnv(render_mode="human")

    # Must reset the env before making the first call to step()
    observation, info = env.reset()

    for _ in range(1000):
        try:
            # Select a random action from the sample space
            action = env.action_space.sample()

            # Apply that action to the environment, store the resulting data
            observation, reward, terminated, truncated, info = env.step(action)

            # End current episode if necessary
            if terminated or truncated:
                observation, info = env.reset()
        except KeyboardInterrupt:
            break


def run_simulation_control():
    """
    Runs the Inchworm environment while allowing the user to control the inchworm themselves
    via their keyboard.

    NOTE: On Unix-like machines, this Python script must be run with `sudo` in order
    for the key press detection library to function

    Controls
    --------
    - 'u'/'j': rotate the left joint clockwise and counterclockwise
    - 'i'/'k': rotate the middle joint clockwise and counterclockwise
    - 'o'/'l': rotate the right joint clockwise and counterclockwise
    - '[': enable the left adhesion gripper
    - ']': enable the right adhesion gripper
    """
    env = InchwormEnv(render_mode="human")

    # Must reset the env before making the first call to step()
    observation, info = env.reset()

    keys: set[keyboard.Key | keyboard.KeyCode | None] = set()
    listener = keyboard.Listener(
        on_press=lambda key: keys.add(key), on_release=lambda key: keys.remove(key)
    )
    listener.start()

    while True:
        try:
            # Determine action
            action = get_action(keys)

            # Break on 'q' press
            if action is None:
                listener.stop()
                break

            # Apply that action to the environment, store the resulting data
            observation, reward, terminated, truncated, info = env.step(action)

            # End current episode if necessary
            if terminated or truncated:
                observation, info = env.reset()
        except KeyboardInterrupt:
            listener.stop()
            break


def get_action(keys: set[keyboard.Key | keyboard.KeyCode | None]):
    def is_pressed(key: str) -> bool:
        return any(k == keyboard.KeyCode.from_char(key) for k in keys)

    action: list[int] = []

    if is_pressed("q"):
        return None

    action.append(1 if is_pressed("j") else -1 if is_pressed("u") else 0)
    action.append(1 if is_pressed("i") else -1 if is_pressed("k") else 0)
    action.append(1 if is_pressed("l") else -1 if is_pressed("o") else 0)
    action.append(1 if is_pressed("[") else -1)
    action.append(1 if is_pressed("]") else -1)

    return np.array(action)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""Run or train an agent to control an inchworm robot"""
    )
    group1 = parser.add_argument_group("Functional arguments (mutually exclusive)")
    group1e = group1.add_mutually_exclusive_group(required=True)
    group1e.add_argument(
        "-t",
        "--train",
        action="store_true",
        help="train a new/existing model in test_models/ with the TD3 algorithm",
    )
    group1e.add_argument(
        "-r",
        "--run",
        action="store_true",
        help="run a model with the TD3 algorithm",
    )
    group1e.add_argument(
        "-R",
        "--random",
        action="store_true",
        help="run the environment with random actions",
    )
    group1e.add_argument(
        "-c",
        "--control",
        action="store_true",
        help="run the environment with user control",
    )
    group2 = parser.add_argument_group("Training and running arguments")
    group2.add_argument(
        "-m",
        "--model-name",
        type=str,
        help="name of the model to run (minus the .zip extension)",
    )
    group2.add_argument(
        "-a",
        "--algorithm",
        type=str,
        default="td3",
        help="algorithm to use for training/running model, either sac or td3 (default: td3)",
    )
    group3 = parser.add_argument_group("Running arguments")
    group3.add_argument(
        "-s",
        "--saved-dir",
        action="store_true",
        help="whether the model will be/is in the saved_models/ directory (otherwise test_models/)",
    )
    group3.add_argument(
        "-e",
        "--eval",
        action="store_true",
        help="whether to print out evaluation data while running the simulation",
    )
    group3.add_argument(
        "-o",
        "--old-model",
        action="store_true",
        help="whether the model was trained with the old version of the Inchworm environment",
    )
    group4 = parser.add_argument_group("Training arguments")
    group4.add_argument(
        "-T",
        "--total-timesteps",
        type=int,
        default=1_000_000,
        help="total number of timesteps to train the model for (default: 1,000,000)",
    )
    group4.add_argument(
        "-l",
        "--learning-rate",
        type=float,
        default=0.0003,
        help="learning rate for training the model (default: 0.0003)",
    )
    args = parser.parse_args()

    if args.train:
        if args.model_name is None:
            parser.error("argument -t/--train requires -m/--model-name")
        if args.saved_dir:
            parser.error(
                "argument -t/--train cannot be used with -s/--saved-dir (cannot train a model in the saved_models/ directory)"
            )
        train_with_sb3_agent(
            model_name=args.model_name,
            algorithm=args.algorithm,
            total_timesteps=args.total_timesteps,
            learning_rate=args.learning_rate,
        )
    elif args.run:
        if args.model_name is None:
            parser.error("argument -r/--run requires -m/--model-name")
        run_simulation_with_sb3_agent(
            model_name=args.model_name,
            algorithm=args.algorithm,
            model_dir="saved_models" if args.saved_dir else "test_models",
            old_model=args.old_model,
            evals=args.eval,
        )
    elif args.random:
        run_simulation_random()
    elif args.control:
        run_simulation_control()
    else:
        parser.print_help()
        exit(0)
