from os import path
from collections import deque

import numpy as np

from gymnasium import utils
from gymnasium.envs.mujoco.mujoco_env import MujocoEnv
from gymnasium.spaces import Box


DEFAULT_CAMERA_CONFIG = {
    "distance": 8.0,
}


class InchwormEnv(MujocoEnv, utils.EzPickle):
    """
    ## Description

    This environment is for CSC 480.

    The inchworm is a 2D robot consisting of four links attached in a line, with
    rotational joints between each link. The goal is to coordinate the four links
    to move in the forward (right) direction by applying torques on the three
    hinges connecting the links together and controlling the adhesion actuators
    on the two feet.

    ## Action Space
    The action space is a `Box(-1, 1, (5,), float32)`.
    An action represents the torques applied at the three hinge joints concatenated
    with the input applied to the two adhesion actuators.

    | Num | Action                                                            | Control Min | Control Max | Name (in corresponding XML file) | Joint    | Unit         |
    | --- | ----------------------------------------------------------------- | ----------- | ----------- | -------------------------------- | -------- | ------------ |
    | 0   | Torque applied on the rotor between the first and second links    | -1          | 1           | left_joint                       | hinge    | torque (N m) |
    | 1   | Torque applied on the rotor between the second and third links    | -1          | 1           | middle_joint                     | hinge    | torque (N m) |
    | 2   | Torque applied on the rotor between the third and fourth links    | -1          | 1           | right_joint                      | hinge    | torque (N m) |
    | 3   | Whether adhesion is activated on the left foot                    | -1          | 1           | left_foot                        | adhesion | force (N) |
    | 4   | Whether adhesion is activated on the right foot                   | -1          | 1           | right_foot                       | adhesion | force (N) |

    ## Observation Space

    Observations consist of positional values of different body parts of the inchworm,
    followed by the velocities of those individual parts (their derivatives) with all
    the positions ordered before all the velocities.

    By default, an observation is a `ndarray` with shape `(12,)`
    where the elements correspond to the following:

    | Num | Observation                                                  | Min    | Max    | Name (in corresponding XML file)       | Joint | Unit                     |
    |-----|--------------------------------------------------------------|--------|--------|----------------------------------------|-------|--------------------------|
    | 0   | y-orientation of the left foot                               | -Inf   | Inf    | hc_joint                               | hinge | angle (deg)              |
    | 1   | x-coordinate of the left foot                                | -Inf   | Inf    | hsx_joint                              | slide | position (m)             |
    | 2   | z-coordinate of the left foot                                | -Inf   | Inf    | hsz_joint                              | slide | position (m)             |
    | 3   | angle between the first and second segments                  | -Inf   | Inf    | left_joint                             | hinge | angle (deg)              |
    | 4   | angle between the second and third segments                  | -Inf   | Inf    | middle_joint                           | hinge | angle (deg)              |
    | 5   | angle between the third and fourth segments                  | -Inf   | Inf    | right_joint                            | hinge | angle (deg)              |
    | 6   | y-coordinate angular velocity of the left foot               | -Inf   | Inf    | hc_joint                               | hinge | angular velocity (deg/s) |
    | 7   | x-coordinate velocity of the left foot                       | -Inf   | Inf    | hsx_joint                              | slide | velocity (m/s)           |
    | 8   | z-coordinate velocity of the left foot                       | -Inf   | Inf    | hsz_joint                              | slide | velocity (m/s)           |
    | 9   | angular velocity of angle between first and second segments  | -Inf   | Inf    | left_joint                             | hinge | angular velocity (deg/s) |
    | 10  | angular velocity of angle between second and third segments  | -Inf   | Inf    | middle_joint                           | hinge | angular velocity (deg/s) |
    | 11  | angular velocity of angle between third and fourth segments  | -Inf   | Inf    | right_joint                            | hinge | angular velocity (deg/s) |

    The (x,y,z) coordinates are translational DOFs while the orientations are rotational
    DOFs expressed as quaternions. One can read more about free joints on the [Mujoco Documentation](https://mujoco.readthedocs.io/en/latest/XMLreference.html).

    ## Rewards
    The reward consists of four parts:

    - *healthy_reward*: Every timestep that the inchworm is healthy (see definition in section "Episode Termination"), it gets a reward of fixed value `healthy_reward`

    - *forward_reward*: A reward of moving forward which is measured as
    *(x-coordinate before action - x-coordinate after action)/dt*. *dt* is the time
    between actions and is dependent on the `frame_skip` parameter (default is 5),
    where the frametime is 0.01 - making the default *dt = 5 * 0.01 = 0.05*.
    This reward would be positive if the inchworm moves forward (in positive x direction).

    - *ctrl_cost*: A negative reward for penalising the inchworm if it takes actions for motors
    that are too large. It is measured as *`ctrl_cost_weight` * sum(action[:3]<sup>2</sup>)*
    where *`ctr_cost_weight`* is a parameter set for the control and has a default value of 0.5.

    - *ungrounded_cost*: A negative reward for penalising the inchworm if both its feet leave
    the ground. It is measured as *`ungrounded_cost_weight` * `grounded`* where *`ungrounded_cost_weight`*
    is a parameter set for the control and has a default value of 100. *`grounded`* indicates whether
    the inchworm is currently touching the ground or not, but will only begin returning False once
    the inchworm has contacted the ground for the first time, to prevent penalising the inchworm
    at the start of each episode (the inchworm spawns in the air).

    The total reward returned is ***reward*** *=* *healthy_reward + forward_reward - ctrl_cost - ungrounded_cost*.

    `info` will also contain the individual reward terms.

    ## Starting State
    All observations start in state
    (0.0, 0.0, 0.0, 0.0, 0.0, ..., 0.0) with a uniform noise in the range
    of [-`reset_noise_scale`, `reset_noise_scale`] added to the positional values and standard normal noise
    with mean 0 and standard deviation `reset_noise_scale` added to the velocity values for
    stochasticity. The initial orientation is designed to make the inchworm face forward.

    ## Episode End
    The inchworm is said to be unhealthy if any of the following happens:

    1. Any of the state space values is no longer finite

    If `terminate_when_unhealthy=True` is passed during construction (which is the default),
    the episode ends when any of the following happens:

    1. Truncation: The episode duration reaches 1000 timesteps
    2. Termination: The inchworm is unhealthy

    If `terminate_when_unhealthy=False` is passed, the episode is ended only when 1000 timesteps are exceeded.

    ## Arguments

    | Parameter                  | Type      | Default          | Description                   |
    |----------------------------|-----------|------------------|-------------------------------|
    | `xml_file`                 | **str**   | `"inchworm.xml"` | Path to a MuJoCo model |
    | `old_model`                | **bool**  | `False`          | If true, use the old version of the inchworm xml environment
    | `episode_length`           | **int**   | `1000`           | Number of timesteps per episode (before truncation) |
    | `evals`                    | **bool**  | `False`          | If true, calculate evaluation metrics on the episodes
    | `ctrl_cost_weight`         | **float** | `0.5`            | Weight for *ctrl_cost* term (see section on reward) |
    | `ungrounded_cost_weight`   | **float** | `100`            | Weight for *ungrounded_cost* term (see section on reward) |
    | `healthy_reward`           | **float** | `1`              | Constant reward given if the inchworm is "healthy" after timestep |
    | `terminate_when_unhealthy` | **bool**  | `True`           | If true, issue a done signal if the inchworm is deemed to be "unhealthy" |
    | `reset_noise_scale`        | **float** | `0.1`            | Scale of random perturbations of initial position and velocity (see section on Starting State) |
    """

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 100,
    }

    root_body = "mid_point"
    left_gripper_geom = "left_gripper_geom"
    right_gripper_geom = "right_gripper_geom"
    left_foot = "left_foot"
    right_foot = "right_foot"

    inchworm_xml_file = path.join(path.dirname(__file__), "inchworm.xml")
    old_inchworm_xml_file = path.join(path.dirname(__file__), "inchworm_old.xml")

    def __init__(
        self,
        xml_file=inchworm_xml_file,
        episode_length=1000,
        old_model=False,
        evals=False,
        ctrl_cost_weight=0.5,
        ungrounded_cost_weight=100,
        healthy_reward=1.0,
        terminate_when_unhealthy=True,
        reset_noise_scale=0.1,
        msr_eta: float | None = 1.5,
        **kwargs,
    ):
        if old_model:
            xml_file = self.old_inchworm_xml_file
            self.root_body = self.left_foot

        utils.EzPickle.__init__(
            self,
            xml_file,
            episode_length,
            old_model,
            evals,
            ctrl_cost_weight,
            ungrounded_cost_weight,
            healthy_reward,
            terminate_when_unhealthy,
            reset_noise_scale,
            msr_eta,
            **kwargs,
        )

        # How many frames to apply an action for when that action is applied to the environment
        frame_skip = 5

        # Store parameters
        self._episode_length = episode_length
        self._old_model = old_model
        self._evals = evals
        self._ctrl_cost_weight = ctrl_cost_weight
        self._ungrounded_cost_weight = ungrounded_cost_weight

        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy

        self._reset_noise_scale = reset_noise_scale

        self._msr_eta = msr_eta

        obs_shape = 12

        observation_space = Box(
            low=-np.inf, high=np.inf, shape=(obs_shape,), dtype=np.float64
        )

        # Indicates whether the inchworm has contacted the ground yet
        self.has_contacted_ground = False

        self._prev_reward: float | None = None

        # Evaluation records
        if self._evals:
            self._evals_reward_record = []
            self._evals_velocity_record = []
            self._evals_motor_input_record = []
            self._evals_ground_contact_record = []
            self._evals_upside_down_record = []

            self._evals_reward_avg_record = []
            self._evals_velocity_avg_record = []
            self._evals_motor_input_avg_record = []
            self._evals_ground_contact_freq_record = []
            self._evals_upside_down_freq_record = []

            self._eval_avgs = {
                "reward_avg": 0,
                "velocity_avg": 0,
                "motor_input_avg": 0,
                "ground_contact_freq": 0,
                "upside_down_freq": 0,
            }

        MujocoEnv.__init__(
            self,
            xml_file,
            frame_skip=frame_skip,
            observation_space=observation_space,
            default_camera_config=DEFAULT_CAMERA_CONFIG,
            **kwargs,
        )

    def _set_action_space(self):
        """
        Overriding this method so that we can manually set our action
        space to be in the range [-1, 1] for all actuators. This consistent
        and somewhat normalized action space helps the performance of many
        RL algorithms
        """
        num_actuators = self.model.nu
        self.action_space = Box(
            low=-1, high=1, shape=(num_actuators,), dtype=np.float32
        )
        return self.action_space

    @property
    def is_grounded(self):
        """
        Whether the inchworm is currently touching the ground with at least one foot
        """
        if self._old_model:
            return self.data.ncon > 0

        left_gripper_id = self.data.geom(self.left_gripper_geom).id
        right_gripper_id = self.data.geom(self.right_gripper_geom).id
        grounded = False

        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            if contact.geom1 == left_gripper_id or contact.geom2 == left_gripper_id:
                grounded = True
            if contact.geom1 == right_gripper_id or contact.geom2 == right_gripper_id:
                grounded = True
        return grounded

    @property
    def is_upside_down(self):
        """
        Returns true if the inchworm is upside down
        """
        left_foot_xpos = self.get_body_com(self.left_foot).copy()[0]
        right_foot_xpos = self.get_body_com(self.right_foot).copy()[0]
        return left_foot_xpos - right_foot_xpos > 1

    @property
    def healthy_reward(self):
        return (
            float(self.is_healthy or self._terminate_when_unhealthy)
            * self._healthy_reward
        )

    @staticmethod
    def positional_reward(x: float, c: float = 50.0, d: float = 0.05) -> float:
        """
        Reward for being at a certain position
        """
        return c * np.tanh(d * x)

    def control_cost(self, action):
        """
        Control cost is a penalty for applying large forces to the hinge motors
        (the first three values of the action)
        """
        control_cost: float = self._ctrl_cost_weight * np.sum(np.square(action[:3]))
        return control_cost

    def ungrounded_cost(self):
        """
        Cost for being ungrounded
        """
        self.has_contacted_ground = self.has_contacted_ground or self.is_grounded
        grounded = (
            not self.has_contacted_ground or self.is_grounded
        )  # Once grounded, must stay grounded
        ungrounded_cost = self._ungrounded_cost_weight * float(not grounded)
        return ungrounded_cost

    @property
    def is_healthy(self):
        # State vector contains all the positions and velocities
        state = self.state_vector()
        is_finite = np.isfinite(state).all()
        is_healthy = is_finite
        return is_healthy

    @property
    def terminated(self):
        terminated = not self.is_healthy if self._terminate_when_unhealthy else False
        return terminated

    def step(self, action):
        """
        Correct the inputs to the adhesion actuators so that
        they are either 0 (off) or 1 (on). They are given in
        the range [-1, 1]
        """
        rescaled_adhesion_actions = (action[3:5] + 1) / 2  # Rescale to [0, 1]
        action[3:5] = np.round(rescaled_adhesion_actions)  # Round to 0 or 1

        # Record current robot position, apply the action to the simulation, then record the resulting robot position
        xpos_before, _, _ = self.get_body_com(self.root_body).copy()
        self.do_simulation(action, self.frame_skip)
        xpos_after, _, _ = self.get_body_com(self.root_body).copy()

        # Calculate the robot's forward (x-axis) velocity based on its change in position
        x_velocity = (xpos_after - xpos_before) / self.dt

        # Calculate positive rewards
        positional_reward = InchwormEnv.positional_reward(xpos_after, d=0.05)
        forward_reward = x_velocity * 3
        healthy_reward = self.healthy_reward

        rewards = positional_reward + forward_reward + healthy_reward

        # Calculate penalties
        ctrl_cost = self.control_cost(action)
        ungrounded_cost = self.ungrounded_cost()

        costs = ungrounded_cost  # + ctrl_cost

        # Calculate total reward to give to the agent
        reward = rewards - costs
        # Conditionally apply MSR algorithm to reward
        if self._msr_eta is not None:
            reward = InchwormEnv._msr(
                self._prev_reward or reward, reward, self._msr_eta
            )
        self.displacement = max(self.displacement, xpos_after)

        terminated = self.terminated

        self.num_steps += 1
        truncated = self.num_steps == self._episode_length

        observation = self._get_obs()

        # Compile informative statistics to pass back to the caller
        info = {
            "reward_forward": forward_reward,
            "reward_survive": healthy_reward,
            "penalty_ctrl": ctrl_cost,
            "x_position": xpos_after,
            "x_velocity": x_velocity,
        }

        # Render the current simulation frame
        if self.render_mode == "human":
            self.render()

        if self._evals:
            self._evals_reward_record.append(reward)
            self._evals_velocity_record.append(x_velocity)
            self._evals_motor_input_record.append(ctrl_cost / self._ctrl_cost_weight)
            self._evals_ground_contact_record.append(self.is_grounded)
            self._evals_upside_down_record.append(self.is_upside_down)

            info["evals"] = self._eval_avgs

        self._prev_reward = reward

        return observation, reward, terminated, truncated, info

    def _get_obs(self):
        # Agent is allowed to sense the position and velocity of each DOF across all its joints
        position = self.data.qpos.flat.copy()
        velocity = self.data.qvel.flat.copy()

        return np.concatenate((position, velocity))

    @staticmethod
    def _msr(r1: float, r2: float, eta: float) -> float:
        sigma = 0.0001
        r1p = r1 + sigma
        r2p = r2 + sigma
        rho = (r2p - r1p) / min(abs(r2p), abs(r2p))
        lambda_ = np.sign(r2 - r1)
        x = rho + lambda_
        f = np.arctan(x * (np.pi / 2) * (1 / eta))
        msr = r2 + (f - lambda_) * abs(r2)
        if -eta < x < eta:
            return r2
        return msr

    def reset_model(self):
        # Low and high ends of the random noise that gets added to the initial positions and velocities
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        # Add noise to the initial positions and velocities
        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq
        )
        qvel = (
            self.init_qvel
            + self._reset_noise_scale * self.np_random.standard_normal(self.model.nv)
        )
        self.set_state(qpos, qvel)

        self._prev_reward = None

        if self._evals and len(self._evals_reward_record) > 100:
            ep_eval = InchwormEnv.calc_evals(
                {
                    "reward_avg": self._evals_reward_record,
                    "velocity_avg": self._evals_velocity_record,
                    "motor_input_avg": self._evals_motor_input_record,
                    "ground_contact_freq": self._evals_ground_contact_record,
                    "upside_down_freq": self._evals_upside_down_record,
                }
            )
            InchwormEnv.print_evals(ep_eval, "Episode Evaluation")

            # Save averages and frequencies to lists
            self._evals_reward_avg_record.append(ep_eval["reward_avg"])
            self._evals_velocity_avg_record.append(ep_eval["velocity_avg"])
            self._evals_motor_input_avg_record.append(ep_eval["motor_input_avg"])
            self._evals_ground_contact_freq_record.append(
                ep_eval["ground_contact_freq"]
            )
            self._evals_upside_down_freq_record.append(ep_eval["upside_down_freq"])

            # Calculate the average of the averages and the average of the frequencies
            self._eval_avgs = InchwormEnv.calc_evals(
                {
                    "reward_avg": self._evals_reward_avg_record,
                    "velocity_avg": self._evals_velocity_avg_record,
                    "motor_input_avg": self._evals_motor_input_avg_record,
                    "ground_contact_freq": self._evals_ground_contact_freq_record,
                    "upside_down_freq": self._evals_upside_down_freq_record,
                }
            )

            # Reset the evaluation statistics
            self._evals_reward_record = []
            self._evals_velocity_record = []
            self._evals_motor_input_record = []
            self._evals_ground_contact_record = []
            self._evals_upside_down_record = []

        self.num_steps = 0
        self.displacement = self.get_body_com(self.root_body)[0].copy()
        self.contacted_ground = False

        # Retrieve and return the first observation of the reset environment
        observation = self._get_obs()

        return observation

    @staticmethod
    def calc_evals(evals) -> dict:
        return {
            "reward_avg": np.mean(evals["reward_avg"]),
            "velocity_avg": np.mean(evals["velocity_avg"]),
            "motor_input_avg": np.mean(evals["motor_input_avg"]),
            "ground_contact_freq": np.sum(evals["ground_contact_freq"])
            / len(evals["ground_contact_freq"]),
            "upside_down_freq": np.sum(evals["upside_down_freq"])
            / len(evals["upside_down_freq"]),
        }

    @staticmethod
    def print_evals(evals: dict, label: str):
        print(
            f"\n{label}\n"
            + f"\treward_avg:          {evals['reward_avg']}\n"
            + f"\tvelocity_avg:        {evals['velocity_avg']}\n"
            + f"\tmotor_input_avg:     {evals['motor_input_avg']}\n"
            + f"\tground_contact_freq: {evals['ground_contact_freq']}\n"
            + f"\tupside_down_freq:    {evals['upside_down_freq']}\n"
        )
