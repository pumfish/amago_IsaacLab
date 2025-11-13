"""
Wrapper for k-attempt IsaacLab tasks
"""

import torch
import random
import numpy as np

from amago.utils import amago_warning
from amago.envs.env_utils import space_convert
from amago.envs import AMAGOEnv, AMAGO_ENV_LOG_PREFIX

from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

import gym as og_gym
import gymnasium as gym

import isaaclab_tasks
from isaaclab_tasks.utils import load_cfg_from_registry


class IsaacLabTask(AMAGOEnv):
    def __init__(self,
                 env_name="Isaac-Grasp-Cube-Franka-DR",
                 batched_envs=2048,
        ):
        cfg = load_cfg_from_registry(env_name, "env_cfg_entry_point")
        cfg.scene.num_envs=batched_envs
        self.env = gym.make(env_name, cfg=cfg)
        self.is_isaaclab_task=True

        self.batched_envs = batched_envs
        self._env_name = env_name

        # action space conversion
        self.discrete = isinstance(self.env.action_space, gym.spaces.Discrete)
        self.multibinary = isinstance(self.env.action_space, gym.spaces.MultiBinary)

        # skip action wrapper in original AMAGOEnv
        self.action_space = self.env.action_space
        self.action_size = self.action_space.shape[-1]
        self.action_dtype = np.float32
        self._batch_idxs = np.arange(self.batched_envs)

        # observation space conversion (defaults to dict)
        obs_space = self.env.observation_space
        # AMAGO may need key="observation"
        obs_space["observation"] = obs_space["policy"]
        self.observation_space = obs_space

    
    @property
    def env_name(self):
        return self._env_name
    
    def inner_step(self, action):
        # numpy -> tensor
        if hasattr(self, "is_isaaclab_task"):
            if self.is_isaaclab_task:
                action = torch.as_tensor(action, device=self.env.device)
        obs, rewards, terminateds, truncateds, infos = self.env.step(action)
        # tensor -> numpy
        # key "policy" will be key "observation"
        obs = obs['policy'].cpu().numpy()
        rewards = rewards.cpu().numpy()
        terminateds = terminateds.cpu().numpy()
        truncateds = truncateds.cpu().numpy()
        return obs, rewards, terminateds, truncateds, infos


if __name__ == "__main__":
    test_env = IsaacLabTask()
    print("Observation space:", test_env.observation_space)
    print("Action space:", test_env.action_space)
    print("Env name:", test_env.env_name)

    timestep, info = test_env.reset()
    print("Reset observation shape:", {k: v.shape for k, v in timestep.obs.items()})
    print("Initial prev_action shape:", timestep.prev_action.shape)
    print("Initial reward:", timestep.reward)
    print("Initial terminal flags:", timestep.terminal)

    for i in range(5):
        action = test_env.action_space.sample()
        timestep, reward, terminated, truncated, metrics = test_env.step(action)
        print(f"\nStep {i+1}:")
        print("Action:", action)
        print("Next obs shape:", {k: v.shape for k, v in timestep.obs.items()})
        print("Reward:", reward)
        print("Terminated:", terminated)
        print("Truncated:", truncated)
        print("Metrics:", metrics)

