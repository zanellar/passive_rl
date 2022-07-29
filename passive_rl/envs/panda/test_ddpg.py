
import os 
import numpy as np  
from typing import Callable
import numpy as np 
from stable_baselines3 import HerReplayBuffer, SAC, TD3, DDPG
from stable_baselines3.common.callbacks import BaseCallback,CallbackList, EvalCallback,CheckpointCallback 
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from gym.wrappers.normalize import NormalizeObservation, NormalizeReward 
from stable_baselines3.common.evaluation import evaluate_policy
 
from panda_ebud import PandaEBud

SAVE_DATA_FOLDER = "/home/riccardo/projects/nrg_rl/contrib1/panda"

max_episode_length = int(1e4) 

env = PandaEBud(
    max_episode_length=max_episode_length,
    energy_tank_init = 5, # initial energy in the tank
    energy_tank_threshold = 0, # minimum energy in the tank  
)  
  

env = Monitor(env)                      
env = NormalizeObservation(env)
env = NormalizeReward(env)

model = DDPG.load("./logs/best_ddpg/best_model")
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10, render=True, deterministic=True)

env.close()
