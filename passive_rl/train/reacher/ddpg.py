import os
import gym
from typing import Callable
import numpy as np 
from stable_baselines3 import HerReplayBuffer, SAC, TD3, DDPG
from stable_baselines3.common.callbacks import BaseCallback,CallbackList, EvalCallback,CheckpointCallback 
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from gym.wrappers.normalize import NormalizeObservation, NormalizeReward

from passive_rl.utils.pkgpaths import PkgPath
from passive_rl.envs.reacher.reacher_ebud import ReacherEBud

AGENT_NAME = "DDPG"
ENV_NAME = 'reacher' 

def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        # print(progress_remaining * initial_value)
        return progress_remaining * initial_value

    return func


########################################################################
########################################################################
########################################################################

max_episode_length = 50

env = gym.make('Reacher-v2')

# env = ReacherEBud(
#     max_episode_length=max_episode_length,  
#     energy_tank_init = 7, # initial energy in the tank
#     energy_tank_threshold = 0, # minimum energy in the tank  
#     debug = False
# )  

# env = Monitor(env)                      # A monitor wrapper for Gym environments, it is used to know the episode reward, length, time and other data
# env = NormalizeObservation(env) 


action_noise = OrnsteinUhlenbeckActionNoise(
    mean=np.zeros(env.action_space.shape), 
    sigma=0.2* np.ones(env.action_space.shape)  
)

model = DDPG(
    policy = 'MlpPolicy',
    env = env,  
    buffer_size=100*256,
    learning_rate= linear_schedule(1e-3), 
    gamma=0.9,
    tau=0.01,  
    batch_size=256,    
    action_noise = action_noise,
    # replay_buffer_class=HerReplayBuffer,
    # replay_buffer_kwargs=dict(
    #     n_sampled_goal=4,
    #     goal_selection_strategy='future',
    #     online_sampling=True,
    #     max_episode_length=max_episode_length,
    # ),
    # train_freq = (1, "episode"),   
    # gradient_steps =  -1,  
    learning_starts = 100*max_episode_length, 
    verbose = 1, 
    tensorboard_log =  PkgPath.trainingdata(f"tensorboard/{ENV_NAME}")
) 

########################################################################
########################################################################
########################################################################

callbackslist = []

# class EnergyCallBack(BaseCallback): 
#     def __init__(self, env, verbose=0):
#         super(EnergyCallBack, self).__init__(verbose)
#         self.env = env
#         self.min_tank_level = self.env.energy_tank_init
 
#     def _on_step(self) -> bool:
#         self.logger.record('energy/termination_times', self.env.energy_stop_ct)  # tensorboard
#         self.min_tank_level = min(self.min_tank_level, self.env.energy_tank)
#         self.logger.record('energy/min_tank_level', self.min_tank_level)  # tensorboard
#         return True
  
# callbackslist.append(EnergyCallBack(env))

# callbackslist.append(
#     CheckpointCallback(
#         save_freq = 1000*max_episode_length, 
#         save_path = PkgPath.trainingdata(f"checkpoints/{ENV_NAME}/{AGENT_NAME}")
#     )
# )

callbackslist.append(
    EvalCallback(
        env, 
        best_model_save_path=PkgPath.trainingdata(f'checkpoints/{ENV_NAME}/{AGENT_NAME}/best_model'),
        log_path=PkgPath.trainingdata( f'checkpoints/{ENV_NAME}/{AGENT_NAME}/eval_results'), 
        eval_freq=50*max_episode_length,
        n_eval_episodes=10, 
        deterministic = True, 
        render = True
    )
)  # NB: need to comment "sync_envs_normalization" function in EvalCallback._on_step() method 

callbacks = CallbackList(callbackslist)

 
########################################################################
########################################################################
########################################################################

env.reset()   
model.learn(
    total_timesteps= 1000*max_episode_length, 
    log_interval=1,  
    callback = callbacks
)  

env.close()
