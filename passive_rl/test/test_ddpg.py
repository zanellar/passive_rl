 
import numpy as np 
from stable_baselines3 import HER, SAC, TD3, DDPG 
from stable_baselines3.common.evaluation import evaluate_policy 
    
from passive_rl.utils.pkgpaths import PkgPath
from passive_rl.envs.reacher.reacher_ebud import ReacherEBud


max_episode_length = 50

env = ReacherEBud(
    max_episode_length = max_episode_length,  
    energy_tank_init = 7, # initial energy in the tank
    energy_tank_threshold = 0, # minimum energy in the tank  
    debug = False,
    mjmodel="reacher_test"
)  
  
model = DDPG.load(PkgPath.trainingdata("checkpoints/reacher/DDPG/best_model/best_model.zip"))
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10, render=True, deterministic=True)

env.close()
