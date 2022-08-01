import os 
from pickle import FALSE
import numpy as np 
from stable_baselines3 import HER, SAC, TD3, DDPG 
from stable_baselines3.common.evaluation import evaluate_policy 
from stable_baselines3.common.callbacks import BaseCallback
from passive_rl.utils.pkgpaths import PkgPath
from passive_rl.envs.reacher.reacher_ebud import ReacherEBud

PATH_TRAINED_MODEL = "/home/riccardo/projects/passive_rl/data/training/checkpoints/reacher/DDPG/best_model/DDPG_reacher_gym.zip"
MAX_EPISODE_LENGTH = 100
RENDERING = True
# RENDERING = False
# MUJOCO_SIMULATION_SCENARIO = "reacher_original"
MUJOCO_SIMULATION_SCENARIO = "reacher_test"

########################################################################
########################################################################
########################################################################

########################################################################

if not os.path.exists(PkgPath.testingdata()):
    os.makedirs(PkgPath.testingdata())

with open(PkgPath.testingdata("output.txt"), 'w') as f: 
    line = "timestep,energy_tank,energy_exchanged" 
    f.write(line)
    f.close()

########################################################################

def energycallback(locals_, _globals): 
    energy_tank = locals_["info"]["energy_tank"]
    energy_exchanged = locals_["info"]["energy_exchanged"]
    t = locals_["current_lengths"][0]
    with open(PkgPath.testingdata("output.txt"), 'a') as f: 
        line = f"\n{t},{energy_tank},{energy_exchanged}" 
        f.write(line)
        f.close()
 

########################################################################

env = ReacherEBud(
    max_episode_length = MAX_EPISODE_LENGTH,  
    energy_tank_init = 0.05, # initial energy in the tank
    energy_tank_threshold = 0.005, # minimum energy in the tank  
    testing_mode = True, 
    mjmodel = MUJOCO_SIMULATION_SCENARIO
)  
  
model = DDPG.load(PkgPath.trainingdata(PATH_TRAINED_MODEL))
 
mean_reward, std_reward = evaluate_policy(
                            model,
                            env, 
                            n_eval_episodes=100,  
                            render = RENDERING, 
                            deterministic=True,
                            callback=energycallback
                            )

print(mean_reward, std_reward)

env.close()
