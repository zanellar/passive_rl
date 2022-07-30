import os
import gym
import itertools
from typing import Callable
import numpy as np 
from stable_baselines3 import HerReplayBuffer, PPO, DDPG
from stable_baselines3.common.callbacks import BaseCallback,CallbackList, EvalCallback,CheckpointCallback 
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from gym.wrappers.normalize import NormalizeObservation, NormalizeReward

from passive_rl.utils.pkgpaths import PkgPath
from passive_rl.utils.trainfuncs import linear_schedule
from passive_rl.envs.reacher.reacher_ebud import ReacherEBud

AGENT_NAME = "PPO"
ENV_NAME = 'reacher' 
max_episode_length = 50 

params = dict(
    gamma = [0.9,0.95,0.99], 
    bsize = [256,64],   
    gae_lambda = [0.9,0.95,1],
    clip_range = [0.1,0.3,linear_schedule(0.3)],
    ent_coef = [0,0.001,0.01],
    vf_coef = [0.5,0.8,1],
    use_sde = [True,False],
    sde_sample_freq = [-1,64,8], 
    normalize_advantage=[True,False], 

)

keys = params.keys()
values = (params[key] for key in keys)
allconfigurations = [dict(zip(keys, combination)) for combination in itertools.product(*values)]

########################################################################
########################################################################
########################################################################

env = gym.make('Reacher-v2') 

for i,config in enumerate(allconfigurations):
 
    model = PPO(
        policy = 'MlpPolicy',
        env = env,   
        learning_rate= linear_schedule(1e-4), 
        gamma = config["gamma"],
        batch_size = config["bsize"],    
        gae_lambda = config["gae_lambda"], 
        clip_range = config["clip_range"], 
        clip_range_vf = None, 
        normalize_advantage = config["normalize_advantage"], 
        ent_coef = config["ent_coef"], 
        vf_coef = config["vf_coef"], 
        max_grad_norm = 0.5, 
        use_sde = config["use_sde"], 
        sde_sample_freq = config["sde_sample_freq"], 
        target_kl = None,
        verbose = 1, 
        tensorboard_log =  PkgPath.trainingdata(f"tensorboard/{ENV_NAME}")
    )  
 
    ######################################################################## 

    callbackslist = []
    
    # callbackslist.append(
    #     CheckpointCallback(
    #         save_freq = 1000*max_episode_length, 
    #         save_path = PkgPath.trainingdata(f"checkpoints/{ENV_NAME}/{AGENT_NAME}")
    #     )
    # )

    callbackslist.append(
        EvalCallback( # NB: need to comment "sync_envs_normalization" function in EvalCallback._on_step() method 
            env, 
            best_model_save_path = PkgPath.trainingdata(f'checkpoints/{ENV_NAME}/{AGENT_NAME+"_"+str(i+1)}/best_model'),
            log_path = PkgPath.trainingdata( f'checkpoints/{ENV_NAME}/{AGENT_NAME+"_"+str(i+1)}/eval_results'), 
            eval_freq = 50*max_episode_length,
            n_eval_episodes = 1, 
            deterministic = True, 
            render = False
        )
    )  
 
    callbacks = CallbackList(callbackslist)


    ########################################################################

    if not os.path.exists(PkgPath.trainingdata()):
        os.makedirs(PkgPath.trainingdata())

    with open(PkgPath.trainingdata("configs.txt"), 'a') as f:
        f.write("\n"+AGENT_NAME+"_"+str(i+1)+":  "+str(config))
        f.close()
        
    ########################################################################

    env.reset()   
    model.learn(
        total_timesteps = 1000*max_episode_length, 
        log_interval = 1,  
        callback = callbacks
    )  

    env.close()
 