import os
import gym
import itertools
import numpy as np 
from stable_baselines3 import HerReplayBuffer, SAC, TD3, DDPG
from stable_baselines3.common.callbacks import BaseCallback,CallbackList, EvalCallback,CheckpointCallback 
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from gym.wrappers.normalize import NormalizeObservation, NormalizeReward

from passive_rl.utils.pkgpaths import PkgPath
from passive_rl.utils.trainfuncs import linear_schedule
from passive_rl.envs.reacher.reacher_ebud import ReacherEBud

AGENT_NAME = "DDPG"
ENV_NAME = 'reacher' 
max_episode_length = 50

 
params = dict(
    gamma = [0.9,0.95,0.99],
    tau = [1, 1e-2, 1e-3, 1e-4],
    bsize = [256,64],
    start = [0, 10*max_episode_length, 100*max_episode_length],
    sigma = [0.2,0.3,0.5],
    # normobs = [True, False]
)

keys = params.keys()
values = (params[key] for key in keys)
allconfigurations = [dict(zip(keys, combination)) for combination in itertools.product(*values)]
 
########################################################################
########################################################################
########################################################################

env = gym.make('Reacher-v2')

for i,config in enumerate(allconfigurations):
  
    action_noise = OrnsteinUhlenbeckActionNoise(
        mean = np.zeros(env.action_space.shape), 
        sigma = config["sigma"]* np.ones(env.action_space.shape)  
    )

    model = DDPG(
        policy = 'MlpPolicy',
        env = env,  
        buffer_size=int(1e6),
        learning_rate= linear_schedule(1e-4), 
        gamma = config["gamma"],
        tau = config["tau"],  
        batch_size = config["bsize"],    
        action_noise = action_noise, 
        learning_starts = config["start"], 
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
        EvalCallback(
            env, 
            best_model_save_path = PkgPath.trainingdata(f'checkpoints/{ENV_NAME}/{AGENT_NAME+"_"+str(i+1)}/best_model'),
            log_path = PkgPath.trainingdata( f'checkpoints/{ENV_NAME}/{AGENT_NAME+"_"+str(i+1)}/eval_results'), 
            eval_freq = 50*max_episode_length,
            n_eval_episodes = 1, 
            deterministic = True, 
            render = False
        )
    )  # NB: need to comment "sync_envs_normalization" function in EvalCallback._on_step() method 
 
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

 