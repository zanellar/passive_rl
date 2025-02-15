import math
import numpy as np  
from mjrlenvs.scripts.train.trainer import run 
from mjrlenvs.scripts.train.trainutils import linear_schedule 
from mjrlenvs.scripts.args.runargsbase import DefaultArgs 
from passive_rl.envs.pendulum import PendulumEBud, PendulumEBudAw
from passive_rl.scripts.pkgpaths import PkgPath
from passive_rl.scripts.energycb import SaveEnergyLogsCallback 
from passive_rl.scripts.errorscb import ErrorsCallback  

class Args(DefaultArgs):  
  
    OUT_TRAIN_FOLDER = PkgPath.OUT_TRAIN_FOLDER
    OUT_TEST_FOLDER = PkgPath.OUT_TEST_FOLDER

    REPETE_TRAINING_TIMES = 1 # times
    TRAINING_EPISODES = 200 # episodes
    EXPL_EPISODE_HORIZON = 2500 # timesteps 
    EVAL_EPISODE_HORIZON = 500 # timesteps  
    EVAL_MODEL_FREQ = 20 # episodes
    NUM_EVAL_EPISODES = 1
    NUM_EVAL_EPISODES_BEST_MODEL = 1
    EARLY_STOP = False 

    SAVE_EVAL_MODEL_WEIGHTS = True 
    SAVE_CHECKPOINTS = True
    SAVE_ALL_TRAINING_LOGS = False
 
    CALLBACKS = [SaveEnergyLogsCallback(), ErrorsCallback()]
  
    AGENT = "SAC"  
    AGENT_PARAMS = dict(
        seed = [0,7,11,17,29], #[17,29,67,157,109,211,277,331,359,419] #TODO
        buffer_size = [int(1e6)],
        batch_size = [256],
        learning_starts = [1*EXPL_EPISODE_HORIZON],  
        train_freq = [(500,"step") ], 
        gradient_steps = [500], #1000
        learning_rate = [ linear_schedule(5e-3) ], # 3e-3
        gamma = [0.99],
        tau = [3e-3],
        noise = ["gauss"],
        sigma = [0.1],
        policy_kwargs = [dict(net_arch=dict(pi=[256,256],qf=[256,256]))],
        use_sde_at_warmup = [True ],
        use_sde = [True ],
        sde_sample_freq = [EXPL_EPISODE_HORIZON], 
        ent_coef = ['auto'], 
        target_update_interval = [5],   
    )
  
    @staticmethod
    def set(args): 
        Args.RUN_ID = args["RUN_ID"]
        Args.ENERGY_TANK_INIT = args["ENERGY_TANK_INIT"] 
        Args.ENVIRONMENT = args["ENVIRONMENT"]  if "ENVIRONMENT" in args.keys() else "pendulum" 
        Args.INIT_JOINT_CONFIG = args["INIT_JOINT_CONFIG"] if "INIT_JOINT_CONFIG" in args.keys() else "random"
        Args.INIT_JOINT_CONFIG_STD_NOISE = args["INIT_JOINT_CONFIG_STD_NOISE"] if "INIT_JOINT_CONFIG_STD_NOISE" in args.keys() else 0.1*math.pi/2 
        Args.ENERGY_TERMINATE  = args["ENERGY_TERMINATE"]  if "ENERGY_TERMINATE" in args.keys() else False
        Args.REWARD_ID = args["REWARD_ID"]  if "REWARD_ID" in args.keys() else 0

        if args["ENERGY_AWARE"]:
            Args.NORMALIZE_ENV = dict(training=True, norm_obs=True, norm_reward=True, clip_obs=np.inf, clip_reward=10) 
            Args.ENV_EXPL = PendulumEBudAw(
                            max_episode_length = Args.EXPL_EPISODE_HORIZON,  
                            energy_tank_init = Args.ENERGY_TANK_INIT, # initial energy in the tank
                            energy_tank_threshold = 0, # minimum energy in the tank  
                            init_joint_config = Args.INIT_JOINT_CONFIG, # [-1.57] or "random"
                        init_joint_config_std_noise = Args.INIT_JOINT_CONFIG_STD_NOISE,  
                            folder_path = PkgPath.ENV_DESC_FOLDER,
                            env_name = Args.ENVIRONMENT, 
                            reward_id = Args.REWARD_ID  
                        ) 
            Args.ENV_EVAL = PendulumEBudAw(
                            max_episode_length = Args.EVAL_EPISODE_HORIZON,  
                            energy_tank_init = Args.ENERGY_TANK_INIT, # initial energy in the tank
                            energy_tank_threshold = 0, # minimum energy in the tank  
                            init_joint_config = Args.INIT_JOINT_CONFIG,# [-1.57] or "random"
                        init_joint_config_std_noise = Args.INIT_JOINT_CONFIG_STD_NOISE,  
                            folder_path = PkgPath.ENV_DESC_FOLDER,
                            env_name = Args.ENVIRONMENT, 
                            reward_id = Args.REWARD_ID  
                        )    
        else:
            Args.NORMALIZE_ENV = dict(training=True, norm_obs=True, norm_reward=True, clip_obs=1, clip_reward=10)
            Args.ENV_EXPL = PendulumEBud(
                        max_episode_length = Args.EXPL_EPISODE_HORIZON,  
                        energy_tank_init = Args.ENERGY_TANK_INIT, # initial energy in the tank
                        energy_tank_threshold = 0, # minimum energy in the tank  
                        init_joint_config = Args.INIT_JOINT_CONFIG, # [-1.57] or "random"
                        init_joint_config_std_noise = Args.INIT_JOINT_CONFIG_STD_NOISE,  
                        folder_path = PkgPath.ENV_DESC_FOLDER,
                        env_name = Args.ENVIRONMENT,
                        energy_terminate = Args.ENERGY_TERMINATE, 
                        reward_id = Args.REWARD_ID  
                    )

            Args.ENV_EVAL = PendulumEBud(
                        max_episode_length = Args.EVAL_EPISODE_HORIZON, 
                        energy_tank_init = Args.ENERGY_TANK_INIT, # initial energy in the tank
                        energy_tank_threshold = 0, # minimum energy in the tank  
                        init_joint_config = Args.INIT_JOINT_CONFIG, # [-1.57] or "random"
                        init_joint_config_std_noise = Args.INIT_JOINT_CONFIG_STD_NOISE,  
                        folder_path = PkgPath.ENV_DESC_FOLDER,
                        env_name = Args.ENVIRONMENT,
                        hard_reset = False,
                        energy_terminate = Args.ENERGY_TERMINATE, 
                        reward_id = Args.REWARD_ID
                    )
            