import numpy as np  
from mjrlenvs.scripts.train.trainer import run 
from mjrlenvs.scripts.train.trainutils import linear_schedule 
from mjrlenvs.scripts.args.runargsbase import DefaultArgs 
from passive_rl.envs.halfcheetah import HalfCheetahEBud, HalfCheetahEBudAw
from passive_rl.scripts.pkgpaths import PkgPath
from passive_rl.scripts.energycb import SaveEnergyLogsCallback 
from passive_rl.scripts.errorscb import ErrorsCallback  

class Args(DefaultArgs):  
  
    OUT_TRAIN_FOLDER = PkgPath.OUT_TRAIN_FOLDER
    OUT_TEST_FOLDER = PkgPath.OUT_TEST_FOLDER

    REPETE_TRAINING_TIMES = 1 # times
    TRAINING_EPISODES = 1000 # episodes
    EXPL_EPISODE_HORIZON = 1000 # timesteps 
    EVAL_EPISODE_HORIZON = 1000 # timesteps  
    EVAL_MODEL_FREQ = 200 # episodes
    NUM_EVAL_EPISODES = 1
    NUM_EVAL_EPISODES_BEST_MODEL = 1
    EARLY_STOP = False 

    ENV_EVAL_RENDERING = False
    SAVE_EVAL_MODEL_WEIGHTS = True 
    SAVE_CHECKPOINTS = True
    SAVE_ALL_TRAINING_LOGS = False
 
    CALLBACKS = [SaveEnergyLogsCallback() ]
  
    AGENT = "SAC"  
    AGENT_PARAMS = dict(
        seed = [[0,7,11,17,29]], #[17,29,67,157,109,211,277,331,359,419] #TODO
        buffer_size = [int(1e6)],
        batch_size = [256],
        learning_starts = [1*EXPL_EPISODE_HORIZON],  
        train_freq = [(8,"step")], 
        gradient_steps = [8], #1000
        learning_rate = [7.3e-4], # 3e-3
        gamma = [0.99],
        tau = [0.02],
        noise = [None],
        sigma = [0.1],
        policy_kwargs = [dict(log_std_init=-3, net_arch=[400, 300])],
        use_sde_at_warmup = [False],
        use_sde = [True ],
        sde_sample_freq = [-1], 
        ent_coef = ['auto'], 
        target_update_interval = [1],   
    )
  
    @staticmethod
    def set(args): 
        Args.RUN_ID = args["RUN_ID"]
        Args.ENERGY_TANK_INIT = args["ENERGY_TANK_INIT"] 
        Args.ENVIRONMENT = args["ENVIRONMENT"]  if "ENVIRONMENT" in args.keys() else "halfcheetah"  
        Args.ENERGY_TERMINATE  = args["ENERGY_TERMINATE"]  if "ENERGY_TERMINATE" in args.keys() else False 

        if args["ENERGY_AWARE"]:
            # Args.NORMALIZE_ENV = dict(training=True, norm_obs=True, norm_reward=True, clip_obs=np.inf, clip_reward=10) 
            Args.ENV_EXPL = HalfCheetahEBudAw( 
                            energy_tank_init = Args.ENERGY_TANK_INIT, # initial energy in the tank
                            energy_tank_threshold = 0, # minimum energy in the tank   
                        ) 
            Args.ENV_EVAL = HalfCheetahEBudAw( 
                            energy_tank_init = Args.ENERGY_TANK_INIT, # initial energy in the tank
                            energy_tank_threshold = 0, # minimum energy in the tank   
                        )    
        else:
            # Args.NORMALIZE_ENV = dict(training=True, norm_obs=True, norm_reward=True, clip_obs=1, clip_reward=10)
            Args.ENV_EXPL = HalfCheetahEBud( 
                        energy_tank_init = Args.ENERGY_TANK_INIT, # initial energy in the tank
                        energy_tank_threshold = 0, # minimum energy in the tank   
                        energy_terminate = Args.ENERGY_TERMINATE   
                    )

            Args.ENV_EVAL = HalfCheetahEBud( 
                        energy_tank_init = Args.ENERGY_TANK_INIT, # initial energy in the tank
                        energy_tank_threshold = 0, # minimum energy in the tank   
                        energy_terminate = Args.ENERGY_TERMINATE  
                    )
            