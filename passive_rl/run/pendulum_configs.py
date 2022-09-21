
from mjrlenvs.scripts.train.trainer import run 
from mjrlenvs.scripts.train.trainutils import linear_schedule 
from mjrlenvs.scripts.args.runargsbase import DefaultArgs 
from passive_rl.envs.pendulum.pendulum import PendulumEBud
from passive_rl.scripts.pkgpaths import PkgPath
from passive_rl.scripts.energycb import SaveEnergyLogsCallback
from passive_rl.scripts.tester import TestRunEBud 

class Args(DefaultArgs): 

    ############################## RUN #########################################################

    RUN_ID = "etank1000"   
    OUT_TRAIN_FOLDER = PkgPath.OUT_TRAIN_FOLDER
    OUT_TEST_FOLDER = PkgPath.OUT_TEST_FOLDER
    EXPL_EPISODE_HORIZON = 2500 # timesteps 
    EVAL_EPISODE_HORIZON = 500 # timesteps  
    TRAINING_EPISODES = 100 # episodes
    EVAL_MODEL_FREQ = 10*EXPL_EPISODE_HORIZON 
    NUM_EVAL_EPISODES = 5
    NUM_EVAL_EPISODES_BEST_MODEL = 5
    REPETE_TRAINING_TIMES = 20 # times
    SAVE_EVAL_MODEL_WEIGHTS = True 
    SAVE_CHECKPOINTS = True
    EARLY_STOP = False
    # EARLY_STOP_MAX_NO_IMPROVEMENTS = 3
    # EARLY_STOP_MIN_EVALS = 5
    SAVE_ALL_TRAINING_LOGS = False

    CALLBACKS = [SaveEnergyLogsCallback()]
 
    ########################## ENVIRONMENT ######################################################

    ENVIRONMENT = "pendulum_f001" 

    ENV_EXPL = PendulumEBud(
                max_episode_length=EXPL_EPISODE_HORIZON,  
                energy_tank_init = 1000, # initial energy in the tank
                energy_tank_threshold = 0, # minimum energy in the tank  
                init_joint_config = "random",
                folder_path = PkgPath.ENV_DESC_FOLDER,
                env_name = ENVIRONMENT
            )

    ENV_EVAL = PendulumEBud(
                max_episode_length=EVAL_EPISODE_HORIZON, 
                energy_tank_init = 1000, # initial energy in the tank
                energy_tank_threshold = 0, # minimum energy in the tank  
                init_joint_config = "random",
                folder_path = PkgPath.ENV_DESC_FOLDER,
                env_name = ENVIRONMENT,
                hard_reset = False
            )

    NORMALIZE_ENV = dict(training=True, norm_obs=True, norm_reward=True, clip_obs=10, clip_reward=10) 

    ENV_EVAL_RENDERING = False

    ############################## AGENT #######################################################

    AGENT = "SAC"  

    AGENT_PARAMS = dict(
        seed = [17],
        buffer_size = [int(1e6)],
        batch_size = [128],
        learning_starts = [1*EXPL_EPISODE_HORIZON],  
        train_freq = [(500,"step") ], 
        gradient_steps = [1000],
        learning_rate = [ linear_schedule(3e-3) ], # 1e-3
        gamma = [0.99],
        tau = [1e-3],
        noise = ["gauss"],
        sigma = [0.1],
        policy_kwargs = [
                        dict(#log_std_init=-2, 
                        net_arch=dict(pi=[256,256],qf=[256,256]))
                        ],
        use_sde_at_warmup = [True ],
        use_sde = [True ],
        sde_sample_freq = [EXPL_EPISODE_HORIZON], 
        ent_coef = ['auto'], 
        target_update_interval = [5],   
    )
 
 