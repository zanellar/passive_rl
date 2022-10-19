import os
import numpy as np
from mjrlenvs.scripts.train.trainer import run   
from passive_rl.scripts.pkgpaths import PkgPath 
from passive_rl.scripts.tester import TestRunEBud     
from mjrlenvs.scripts.train.trainutils import linear_schedule 
from mjrlenvs.scripts.args.runargsbase import DefaultArgs  
from passive_rl.scripts.energycb import SaveEnergyLogsCallback 
from passive_rl.scripts.errorscb import ErrorsCallback  
  
from mjrlenvs.envrl.pendulum import Pendulum  
from passive_rl.scripts.ebudenv import EBudBaseEnv,EBudAwEnv
 
class PendulumEBud(EBudBaseEnv): 

    def __init__(self,    
                max_episode_length = 50, 
                energy_tank_init = 10, # initial energy in the tank
                energy_tank_threshold = 0, # minimum energy in the tank   
                init_joint_config = "random",
                debug = False,
                energy_terminate = False,
                folder_path = None,
                env_name = "pendulum",
                hard_reset = True
                ):
        super(PendulumEBud, self).__init__(
            env = Pendulum(
                env_name = env_name, 
                folder_path = folder_path,
                max_episode_length = max_episode_length, 
                init_joint_config = init_joint_config,
                hard_reset = hard_reset
            ),
            energy_tank_init = energy_tank_init, # initial energy in the tank
            energy_tank_threshold = energy_tank_threshold, # minimum energy in the tank  
            debug = debug,
            energy_terminate = energy_terminate  
        )
 
    def get_joints(self):
        obs = self.env.get_obs()
        joints = np.arcsin(obs[0])
        return joints

    def get_torques(self):
        return self.action

    def _update_energy_tank(self):
        ''' 
        Etank(t) = Etank(t-1) - Eex(t)
        Ex(t) = sum(tau(t-1,i)*(q(t,i) - q(t-1,i)), i=1,2)
        ''' 

        joints = self.get_joints()
        torques = self.get_torques()

        if self.joints is None:
            self.joints = joints
            return False

        old_joints = self.joints 
        new_joints = joints 
        d_joints = new_joints - old_joints
        self.energy_joints = torques*d_joints  
        self.energy_exchanged = sum(self.energy_joints)
        if self.energy_exchanged <=0: 
            self.energy_tank -= self.energy_exchanged 
        tank_is_empty = self.energy_tank <= self.energy_tank_threshold
        self.energy_avaiable = not tank_is_empty
        self.joints = new_joints
        return tank_is_empty
            
################################################################################################ 
################################################################################################ 
################################################################################################ 
################################################################################################ 
################################################################################################ 

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
        seed = [17,29,67,157,109,211,277,331,359,419],
        buffer_size = [int(1e6)],
        batch_size = [256],
        learning_starts = [1*EXPL_EPISODE_HORIZON],  
        train_freq = [(500,"step") ], 
        gradient_steps = [1000],
        learning_rate = [ linear_schedule(3e-3) ], # 1e-3
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
        Args.ENVIRONMENT = args["ENVIRONMENT"] 
        Args.ENERGY_TANK_INIT = args["ENERGY_TANK_INIT"]
        if args["ENERGY_AWARE"]:
            Args.NORMALIZE_ENV = dict(training=True, norm_obs=True, norm_reward=True, clip_obs=np.inf, clip_reward=10) 
            Args.ENV_EXPL = PendulumEBudAw(
                            max_episode_length = Args.EXPL_EPISODE_HORIZON,  
                            energy_tank_init = Args.ENERGY_TANK_INIT, # initial energy in the tank
                            energy_tank_threshold = 0, # minimum energy in the tank  
                            init_joint_config = [0],
                            folder_path = PkgPath.ENV_DESC_FOLDER,
                            env_name = Args.ENVIRONMENT
                        ) 
            Args.ENV_EVAL = PendulumEBudAw(
                            max_episode_length = Args.EVAL_EPISODE_HORIZON,  
                            energy_tank_init = Args.ENERGY_TANK_INIT, # initial energy in the tank
                            energy_tank_threshold = 0, # minimum energy in the tank  
                            init_joint_config = [0],
                            folder_path = PkgPath.ENV_DESC_FOLDER,
                            env_name = Args.ENVIRONMENT
                        )    
        else:
            Args.NORMALIZE_ENV = dict(training=True, norm_obs=True, norm_reward=True, clip_obs=1, clip_reward=10)
            Args.ENV_EXPL = PendulumEBud(
                        max_episode_length = Args.EXPL_EPISODE_HORIZON,  
                        energy_tank_init = Args.ENERGY_TANK_INIT, # initial energy in the tank
                        energy_tank_threshold = 0, # minimum energy in the tank  
                        init_joint_config = [0],
                        folder_path = PkgPath.ENV_DESC_FOLDER,
                        env_name =Args.ENVIRONMENT
                    )

            Args.ENV_EVAL = PendulumEBud(
                        max_episode_length = Args.EVAL_EPISODE_HORIZON, 
                        energy_tank_init = Args.ENERGY_TANK_INIT, # initial energy in the tank
                        energy_tank_threshold = 0, # minimum energy in the tank  
                        init_joint_config = [0],
                        folder_path = PkgPath.ENV_DESC_FOLDER,
                        env_name = Args.ENVIRONMENT,
                        hard_reset = False
                    )
            
################################################################################################ 
################################################################################################ 
################################################################################################ 
################################################################################################ 
################################################################################################ 

n_eval_episodes = 50

run_results_file_path = os.path.join(PkgPath.OUT_TRAIN_FOLDER,"run_results.txt")  
with open(run_results_file_path, 'w') as file: 
    line = ""   
    file.write(line) 

def test(x=None, test_id=""):    
    if x is not None:
        Args.set(x)  
    tester = TestRunEBud(Args, test_id=test_id)  
    tester.eval_returns_run(n_eval_episodes=n_eval_episodes, save=True)
    min_emin = tester.eval_emin_run(n_eval_episodes=n_eval_episodes, save=True)
    return min_emin

def train_and_test(x, test_id=""): 
    Args.set(x) 
    run(Args) 
    min_emin = test(Args, test_id=test_id) 
 
    with open(run_results_file_path, 'a') as file: 
        line = f"\n {Args.RUN_ID} {Args.ENVIRONMENT}, {Args.ENV_EXPL.energy_tank_init}, {min_emin}"   
        file.write(line)

    return min_emin


################################################################################################ 

train_and_test(
    dict(
        RUN_ID = "etank_inf",
        ENVIRONMENT = "pendulum_f1",
        ENERGY_TANK_INIT = 1000,
        ENERGY_AWARE = False
    ),
    test_id="inf"
)

train_and_test(
    dict(
        RUN_ID = "etank_inf",
        ENVIRONMENT = "pendulum_f0",
        ENERGY_TANK_INIT = 1000,
        ENERGY_AWARE = False  
    ),
    test_id="inf"
)

train_and_test(
    dict(
        RUN_ID = "etank_inf",
        ENVIRONMENT = "pendulum_f001",
        ENERGY_TANK_INIT = 1000,
        ENERGY_AWARE = False  
    ),
    test_id="inf"
)
  