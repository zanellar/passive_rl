import os
import numpy as np
from mjrlenvs.scripts.train.trainer import run   
from passive_rl.scripts.pkgpaths import PkgPath 
from passive_rl.scripts.tester import TestRunEBud  
from passive_rl.run.pendulum_configs import Args 

run_results_file_path = os.path.join(PkgPath.OUT_TRAIN_FOLDER,"run_results.txt")  
with open(run_results_file_path, 'w') as file: 
    line = ""   
    file.write(line)

################################################################################################
Args.RUN_ID = "etank_1000" 
Args.ENVIRONMENT = "pendulum_f1"
Args.ENV_EXPL.energy_tank_init = 1000
Args.ENV_EVAL.energy_tank_init = Args.ENV_EXPL.energy_tank_init 
print(f"{'@'*100}\n{'@'*100}\n{'@'*100}\n Training {Args.RUN_ID} {Args.ENVIRONMENT}")
run(Args)
print(f"{'@'*100}\n{'@'*100}\n{'@'*100}\n Testing {Args.RUN_ID} {Args.ENVIRONMENT}")
tester = TestRunEBud(Args)  
returns = tester.eval_returns_run(n_eval_episodes=50, save=True)
min_emin_ref_f1, max_emin_ref_f1, mean_emin_ref_f1, std_emin_ref_f1 = tester.eval_emin_run(n_eval_episodes=50, save=True)

  
with open(run_results_file_path, 'a') as file: 
    line = f"\n{Args.ENV_EXPL.energy_tank_init}, {min_emin_ref_f1}, {max_emin_ref_f1}, {mean_emin_ref_f1}, {std_emin_ref_f1}"   
    file.write(line)

################################################################################################
Args.RUN_ID = "etank_1000" 
Args.ENVIRONMENT = "pendulum_f0" 
Args.ENV_EXPL.energy_tank_init = 1000
Args.ENV_EVAL.energy_tank_init = Args.ENV_EXPL.energy_tank_init 
print(f"{'@'*100}\n{'@'*100}\n{'@'*100}\n Training {Args.RUN_ID} {Args.ENVIRONMENT}")
run(Args)
print(f"{'@'*100}\n{'@'*100}\n{'@'*100}\n Testing {Args.RUN_ID} {Args.ENVIRONMENT}")
tester = TestRunEBud(Args)  
returns = tester.eval_returns_run(n_eval_episodes=50, save=True)
min_emin_ref_f01, max_emin_ref_f01, mean_emin_ref_f01, std_emin_ref_f01 = tester.eval_emin_run(n_eval_episodes=50, save=True)

  
with open(run_results_file_path, 'a') as file: 
    line = f"\n{Args.ENV_EXPL.energy_tank_init}, {min_emin_ref_f01}, {max_emin_ref_f01}, {mean_emin_ref_f01}, {std_emin_ref_f01}"   
    file.write(line)

################################################################################################
################################################################################################
################################################################################################
################################################################################################
################################################################################################

Args.RUN_ID = "etank_1000" 
Args.ENVIRONMENT = "pendulum_f001" 
Args.ENV_EXPL.energy_tank_init = 1000
Args.ENV_EVAL.energy_tank_init = Args.ENV_EXPL.energy_tank_init 
print(f"{'@'*100}\n{'@'*100}\n{'@'*100}\n Training {Args.RUN_ID} {Args.ENVIRONMENT}")
run(Args)
print(f"{'@'*100}\n{'@'*100}\n{'@'*100}\n Testing {Args.RUN_ID} {Args.ENVIRONMENT}")
tester = TestRunEBud(Args)  
returns = tester.eval_returns_run(n_eval_episodes=50, save=True)
min_emin, max_emin, mean_emin, std_emin = tester.eval_emin_run(n_eval_episodes=50, save=True)

  
with open(run_results_file_path, 'a') as file: 
    line = f"\n{Args.ENV_EXPL.energy_tank_init}, {min_emin}, {max_emin}, {mean_emin}, {std_emin}"   
    file.write(line)

min_emin_ref_f001 = Args.ENV_EXPL.energy_tank_init - min_emin 

################################################################################################
Args.RUN_ID = "etank_minemin" 
Args.ENVIRONMENT = "pendulum_f001"  
Args.ENV_EXPL.energy_tank_init = min_emin_ref_f001 
Args.ENV_EVAL.energy_tank_init = Args.ENV_EXPL.energy_tank_init 
print(f"{'@'*100}\n{'@'*100}\n{'@'*100}\n Training {Args.RUN_ID} {Args.ENVIRONMENT}")
run(Args)
print(f"{'@'*100}\n{'@'*100}\n{'@'*100}\n Testing {Args.RUN_ID} {Args.ENVIRONMENT}")
tester = TestRunEBud(Args)  
returns = tester.eval_returns_run(n_eval_episodes=50, save=True)
min_emin, max_emin, mean_emin, std_emin = tester.eval_emin_run(n_eval_episodes=50, save=True)

  
with open(run_results_file_path, 'a') as file: 
    line = f"\n{Args.ENV_EXPL.energy_tank_init}, {min_emin}, {max_emin}, {mean_emin}, {std_emin}"   
    file.write(line)

################################################################################################
Args.RUN_ID = "etank_minemin08" 
Args.ENVIRONMENT = "pendulum_f001"  
Args.ENV_EXPL.energy_tank_init = min_emin_ref_f001*0.8
Args.ENV_EVAL.energy_tank_init = Args.ENV_EXPL.energy_tank_init 
print(f"{'@'*100}\n{'@'*100}\n{'@'*100}\n Training {Args.RUN_ID} {Args.ENVIRONMENT}")
run(Args)
print(f"{'@'*100}\n{'@'*100}\n{'@'*100}\n Testing {Args.RUN_ID} {Args.ENVIRONMENT}")
tester = TestRunEBud(Args)  
returns = tester.eval_returns_run(n_eval_episodes=50, save=True)
min_emin, max_emin, mean_emin, std_emin = tester.eval_emin_run(n_eval_episodes=50, save=True)

  
with open(run_results_file_path, 'a') as file: 
    line = f"\n{Args.ENV_EXPL.energy_tank_init}, {min_emin}, {max_emin}, {mean_emin}, {std_emin}"   
    file.write(line)

################################################################################################
Args.RUN_ID = "etank_minemin06" 
Args.ENVIRONMENT = "pendulum_f001" 
Args.ENV_EXPL.energy_tank_init = min_emin_ref_f001*0.6
Args.ENV_EVAL.energy_tank_init = Args.ENV_EXPL.energy_tank_init 
print(f"{'@'*100}\n{'@'*100}\n{'@'*100}\n Training {Args.RUN_ID} {Args.ENVIRONMENT}")
run(Args)
print(f"{'@'*100}\n{'@'*100}\n{'@'*100}\n Testing {Args.RUN_ID} {Args.ENVIRONMENT}")
tester = TestRunEBud(Args)  
returns = tester.eval_returns_run(n_eval_episodes=50, save=True)
min_emin, max_emin, mean_emin, std_emin = tester.eval_emin_run(n_eval_episodes=50, save=True)

  
with open(run_results_file_path, 'a') as file: 
    line = f"\n{Args.ENV_EXPL.energy_tank_init}, {min_emin}, {max_emin}, {mean_emin}, {std_emin}"   
    file.write(line)

################################################################################################
Args.RUN_ID = "etank_minemin03" 
Args.ENVIRONMENT = "pendulum_f001" 
Args.ENV_EXPL.energy_tank_init = min_emin_ref_f001*0.3
Args.ENV_EVAL.energy_tank_init = Args.ENV_EXPL.energy_tank_init 
print(f"{'@'*100}\n{'@'*100}\n{'@'*100}\n Training {Args.RUN_ID} {Args.ENVIRONMENT}")
run(Args)
print(f"{'@'*100}\n{'@'*100}\n{'@'*100}\n Testing {Args.RUN_ID} {Args.ENVIRONMENT}")
tester = TestRunEBud(Args)  
returns = tester.eval_returns_run(n_eval_episodes=50, save=True)
min_emin, max_emin, mean_emin, std_emin = tester.eval_emin_run(n_eval_episodes=50, save=True)

  
with open(run_results_file_path, 'a') as file: 
    line = f"\n{Args.ENV_EXPL.energy_tank_init}, {min_emin}, {max_emin}, {mean_emin}, {std_emin}"   
    file.write(line)

################################################################################################
################################################################################################
################################################################################################
################################################################################################
################################################################################################

from passive_rl.envs.pendulum.pendulum import PendulumEBudAw
 
Args.RUN_ID = "etank_minemin_eaw" 
Args.ENVIRONMENT = "pendulum_f001"  
Args.ENERGY_TANK_INIT = min_emin_ref_f001  
Args.NORMALIZE_ENV["clip_obs"] = np.inf
Args.ENV_EXPL = PendulumEBudAw(
                max_episode_length = Args.EXPL_EPISODE_HORIZON,  
                energy_tank_init = Args.ENERGY_TANK_INIT, # initial energy in the tank
                energy_tank_threshold = 0, # minimum energy in the tank  
                init_joint_config = "random",
                folder_path = PkgPath.ENV_DESC_FOLDER,
                env_name = Args.ENVIRONMENT
            ) 
Args.ENV_EVAL = PendulumEBudAw(
                max_episode_length=Args.EVAL_EPISODE_HORIZON,  
                energy_tank_init = Args.ENERGY_TANK_INIT, # initial energy in the tank
                energy_tank_threshold = 0, # minimum energy in the tank  
                init_joint_config = "random",
                folder_path = PkgPath.ENV_DESC_FOLDER,
                env_name = Args.ENVIRONMENT
            )   
print(f"{'@'*100}\n{'@'*100}\n{'@'*100}\n Training {Args.RUN_ID} {Args.ENVIRONMENT}")
run(Args)
print(f"{'@'*100}\n{'@'*100}\n{'@'*100}\n Testing {Args.RUN_ID} {Args.ENVIRONMENT}")
tester = TestRunEBud(Args)  
returns = tester.eval_returns_run(n_eval_episodes=50, save=True)
min_emin, max_emin, mean_emin, std_emin = tester.eval_emin_run(n_eval_episodes=50, save=True)

  
with open(run_results_file_path, 'a') as file: 
    line = f"\n{Args.ENV_EXPL.energy_tank_init}, {min_emin}, {max_emin}, {mean_emin}, {std_emin}"   
    file.write(line)

################################################################################################
Args.RUN_ID = "etank_minemin08_eaw" 
Args.ENVIRONMENT = "pendulum_f001"  
Args.ENERGY_TANK_INIT = min_emin_ref_f001*0.8
Args.NORMALIZE_ENV["clip_obs"] = np.inf
Args.ENV_EXPL = PendulumEBudAw(
                max_episode_length = Args.EXPL_EPISODE_HORIZON,  
                energy_tank_init = Args.ENERGY_TANK_INIT, # initial energy in the tank
                energy_tank_threshold = 0, # minimum energy in the tank  
                init_joint_config = "random",
                folder_path = PkgPath.ENV_DESC_FOLDER,
                env_name = Args.ENVIRONMENT
            ) 
Args.ENV_EVAL = PendulumEBudAw(
                max_episode_length=Args.EVAL_EPISODE_HORIZON,  
                energy_tank_init = Args.ENERGY_TANK_INIT, # initial energy in the tank
                energy_tank_threshold = 0, # minimum energy in the tank  
                init_joint_config = "random",
                folder_path = PkgPath.ENV_DESC_FOLDER,
                env_name = Args.ENVIRONMENT
            )   
print(f"{'@'*100}\n{'@'*100}\n{'@'*100}\n Training {Args.RUN_ID} {Args.ENVIRONMENT}")
run(Args)
print(f"{'@'*100}\n{'@'*100}\n{'@'*100}\n Testing {Args.RUN_ID} {Args.ENVIRONMENT}")
tester = TestRunEBud(Args)  
returns = tester.eval_returns_run(n_eval_episodes=50, save=True)
min_emin, max_emin, mean_emin, std_emin = tester.eval_emin_run(n_eval_episodes=50, save=True)

  
with open(run_results_file_path, 'a') as file: 
    line = f"\n{Args.ENV_EXPL.energy_tank_init}, {min_emin}, {max_emin}, {mean_emin}, {std_emin}"   
    file.write(line)

################################################################################################
Args.RUN_ID = "etank_minemin06_eaw" 
Args.ENVIRONMENT = "pendulum_f001" 
Args.ENERGY_TANK_INIT = min_emin_ref_f001*0.6  
Args.NORMALIZE_ENV["clip_obs"] = np.inf
Args.ENV_EXPL = PendulumEBudAw(
                max_episode_length = Args.EXPL_EPISODE_HORIZON,  
                energy_tank_init = Args.ENERGY_TANK_INIT, # initial energy in the tank
                energy_tank_threshold = 0, # minimum energy in the tank  
                init_joint_config = "random",
                folder_path = PkgPath.ENV_DESC_FOLDER,
                env_name = Args.ENVIRONMENT
            ) 
Args.ENV_EVAL = PendulumEBudAw(
                max_episode_length=Args.EVAL_EPISODE_HORIZON,  
                energy_tank_init = Args.ENERGY_TANK_INIT, # initial energy in the tank
                energy_tank_threshold = 0, # minimum energy in the tank  
                init_joint_config = "random",
                folder_path = PkgPath.ENV_DESC_FOLDER,
                env_name = Args.ENVIRONMENT
            )    
print(f"{'@'*100}\n{'@'*100}\n{'@'*100}\n Training {Args.RUN_ID} {Args.ENVIRONMENT}")
run(Args)
print(f"{'@'*100}\n{'@'*100}\n{'@'*100}\n Testing {Args.RUN_ID} {Args.ENVIRONMENT}")
tester = TestRunEBud(Args)  
returns = tester.eval_returns_run(n_eval_episodes=50, save=True)
min_emin, max_emin, mean_emin, std_emin = tester.eval_emin_run(n_eval_episodes=50, save=True)

with open(run_results_file_path, 'a') as file: 
    line = f"\n{Args.ENV_EXPL.energy_tank_init}, {min_emin}, {max_emin}, {mean_emin}, {std_emin}"   
    file.write(line)


################################################################################################
Args.RUN_ID = "etank_minemin03_eaw" 
Args.ENVIRONMENT = "pendulum_f001" 
Args.ENERGY_TANK_INIT = min_emin_ref_f001*0.3 
Args.NORMALIZE_ENV["clip_obs"] = np.inf
Args.ENV_EXPL = PendulumEBudAw(
                max_episode_length = Args.EXPL_EPISODE_HORIZON,  
                energy_tank_init = Args.ENERGY_TANK_INIT, # initial energy in the tank
                energy_tank_threshold = 0, # minimum energy in the tank  
                init_joint_config = "random",
                folder_path = PkgPath.ENV_DESC_FOLDER,
                env_name = Args.ENVIRONMENT
            ) 
Args.ENV_EVAL = PendulumEBudAw(
                max_episode_length=Args.EVAL_EPISODE_HORIZON,  
                energy_tank_init = Args.ENERGY_TANK_INIT, # initial energy in the tank
                energy_tank_threshold = 0, # minimum energy in the tank  
                init_joint_config = "random",
                folder_path = PkgPath.ENV_DESC_FOLDER,
                env_name = Args.ENVIRONMENT
            )    
print(f"{'@'*100}\n{'@'*100}\n{'@'*100}\n Training {Args.RUN_ID} {Args.ENVIRONMENT}")
run(Args)
print(f"{'@'*100}\n{'@'*100}\n{'@'*100}\n Testing {Args.RUN_ID} {Args.ENVIRONMENT}")
tester = TestRunEBud(Args)  
returns = tester.eval_returns_run(n_eval_episodes=50, save=True)
min_emin, max_emin, mean_emin, std_emin = tester.eval_emin_run(n_eval_episodes=50, save=True)

with open(run_results_file_path, 'a') as file: 
    line = f"\n{Args.ENV_EXPL.energy_tank_init}, {min_emin}, {max_emin}, {mean_emin}, {std_emin}"   
    file.write(line)