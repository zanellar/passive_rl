from math import fabs
import os
import numpy as np
from mjrlenvs.scripts.train.trainer import run   
from passive_rl.scripts.pkgpaths import PkgPath 
from passive_rl.scripts.tester import TestRunEBud  
from passive_rl.run.pendulum.pendulum_configs import Args 
from passive_rl.envs.pendulum import PendulumEBud,PendulumEBudAw
from passive_rl.scripts.statistics import confidence_interval



run_results_file_path = os.path.join(PkgPath.OUT_TRAIN_FOLDER,"run_results.txt")  
with open(run_results_file_path, 'w') as file: 
    line = ""   
    file.write(line) 

def test(x=None, test_id="", n_eval_episodes = 100):    
    if x is not None:
        Args.set(x)  
    tester = TestRunEBud(Args, test_id=test_id)   
    data = tester.eval_run(n_eval_episodes=n_eval_episodes, save=True)
    err_min, err_max = confidence_interval(data=data["etankmin"], width=99) 
    return err_min 

def train_and_test(x, test_id="", n_eval_episodes = 100): 
    Args.set(x) 
    run(Args) 
    min_etankmin = test(x, test_id=test_id,n_eval_episodes=n_eval_episodes) 
 
    with open(run_results_file_path, 'a') as file: 
        line = f"\n{Args.RUN_ID} {Args.ENVIRONMENT} {Args.ENV_EXPL.energy_tank_init} {min_etankmin}"   
        file.write(line)

    return min_etankmin


################################################################################################ 
   
min_etank_init = 1000 - 997.7762083274018
 
train_and_test(
    dict(
        RUN_ID = "etank_min",
        ENVIRONMENT = "pendulum",
        ENERGY_TANK_INIT = min_etank_init,
        ENERGY_AWARE = False,
        INIT_JOINT_CONFIG =  "random" ,
        ENERGY_TERMINATE = True,
        REWARD_ID = 1
    ),
    test_id="min", 
    n_eval_episodes = 100
)

# #############################################

# train_and_test(
#     dict(
#         RUN_ID = "etank_min08",
#         ENVIRONMENT = "pendulum",
#         ENERGY_TANK_INIT = min_etank_init*0.6,
#         ENERGY_AWARE = False,
#         INIT_JOINT_CONFIG =  "random",
#         ENERGY_TERMINATE = True,
#         REWARD_ID = 1
#     ),
#     test_id="min08", 
#     n_eval_episodes = 10
# )

# train_and_test(
#     dict(
#         RUN_ID = "etank_min06",
#         ENVIRONMENT = "pendulum",
#         ENERGY_TANK_INIT = min_etank_init*0.3,
#         ENERGY_AWARE = False,
#         INIT_JOINT_CONFIG =  "random" ,
#         ENERGY_TERMINATE = True,
#         REWARD_ID = 1
#     ),
#     test_id="min06", 
#     n_eval_episodes = 10
# )

# train_and_test(
#     dict(
#         RUN_ID = "etank_min03",
#         ENVIRONMENT = "pendulum",
#         ENERGY_TANK_INIT = min_etank_init*0.1,
#         ENERGY_AWARE = False,
#         INIT_JOINT_CONFIG =  "random",
#         ENERGY_TERMINATE = True,
#         REWARD_ID = 1
#     ),
#     test_id="min03", 
#     n_eval_episodes = 10
# ) 