import os
import numpy as np
from mjrlenvs.scripts.train.trainer import run   
from passive_rl.scripts.pkgpaths import PkgPath 
from passive_rl.scripts.tester import TestRunEBud  
from passive_rl.run.halfcheetah.halfcheetah_configs import Args  
from passive_rl.scripts.statistics import confidence_interval



run_results_file_path = os.path.join(PkgPath.OUT_TRAIN_FOLDER,"run_results.txt")  
with open(run_results_file_path, 'w') as file: 
    line = ""   
    file.write(line) 

def test(x=None, test_id="", n_eval_episodes = 10):    
    if x is not None:
        Args.set(x)  
    tester = TestRunEBud(Args, test_id=test_id)   
    data = tester.eval_run(n_eval_episodes=n_eval_episodes, save=True, render=False  ) 
    return  np.amin(data["etankmin"])
 

################################################################################################ 

  
min_etankmin = 9344.553728492554 
min_etank_init = 10000 - min_etankmin 
 

test(
    x = dict(
            RUN_ID = "etank_inf",
            ENVIRONMENT = "halfcheetah_wind",
            ENERGY_TANK_INIT = 1000,
            ENERGY_AWARE = False,
            INIT_JOINT_CONFIG = "random",
            ENERGY_TERMINATE = False,
            REWARD_ID = 1
        ),
    test_id="inf", 
    n_eval_episodes = 10
) 

test(
    dict(
        RUN_ID = "etank_inf",
        ENVIRONMENT = "halfcheetah_wind",
        ENERGY_TANK_INIT = min_etank_init,
        ENERGY_AWARE = False,
        INIT_JOINT_CONFIG = "random",
        ENERGY_TERMINATE = False,
        REWARD_ID = 1
    ),
    test_id="min", 
    n_eval_episodes = 10
)
 