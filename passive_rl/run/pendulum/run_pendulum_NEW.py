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

def test(x=None, test_id="", n_eval_episodes = 10, save=False):    
    if x is not None:
        Args.set(x)  
    tester = TestRunEBud(Args, test_id=test_id)   
    data = tester.eval_run(n_eval_episodes=n_eval_episodes, save=save)
    return np.amin(data["etankmin"])

def train(x ): 
    Args.set(x) 
    run(Args) 
      
################################################################################################ 
  
x_inf = dict(
            RUN_ID = "etank_inf",
            ENVIRONMENT = "pendulum",
            ENERGY_TANK_INIT = 1000,
            ENERGY_AWARE = False,
            INIT_JOINT_CONFIG =  "random",
            ENERGY_TERMINATE = True,
            REWARD_ID = 1
        )

train(x_inf) 

### -------------------------------------------------------

x_inf.update(ENERGY_TERMINATE = False) 

min_etankmin = test(
    x = x_inf,
    test_id = "inf", 
    n_eval_episodes = 100,
    save = True
)

min_etank_init = 1000 - min_etankmin 

# x_inf.update(ENERGY_TANK_INIT = min_etank_init) 

# test(
#     x_inf,
#     test_id="min", 
#     n_eval_episodes = 10
# )

################################################################################################ 

x_min = dict(
        RUN_ID = "etank_min",
        ENVIRONMENT = "pendulum",
        ENERGY_TANK_INIT = min_etank_init,
        ENERGY_AWARE = False,
        INIT_JOINT_CONFIG =  "random" ,
        ENERGY_TERMINATE = True,
        REWARD_ID = 1
    ) 

train(x_min)

# ### -------------------------------------------------------
 
# x_min.update(ENERGY_TERMINATE = False) 

# test(
#     x = x_min,
#     test_id = "min", 
#     n_eval_episodes = 10
# )


################################################################################################ 

from distutils.dir_util import copy_tree
 
from_directory = os.path.join(PkgPath.OUT_TRAIN_FOLDER,"pendulum")   
to_directory = os.path.join(PkgPath.OUT_TRAIN_FOLDER,"pendulum_wind")   
if not os.path.exists(to_directory):
    os.makedirs(to_directory) 
copy_tree(from_directory, to_directory)
