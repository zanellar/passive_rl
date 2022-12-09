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
    data = tester.eval_run(n_eval_episodes=n_eval_episodes, save=True)
    err_min, err_max = confidence_interval(data=data["etankmin"], width=100) 
    return err_min
 

################################################################################################ 

  
min_etankmin = test(
    x = dict(
            RUN_ID = "etank_inf",
            ENVIRONMENT = "halfcheetah",
            ENERGY_TANK_INIT = 10000,
            ENERGY_AWARE = False,
            ENERGY_TERMINATE = True,
        ),
    test_id="inf", 
    n_eval_episodes = 10
)

min_etank_init = 10000 - min_etankmin
print(min_etank_init)
exit() 

test(
    x = dict(
            RUN_ID = "etank_inf",
            ENVIRONMENT = "halfcheetah",
            ENERGY_TANK_INIT = min_etank_init,
            ENERGY_AWARE = False,
            ENERGY_TERMINATE = True,
        ),
    test_id="min", 
    n_eval_episodes = 10
)

test(
    dict(
        RUN_ID = "etank_min",
        ENVIRONMENT = "halfcheetah",
        ENERGY_TANK_INIT = min_etank_init,
        ENERGY_AWARE = False,
        ENERGY_TERMINATE = True,
    ),
    test_id="min", 
    n_eval_episodes = 10
)
 