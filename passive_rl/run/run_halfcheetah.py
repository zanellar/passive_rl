import os
import numpy as np
from mjrlenvs.scripts.train.trainer import run   
from passive_rl.scripts.pkgpaths import PkgPath 
from passive_rl.scripts.tester import TestRunEBud  
from passive_rl.run.halfcheetah_configs import Args  

n_eval_episodes = 100

run_results_file_path = os.path.join(PkgPath.OUT_TRAIN_FOLDER,"run_results.txt")  
with open(run_results_file_path, 'w') as file: 
    line = ""   
    file.write(line) 

def test(x=None, test_id=""):    
    if x is not None:
        Args.set(x)  
    tester = TestRunEBud(Args, test_id=test_id)   
    data = tester.eval_run(n_eval_episodes=n_eval_episodes, save=True)
    return np.amin(data["emin"])

def train_and_test(x, test_id=""): 
    Args.set(x) 
    run(Args) 
    min_emin = test(x, test_id=test_id) 
 
    with open(run_results_file_path, 'a') as file: 
        line = f"\n{Args.RUN_ID} {Args.ENVIRONMENT} {Args.ENV_EXPL.energy_tank_init} {min_emin}"   
        file.write(line)

    return min_emin


################################################################################################ 
  
min_emin = train_and_test(
    dict(
        RUN_ID = "etank_inf",
        ENVIRONMENT = "halfcheetah",
        ENERGY_TANK_INIT = 10000,
        ENERGY_AWARE = False , 
        ENERGY_TERMINATE = True, 
    ),
    test_id="inf"
)

min_etank_init = 10000 - min_emin

test(
    x = dict(
            RUN_ID = "etank_inf",
            ENVIRONMENT = "halfcheetah",
            ENERGY_TANK_INIT = min_etank_init,
            ENERGY_AWARE = False,
            ENERGY_TERMINATE = True,
        ),
    test_id="min"
)

train_and_test(
    dict(
        RUN_ID = "etank_min",
        ENVIRONMENT = "halfcheetah",
        ENERGY_TANK_INIT = min_etank_init,
        ENERGY_AWARE = False,
        ENERGY_TERMINATE = True,
    ),
    test_id="min"
)

#############################################

train_and_test(
    dict(
        RUN_ID = "etank_min08",
        ENVIRONMENT = "halfcheetah",
        ENERGY_TANK_INIT = min_etank_init*0.8,
        ENERGY_AWARE = False,
        ENERGY_TERMINATE = True,
    ),
    test_id="min08"
)

train_and_test(
    dict(
        RUN_ID = "etank_min06",
        ENVIRONMENT = "halfcheetah",
        ENERGY_TANK_INIT = min_etank_init*0.6,
        ENERGY_AWARE = False,
        ENERGY_TERMINATE = True,
    ),
    test_id="min06"
)

train_and_test(
    dict(
        RUN_ID = "etank_min03",
        ENVIRONMENT = "halfcheetah",
        ENERGY_TANK_INIT = min_etank_init*0.3,
        ENERGY_AWARE = False,
        ENERGY_TERMINATE = True,
    ),
    test_id="min03"
) 