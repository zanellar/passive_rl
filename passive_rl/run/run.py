from math import fabs
import os
import numpy as np
from mjrlenvs.scripts.train.trainer import run   
from passive_rl.scripts.pkgpaths import PkgPath 
from passive_rl.scripts.tester import TestRunEBud  
from passive_rl.run.pendulum_configs import Args 
from passive_rl.envs.pendulum.pendulum import PendulumEBud,PendulumEBudAw

run_results_file_path = os.path.join(PkgPath.OUT_TRAIN_FOLDER,"run_results.txt")  
with open(run_results_file_path, 'w') as file: 
    line = ""   
    file.write(line) 
   
def train_and_test(x): 
    Args.set(x) 
    run(Args) 
    tester = TestRunEBud(Args)  
    tester.eval_returns_run(n_eval_episodes=50, save=True)
    min_emin, max_emin, mean_emin, std_emin = tester.eval_emin_run(n_eval_episodes=50, save=True)
 
    with open(run_results_file_path, 'a') as file: 
        line = f"\n {Args.RUN_ID} {Args.ENVIRONMENT}, {Args.ENV_EXPL.energy_tank_init}, {min_emin}, {max_emin}, {mean_emin}, {std_emin}"   
        file.write(line)

    return min_emin


################################################################################################ 

train_and_test(dict(
    RUN_ID = "etank_1000",
    ENVIRONMENT = "pendulum_f1",
    ENERGY_TANK_INIT = 1000,
    ENERGY_AWARE = False
))

train_and_test(dict(
    RUN_ID = "etank_1000",
    ENVIRONMENT = "pendulum_f0",
    ENERGY_TANK_INIT = 1000,
    ENERGY_AWARE = False  
))

min_emin = train_and_test(dict(
    RUN_ID = "etank_1000",
    ENVIRONMENT = "pendulum_f001",
    ENERGY_TANK_INIT = 1000,
    ENERGY_AWARE = False  
))

train_and_test(dict(
    RUN_ID = "etank_min",
    ENVIRONMENT = "pendulum_f001",
    ENERGY_TANK_INIT = min_emin,
    ENERGY_AWARE = False  
))

train_and_test(dict(
    RUN_ID = "etank_min08",
    ENVIRONMENT = "pendulum_f001",
    ENERGY_TANK_INIT = min_emin*0.8,
    ENERGY_AWARE = False  
))

train_and_test(dict(
    RUN_ID = "etank_min06",
    ENVIRONMENT = "pendulum_f001",
    ENERGY_TANK_INIT = min_emin*0.6,
    ENERGY_AWARE = False  
))

train_and_test(dict(
    RUN_ID = "etank_min03",
    ENVIRONMENT = "pendulum_f001",
    ENERGY_TANK_INIT = min_emin*0.3,
    ENERGY_AWARE = False  
))

train_and_test(dict(
    RUN_ID = "etank_min08_eaw",
    ENVIRONMENT = "pendulum_f001",
    ENERGY_TANK_INIT = min_emin*0.8,
    ENERGY_AWARE = True  
))

train_and_test(dict(
    RUN_ID = "etank_min06_eaw",
    ENVIRONMENT = "pendulum_f001",
    ENERGY_TANK_INIT = min_emin*0.6,
    ENERGY_AWARE = True  
))

train_and_test(dict(
    RUN_ID = "etank_min03_eaw",
    ENVIRONMENT = "pendulum_f001",
    ENERGY_TANK_INIT = min_emin*0.3,
    ENERGY_AWARE = True  
))
 