from math import fabs
import os
import numpy as np
from mjrlenvs.scripts.train.trainer import run   
from passive_rl.scripts.pkgpaths import PkgPath 
from passive_rl.scripts.tester import TestRunEBud  
from passive_rl.run.pendulum.pendulum_configs import Args 
from passive_rl.envs.pendulum import PendulumEBud,PendulumEBudAw
from passive_rl.scripts.statistics import confidence_interval

 
def test(model_id, x=None, test_id="", n_eval_episodes = 10):    
    if x is not None:
        Args.set(x)  
    tester = TestRunEBud(Args, test_id=test_id)  
    data = tester.eval_model(model_id=model_id,n_eval_episodes=n_eval_episodes, save=True, render=True, cumulative_error=False) 
    return np.amin(data["etankmin"])

###########################################################################

model_id = "SAC_4_0"
min_etankmin = 985.1162415689222 
min_etank_init = 1000 - min_etankmin

min_etankmin = test(
    model_id = model_id,
    x = dict(
            RUN_ID = "etank_inf",
            ENVIRONMENT = "pendulum",
            ENERGY_TANK_INIT = 1000,
            ENERGY_AWARE = False,
            INIT_JOINT_CONFIG = [-np.pi/2],
            ENERGY_TERMINATE = False,
            REWARD_ID = 1
        ),
    test_id="inf", 
    n_eval_episodes = 10
) 

print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
input(f"min_etankmin={min_etankmin}...[enter]")
  
test(
    model_id = model_id,
    x = dict(
            RUN_ID = "etank_inf",
            ENVIRONMENT = "pendulum",
            ENERGY_TANK_INIT = min_etank_init,
            ENERGY_AWARE = False,
            INIT_JOINT_CONFIG = [-np.pi/2],
            ENERGY_TERMINATE = False,
            REWARD_ID = 1
        ),
    test_id="min", 
    n_eval_episodes = 10
)


test(
    model_id = model_id,
    x = dict(
            RUN_ID = "etank_min",
            ENVIRONMENT = "pendulum",
            ENERGY_TANK_INIT = min_etank_init,
            ENERGY_AWARE = False,
            INIT_JOINT_CONFIG = [-np.pi/2],
            ENERGY_TERMINATE = False,
            REWARD_ID = 1
        ),
    test_id="min", 
    n_eval_episodes = 10
)

test(
    model_id = model_id,
    x = dict(
            RUN_ID = "etank_min",
            ENVIRONMENT = "pendulum",
            ENERGY_TANK_INIT = 1000,
            ENERGY_AWARE = False,
            INIT_JOINT_CONFIG = [-np.pi/2],
            ENERGY_TERMINATE = False,
            REWARD_ID = 1
        ),
    test_id="inf", 
    n_eval_episodes = 10
)

#####################################################################################

test(
    model_id = model_id,
    x = dict(
            RUN_ID = "etank_inf",
            ENVIRONMENT = "pendulum_wind",
            ENERGY_TANK_INIT = 1000,
            ENERGY_AWARE = False,
            INIT_JOINT_CONFIG = [-np.pi/2],
            ENERGY_TERMINATE = False,
            REWARD_ID = 1
        ),
    test_id="inf", 
    n_eval_episodes = 10
) 

test(
    model_id = model_id,
    x = dict(
            RUN_ID = "etank_inf",
            ENVIRONMENT = "pendulum_wind",
            ENERGY_TANK_INIT = min_etank_init,
            ENERGY_AWARE = False,
            INIT_JOINT_CONFIG = [-np.pi/2],
            ENERGY_TERMINATE = False,
            REWARD_ID = 1
        ),
    test_id="min", 
    n_eval_episodes = 10
)

print(min_etankmin)