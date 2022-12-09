from math import fabs
import os
import numpy as np
from mjrlenvs.scripts.train.trainer import run   
from passive_rl.scripts.pkgpaths import PkgPath 
from passive_rl.scripts.tester import TestRunEBud  
from passive_rl.run.pendulum.pendulum_configs import Args 
from passive_rl.envs.pendulum import PendulumEBud,PendulumEBudAw
from passive_rl.scripts.statistics import confidence_interval

 
def test(x=None, test_id="", n_eval_episodes = 100):    
    if x is not None:
        Args.set(x)  
    tester = TestRunEBud(Args, test_id=test_id)  
    data = tester.eval_run(n_eval_episodes=n_eval_episodes, save=True, render=False, cumulative_error=False)
    err_min, err_max = confidence_interval(data=data["etankmin"], width=99) 
    return err_min

###########################################################################

min_etankmin = 997.7762083274018 #input("min_etankmin = ")
min_etank_init = 1000 - min_etankmin
 
test(
    x = dict(
            RUN_ID = "etank_inf",
            ENVIRONMENT = "pendulum",
            ENERGY_TANK_INIT = 1000,
            ENERGY_AWARE = False,
            INIT_JOINT_CONFIG = "random",
            ENERGY_TERMINATE = True,
            REWARD_ID = 1
        ),
    test_id="inf", 
    n_eval_episodes = 100
)

test(
    dict(
        RUN_ID = "etank_inf",
        ENVIRONMENT = "pendulum",
        ENERGY_TANK_INIT = min_etank_init,
        ENERGY_AWARE = False,
        INIT_JOINT_CONFIG = "random" ,
        ENERGY_TERMINATE = True,
        REWARD_ID = 1
    ),
    test_id="min", 
    n_eval_episodes = 100
)


test(
    dict(
        RUN_ID = "etank_min",
        ENVIRONMENT = "pendulum",
        ENERGY_TANK_INIT = min_etank_init,
        ENERGY_AWARE = False,
        INIT_JOINT_CONFIG = "random" ,
        ENERGY_TERMINATE = True,
        REWARD_ID = 1
    ),
    test_id="min", 
    n_eval_episodes = 100
)

# ##################################################################################

# test(
#     dict(
#         RUN_ID = "etank_min06",
#         ENVIRONMENT = "pendulum",
#         ENERGY_TANK_INIT = min_etank_init*0.6,
#         ENERGY_AWARE = False,
#         INIT_JOINT_CONFIG = "random" ,
#         ENERGY_TERMINATE = True,
#         REWARD_ID = 1
#     ),
#     test_id="min06", 
#     n_eval_episodes = 10
# )

# test(
#     dict(
#         RUN_ID = "etank_min03",
#         ENVIRONMENT = "pendulum",
#         ENERGY_TANK_INIT = min_etank_init*0.3,
#         ENERGY_AWARE = False,
#         INIT_JOINT_CONFIG = "random"  ,
#         ENERGY_TERMINATE = True,
#         REWARD_ID = 1
#     ),
#     test_id="min03", 
#     n_eval_episodes = 10
# )

# test(
#     dict(
#         RUN_ID = "etank_min01",
#         ENVIRONMENT = "pendulum",
#         ENERGY_TANK_INIT = min_etank_init*0.1,
#         ENERGY_AWARE = False,
#         INIT_JOINT_CONFIG = "random" ,
#         ENERGY_TERMINATE = True,
#         REWARD_ID = 1
#     ),
#     test_id="min01", 
#     n_eval_episodes = 10
# )
 