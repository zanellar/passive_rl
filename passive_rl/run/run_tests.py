from math import fabs
import os
import numpy as np
from mjrlenvs.scripts.train.trainer import run   
from passive_rl.scripts.pkgpaths import PkgPath 
from passive_rl.scripts.tester import TestRunEBud  
from passive_rl.run.pendulum_configs import Args 
from passive_rl.envs.pendulum.pendulum import PendulumEBud,PendulumEBudAw

n_eval_episodes = 50


def test(x=None, test_id=""):    
    if x is not None:
        Args.set(x)  
    tester = TestRunEBud(Args, test_id=test_id)  
    tester.eval_returns_run(n_eval_episodes=n_eval_episodes, save=True)
    eval_ebud_data = tester.eval_ebud_run(n_eval_episodes=n_eval_episodes, save=True)
    return np.amin(eval_ebud_data["emin"])



################################################################################################ 
 
  
test(
    dict(
        RUN_ID = "frcomp",
        ENVIRONMENT = "pendulum_f1",
        ENERGY_TANK_INIT = 1000,
        ENERGY_AWARE = False,
        INIT_JOINT_CONFIG = [-np.pi/2]
    ),
    test_id="f0"
)

  
# test(
#     dict(
#         RUN_ID = "frcomp_f0",
#         ENVIRONMENT = "pendulum_f1",
#         ENERGY_TANK_INIT = 1000,
#         ENERGY_AWARE = False,
#         INIT_JOINT_CONFIG = [-np.pi/2]
#     ),
#     test_id="f0inf1"
# )

test(
    dict(
        RUN_ID = "frcomp",
        ENVIRONMENT = "pendulum_f0",
        ENERGY_TANK_INIT = 1000,
        ENERGY_AWARE = False,
        INIT_JOINT_CONFIG = [-np.pi/2]
    ) ,
    test_id="f1"
) 
 
# test(
#     dict(
#         RUN_ID = "frcomp_f1",
#         ENVIRONMENT = "pendulum_f0",
#         ENERGY_TANK_INIT = 1000,
#         ENERGY_AWARE = False,
#         INIT_JOINT_CONFIG = [-np.pi/2]
#     ) ,
#     test_id="f1inf0"
# ) 
 
min_etank_init = 1000 - 994.9937158728585

test(
    x = dict(
            RUN_ID = "etank_inf",
            ENVIRONMENT = "pendulum_f001",
            ENERGY_TANK_INIT = min_etank_init,
            ENERGY_AWARE = False,
            INIT_JOINT_CONFIG =  "random"
        ),
    test_id="min"
)

test(
    dict(
        RUN_ID = "etank_min",
        ENVIRONMENT = "pendulum_f001",
        ENERGY_TANK_INIT = min_etank_init,
        ENERGY_AWARE = False,
        INIT_JOINT_CONFIG =  "random" 
    ),
    test_id="min"
)

test(
    dict(
        RUN_ID = "etank_min08",
        ENVIRONMENT = "pendulum_f001",
        ENERGY_TANK_INIT = min_etank_init*0.8,
        ENERGY_AWARE = False,
        INIT_JOINT_CONFIG =  "random"
    ),
    test_id="min08"
)

test(
    dict(
        RUN_ID = "etank_min06",
        ENVIRONMENT = "pendulum_f001",
        ENERGY_TANK_INIT = min_etank_init*0.6,
        ENERGY_AWARE = False,
        INIT_JOINT_CONFIG =  "random" 
    ),
    test_id="min06"
)

test(
    dict(
        RUN_ID = "etank_min03",
        ENVIRONMENT = "pendulum_f001",
        ENERGY_TANK_INIT = min_etank_init*0.3,
        ENERGY_AWARE = False,
        INIT_JOINT_CONFIG =  "random"
    ),
    test_id="min03"
)

 #############################################

test(
    dict(
        RUN_ID = "etank_min08eaw",
        ENVIRONMENT = "pendulum_f001",
        ENERGY_TANK_INIT = min_etank_init*0.8,
        ENERGY_AWARE = True,
        INIT_JOINT_CONFIG =  "random"  
    ),
    test_id="min08eaw"
)

test(
    dict(
        RUN_ID = "etank_min06eaw",
        ENVIRONMENT = "pendulum_f001",
        ENERGY_TANK_INIT = min_etank_init*0.6,
        ENERGY_AWARE = True,
        INIT_JOINT_CONFIG =  "random"
    ),
    test_id="min06eaw"
)

test(
    dict(
        RUN_ID = "etank_min03eaw",
        ENVIRONMENT = "pendulum_f001",
        ENERGY_TANK_INIT = min_etank_init*0.3,
        ENERGY_AWARE = True,
        INIT_JOINT_CONFIG =  "random" 
    ),
    test_id="min03eaw"
)
 