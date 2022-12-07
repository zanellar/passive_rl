from math import fabs
import os
import numpy as np
from mjrlenvs.scripts.train.trainer import run   
from passive_rl.scripts.pkgpaths import PkgPath 
from passive_rl.scripts.tester import TestRunEBud  
from passive_rl.run.pendulum.pendulum_configs import Args 
from passive_rl.envs.pendulum import PendulumEBud,PendulumEBudAw

n_eval_episodes = 2


def test(x=None, test_id=""):    
    if x is not None:
        Args.set(x)  
    tester = TestRunEBud(Args, test_id=test_id)  
    data = tester.eval_run(n_eval_episodes=n_eval_episodes, save=True, render=False, cumulative_error=False)
    return np.amin(data["etankmin"])

###########################################################################

min_etank_init = 1000 - 994.9937158728585

test(
    x = dict(
            RUN_ID = "etank_inf",
            ENVIRONMENT = "pendulum",
            ENERGY_TANK_INIT = 1000,
            ENERGY_AWARE = False,
            INIT_JOINT_CONFIG =  "random"
        ),
    test_id="min"
)
 
test(
    dict(
        RUN_ID = "etank_min",
        ENVIRONMENT = "pendulum",
        ENERGY_TANK_INIT = min_etank_init,
        ENERGY_AWARE = False,
        INIT_JOINT_CONFIG =  "random" 
    ),
    test_id="min"
)