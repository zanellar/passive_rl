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
    data = tester.eval_run(n_eval_episodes=n_eval_episodes, save=save, render=False)
    return np.amin(data["etankmin"])

def train(x ): 
    Args.set(x) 
    run(Args) 
      
############################################################################################### 
  
# x_inf = dict(
#             RUN_ID = "etank_inf",
#             ENVIRONMENT = "pendulum",
#             ENERGY_TANK_INIT = 1000,
#             ENERGY_AWARE = False,
#             INIT_JOINT_CONFIG =  [-np.pi/2],
#             ENERGY_TERMINATE = True,
#             REWARD_ID = 1
#         )

# # train(x_inf) 

# ### -------------------------------------------------------

# x_inf.update(ENERGY_TERMINATE = False) 

# min_etankmin = test(
#     x = x_inf,
#     test_id = "inf", 
#     n_eval_episodes = 10,
#     save = True
# )  
# print(min_etankmin)
# min_etankmin = 996.7127971100477
# min_etank_init = (1000 - min_etankmin) #*0.6

# x_inf.update(ENERGY_TANK_INIT = min_etank_init) 

# test(
#     x = x_inf,
#     test_id="min", 
#     n_eval_episodes = 10,
#     save = True
# )

# ################################################################################################ 

# x_min = dict(
#         RUN_ID = "etank_min",
#         ENVIRONMENT = "pendulum",
#         ENERGY_TANK_INIT = min_etank_init,
#         ENERGY_AWARE = False,
#         INIT_JOINT_CONFIG =  [-np.pi/2],
#         ENERGY_TERMINATE = True,
#         REWARD_ID = 1
#     ) 

# # train(x_min)

# ### -------------------------------------------------------
 
# x_min.update(ENERGY_TERMINATE = False) 

# test(
#     x = x_min,
#     test_id = "min", 
#     n_eval_episodes = 10,
#     save = True
# )

# ### -------------------------------------------------------
 
# x_min.update(ENERGY_TANK_INIT = 1000) 

# test(
#     x = x_min,
#     test_id = "inf", 
#     n_eval_episodes = 10,
#     save = True
# )



# ############################################################################################### 
# ############################################################################################### 
# ############################################################################################### 

# # from distutils.dir_util import copy_tree
 
# # from_directory = os.path.join(PkgPath.OUT_TRAIN_FOLDER,"pendulum")   
# # to_directory = os.path.join(PkgPath.OUT_TRAIN_FOLDER,"pendulum_wind")   
# # if not os.path.exists(to_directory):
# #     os.makedirs(to_directory) 
# # copy_tree(from_directory, to_directory)


# x_wind_inf = dict(
#         RUN_ID = "etank_inf",
#         ENVIRONMENT = "pendulum_wind",
#         ENERGY_TANK_INIT = 1000,
#         ENERGY_AWARE = False,
#         INIT_JOINT_CONFIG =  [-np.pi/2],
#         ENERGY_TERMINATE = False,
#         REWARD_ID = 1
#     ) 
 
# ### ------------------------------------------------------- 

# test(
#     x = x_wind_inf,
#     test_id = "inf", 
#     n_eval_episodes = 10,
#     save = True
# )

# x_wind_inf.update(ENERGY_TANK_INIT = min_etank_init) 
 
# test(
#     x = x_wind_inf,
#     test_id = "min", 
#     n_eval_episodes = 10,
#     save = True
# )


############################################################################################### 
############################################################################################### 
############################################################################################### 
  
x_inf_ngr = dict(
            RUN_ID = "etank_inf",
            ENVIRONMENT = "pendulum",
            ENERGY_TANK_INIT = 1000,
            ENERGY_AWARE = False,
            INIT_JOINT_CONFIG =  [-np.pi/2],
            ENERGY_TERMINATE = False,
            REWARD_ID = 0
        )

train(x_inf_ngr) 

### -------------------------------------------------------
   
min_etankmin = test(
    x = x_inf_ngr,
    test_id = "inf", 
    n_eval_episodes = 10,
    save = True
)   
min_etank_init = (1000 - min_etankmin)  

x_inf_ngr.update(ENERGY_TANK_INIT = min_etank_init) 

test(
    x = x_inf_ngr,
    test_id="min", 
    n_eval_episodes = 10,
    save = True
)

################################################################################################ 

x_min_ngr = dict(
        RUN_ID = "etank_min",
        ENVIRONMENT = "pendulum",
        ENERGY_TANK_INIT = min_etank_init,
        ENERGY_AWARE = True,
        INIT_JOINT_CONFIG =  [-np.pi/2],
        ENERGY_TERMINATE = False,
        REWARD_ID = 0
    ) 

train(x_min_ngr)
 
test(
    x = x_min_ngr,
    test_id="min_eaw", 
    n_eval_episodes = 10,
    save = True
)

x_min_ngr.update(ENERGY_AWARE = False) 

train(x_min_ngr)

test(
    x = x_min_ngr,
    test_id="min_neaw", 
    n_eval_episodes = 10,
    save = True
)