from mjrlenvs.scripts.train.trainer import run   
from passive_rl.scripts.pkgpaths import PkgPath 
from passive_rl.scripts.tester import TestRunEBud 

from passive_rl.run.pendulum_configs import Args 


################################################################################################
Args.RUN_ID = "etank_1000" 
Args.ENVIRONMENT = "pendulum_f1"
Args.ENERGY_TANK_INIT = 1000 
print(f"{'@'*100}\n{'@'*100}\n{'@'*100}\n Training {Args.RUN_ID} {Args.ENVIRONMENT}")
run(Args)
print(f"{'@'*100}\n{'@'*100}\n{'@'*100}\n Testing {Args.RUN_ID} {Args.ENVIRONMENT}")
tester = TestRunEBud(Args)  
returns = tester.eval_returns_run(n_eval_episodes=50, save=True)
min_emin_ref_f1, mean_emin_ref_f1, _ = tester.eval_emin_run(n_eval_episodes=50, save=True)

################################################################################################
Args.RUN_ID = "etank_1000" 
Args.ENVIRONMENT = "pendulum_f0" 
Args.ENERGY_TANK_INIT = 1000
print(f"{'@'*100}\n{'@'*100}\n{'@'*100}\n Training {Args.RUN_ID} {Args.ENVIRONMENT}")
run(Args)
print(f"{'@'*100}\n{'@'*100}\n{'@'*100}\n Testing {Args.RUN_ID} {Args.ENVIRONMENT}")
tester = TestRunEBud(Args)  
returns = tester.eval_returns_run(n_eval_episodes=50, save=True)
min_emin_ref_f01, mean_emin_ref_f01, _ = tester.eval_emin_run(n_eval_episodes=50, save=True)

################################################################################################
Args.RUN_ID = "etank_1000" 
Args.ENVIRONMENT = "pendulum_f001" 
Args.ENERGY_TANK_INIT = 1000
print(f"{'@'*100}\n{'@'*100}\n{'@'*100}\n Training {Args.RUN_ID} {Args.ENVIRONMENT}")
run(Args)
print(f"{'@'*100}\n{'@'*100}\n{'@'*100}\n Testing {Args.RUN_ID} {Args.ENVIRONMENT}")
tester = TestRunEBud(Args)  
returns = tester.eval_returns_run(n_eval_episodes=50, save=True)
min_emin_ref_f001, mean_emin_ref_f001, _ = tester.eval_emin_run(n_eval_episodes=50, save=True)

################################################################################################
Args.RUN_ID = "etank_meanmin" 
Args.ENVIRONMENT = "pendulum_f001" 
Args.ENERGY_TANK_INIT = mean_emin_ref_f001
print(f"{'@'*100}\n{'@'*100}\n{'@'*100}\n Training {Args.RUN_ID} {Args.ENVIRONMENT}")
run(Args)
print(f"{'@'*100}\n{'@'*100}\n{'@'*100}\n Testing {Args.RUN_ID} {Args.ENVIRONMENT}")
tester = TestRunEBud(Args)  
returns = tester.eval_returns_run(n_eval_episodes=50, save=True)
emin, mean_emin, std_emin = tester.eval_emin_run(n_eval_episodes=50, save=True)

################################################################################################
Args.RUN_ID = "etank_halfmin" 
Args.ENVIRONMENT = "pendulum_f001" 
Args.ENERGY_TANK_INIT = min_emin_ref_f001/2
print(f"{'@'*100}\n{'@'*100}\n{'@'*100}\n Training {Args.RUN_ID} {Args.ENVIRONMENT}")
run(Args)
print(f"{'@'*100}\n{'@'*100}\n{'@'*100}\n Testing {Args.RUN_ID} {Args.ENVIRONMENT}")
tester = TestRunEBud(Args)  
returns = tester.eval_returns_run(n_eval_episodes=50, save=True)
emin, mean_emin, std_emin = tester.eval_emin_run(n_eval_episodes=50, save=True)


