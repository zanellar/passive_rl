
from passive_rl.scripts.pkgpaths import PkgPath 
from passive_rl.scripts.energytester import TestMultiRunEBud, TestRunEBud
from mjrlenvs.scripts.eval.tester import TestRun
from passive_rl.scripts.plotenergydata import *


tester = TestMultiRunEBud()

tester.plot(env_run_ids=["pendulum_f001/etank1000", "pendulum_f001/prova1", "pendulum_f001/prova2"])