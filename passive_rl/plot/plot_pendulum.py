
from re import L
from passive_rl.scripts.pkgpaths import PkgPath
from passive_rl.scripts.plotter import PlotterEBud 
  
out_train_folder = PkgPath.OUT_TRAIN_FOLDER  
out_test_folder = PkgPath.OUT_TEST_FOLDER  
plot_folder = PkgPath.PLOT_FOLDER  

plotter = PlotterEBud(out_train_folder=out_train_folder, out_test_folder=out_test_folder, plot_folder=plot_folder) 
 
env_run_ids = ["pendulum_f001/etank_1000", "pendulum_f1/etank_1000", "pendulum_f0/etank_1000"]
# plotter.plot_avg_train_multirun_energy(env_run_ids=env_run_ids, show=False, labels=["0.01", "1", "0"], plot_name="friction")
plotter.plot_stat_test_multirun_energy(env_run_ids=env_run_ids, show=False, labels=["0.01", "1", "0"], plot_name="friction")

env_run_ids = ["pendulum_f001/etank_1000", "pendulum_f001/etank_minemin", "pendulum_f001/etank_minemin08"]
# plotter.plot_avg_train_multirun_returns(env_run_ids=env_run_ids, show=False, labels=["1000", "min", "0.8*min"], plot_name="tankinit")
plotter.plot_stat_test_multirun_energy(env_run_ids=env_run_ids, show=False, labels=["1000", "min", "0.8*min"], plot_name="tankinit")
 