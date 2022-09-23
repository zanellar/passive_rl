
from passive_rl.scripts.pkgpaths import PkgPath
from passive_rl.scripts.plotter import PlotterEBud 
  
out_train_folder = PkgPath.OUT_TRAIN_FOLDER  
out_test_folder = PkgPath.OUT_TEST_FOLDER  
plot_folder = PkgPath.PLOT_FOLDER  

plotter = PlotterEBud(out_train_folder=out_train_folder, out_test_folder=out_test_folder, plot_folder=plot_folder) 

# plotter.plot_avg_train_run_energy(env_name="pendulum_f001",run_id="etank1000", select="min", show=False)
# plotter.plot_avg_train_run_returns(env_name="pendulum_f001",run_id="etank1000", show=False)

env_run_ids=["pendulum_f001/etank1000", "pendulum_f01/etank1000"]
# plotter.plot_avg_train_multirun_energy(env_run_ids=env_run_ids, show=True)
# plotter.plot_avg_train_multirun_returns(env_run_ids=env_run_ids, show=True)
env_run_ids=["pendulum_f001/etank1000","pendulum_f001/etank1000bis" ]
plotter.plot_stat_test_multirun_energy(env_run_ids=env_run_ids, show=True)
# plotter.plot_boxes_test_multirun_returns(env_run_ids=env_run_ids, show=False)