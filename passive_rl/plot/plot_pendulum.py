
from re import L
from passive_rl.scripts.pkgpaths import PkgPath
from passive_rl.scripts.plotter import PlotterEBud 
  
out_train_folder = PkgPath.OUT_TRAIN_FOLDER  
out_test_folder = PkgPath.OUT_TEST_FOLDER  
plot_folder = PkgPath.PLOT_FOLDER  

inf_init = 1000
min_init = 3.234176997037139
min_init08 = 2.5873415976297114
min_init06 = 1.940506198222283
min_init03 = 0.9702530991111415


label_inf_inf = "$\phi_\infty$" 
label_inf_min = "$\phi_{\infty|{E^*}}$" 
label_min = "$\phi_{E^*}$"
label_min08 = "$\phi_{E^*_{0.8}}$"
label_min06 = "$\phi_{E^*_{0.6}}$"
label_min03 = "$\phi_{E^*_{0.3}}$"
label_min08_eaw = "$\phi_{E^*_{0.8}}^{aw}$"
label_min06_eaw = "$\phi_{E^*_{0.6}}^{aw}$"
label_min03_eaw = "$\phi_{E^*_{0.3}}^{aw}$"

plotter = PlotterEBud(out_train_folder=out_train_folder, out_test_folder=out_test_folder, plot_folder=plot_folder) 

###############################################################################################################################

train_env_run_ids = [  "pendulum_f1/etank_inf", "pendulum_f0/etank_inf", "pendulum_f001/etank_inf"]
plotter.plot_avg_train_multirun_energy(
    env_run_ids=train_env_run_ids,
    labels=["1", "0", "0.01"],  
    show=False, plot_name="friction", ext="pdf", xlim=[0,30], xsteps=False) 
plotter.plot_avg_train_multirun_returns(
    env_run_ids=train_env_run_ids,
    labels=[ "1", "0", "0.01"],  
    show=False, plot_name="friction", ext="pdf", xlim=[0,200], xsteps=False) 
plotter.plot_avg_train_multirun_error(
    env_run_ids=train_env_run_ids,
    labels=[ "1", "0", "0.01"],  
    show=False, plot_name="friction", ext="pdf", xlim=[0,200], xsteps=False)

###############################################################################################################################

test_env_run_ids = ["pendulum_f001/etank_inf_inf", "pendulum_f1/etank_inf_inf", "pendulum_f0/etank_inf_inf"]
plotter.plot_stat_test_multirun_energy(
    env_run_ids=test_env_run_ids, 
    labels=["0.01", "1", "0"], 
    show=False, plot_name="friction_box", ext="pdf", type="boxplot")
plotter.plot_stat_test_multirun_energy(
    env_run_ids=test_env_run_ids, 
    labels=["0.01", "1", "0"], 
    show=False, plot_name="friction_hist", ext="pdf", type="histplot")
plotter.plot_stat_test_multirun_energy(
    env_run_ids=test_env_run_ids, 
    labels=["0.01", "1", "0"], 
    show=False, plot_name="friction_violin", ext="pdf", type="violinplot")
 

 
###############################################################################################################################
###############################################################################################################################
###############################################################################################################################

train_env_run_ids = [  "pendulum_f001/etank_inf", "pendulum_f001/etank_min" ] 
plotter.plot_avg_train_multirun_returns(
    env_run_ids=train_env_run_ids,
    labels=[label_inf_inf, label_min ],   
    show=False, plot_name="etankinit_infmin", ext="pdf", xlim=[0,200], xsteps=False)
plotter.plot_avg_train_multirun_error(
    env_run_ids=train_env_run_ids,
    labels=[label_inf_inf, label_min],   
    show=False, plot_name="etankinit", ext="pdf", xlim=[0,200], xsteps=False)
 
###############################################################################################################################

test_env_run_ids = ["pendulum_f001/etank_inf_min","pendulum_f001/etank_min_min" ]
plotter.plot_stat_test_multirun_tanklv(
    env_run_ids=test_env_run_ids, 
    etank_init_list=[min_init, min_init ], 
    labels=[label_inf_min, label_min], 
    show=False, plot_name="tankinit_box_infmin", ext="pdf", type="boxplot") 
plotter.plot_stat_test_multirun_tanklv(
    env_run_ids=test_env_run_ids, 
    etank_init_list=[min_init, min_init ], 
    labels=[label_inf_min, label_min ], 
    show=False, plot_name="tankinit_hist_infmin", ext="pdf", type="histplot")  
plotter.plot_stat_test_multirun_tanklv(
    env_run_ids=test_env_run_ids, 
    etank_init_list=[min_init, min_init ], 
    labels=[label_inf_min, label_min ], 
    show=False, plot_name="tankinit_violin_infmin", ext="pdf", type="violinplot") 


###############################################################################################################################

train_env_run_ids = [  "pendulum_f001/etank_min", "pendulum_f001/etank_min08" , "pendulum_f001/etank_min03" ]
plotter.plot_avg_train_multirun_tanklv(
    env_run_ids=train_env_run_ids,
    labels=[label_min, label_min08, label_min03 ],  
    etank_init_list = [min_init, min_init08, min_init03],
    show=False, plot_name="etankinit_mincuts", ext="pdf", xlim=[0,200], xsteps=False) 
plotter.plot_avg_train_multirun_returns(
    env_run_ids=train_env_run_ids,
    labels=[label_min, label_min08, label_min03 ],   
    show=False, plot_name="etankinit_mincuts", ext="pdf", xlim=[0,200], xsteps=False)
plotter.plot_avg_train_multirun_error(
    env_run_ids=train_env_run_ids,
    labels=[label_min, label_min08, label_min03 ],   
    show=False, plot_name="etankinit_mincuts", ext="pdf", xlim=[0,200], xsteps=False)

###############################################################################################################################

test_env_run_ids = [ "pendulum_f001/etank_min_min", "pendulum_f001/etank_min08_min08",   "pendulum_f001/etank_min03_min03"]
plotter.plot_stat_test_multirun_energy(
    env_run_ids=test_env_run_ids,  
    labels=[ label_min, label_min08,  label_min03], 
    show=False, plot_name="tankinit_box_mincuts", ext="pdf", type="boxplot") 
plotter.plot_stat_test_multirun_energy(
    env_run_ids=test_env_run_ids,  
    labels=[ label_min, label_min08,  label_min03], 
    show=False, plot_name="tankinit_hist_mincuts", ext="pdf", type="histplot") 
plotter.plot_stat_test_multirun_energy(
    env_run_ids=test_env_run_ids,  
    labels=[  label_min, label_min08,  label_min03], 
    show=False, plot_name="tankinit_violin_mincuts", ext="pdf", type="violinplot") 

plotter.plot_stat_test_multirun_tanklv(
    env_run_ids=test_env_run_ids,  
    labels=[ label_min, label_min08,  label_min03], 
    etank_init_list=[ min_init, min_init08, min_init03 ], 
    show=False, plot_name="tankinit_box_mincuts", ext="pdf", type="boxplot") 
plotter.plot_stat_test_multirun_tanklv(
    env_run_ids=test_env_run_ids,  
    labels=[ label_min, label_min08,  label_min03], 
    etank_init_list=[ min_init, min_init08, min_init03 ], 
    show=False, plot_name="tankinit_hist_mincuts", ext="pdf", type="histplot") 
plotter.plot_stat_test_multirun_tanklv(
    env_run_ids=test_env_run_ids,  
    labels=[ label_min, label_min08,  label_min03], 
    etank_init_list=[ min_init, min_init08, min_init03 ], 
    show=False, plot_name="tankinit_violin_mincuts", ext="pdf", type="violinplot") 

###############################################################################################################################
###############################################################################################################################
###############################################################################################################################


train_env_run_ids = [  "pendulum_f001/etank_min", "pendulum_f001/etank_min08eaw" ,  "pendulum_f001/etank_min03eaw"  ]
plotter.plot_avg_train_multirun_tanklv(
    env_run_ids=train_env_run_ids,
    labels=[label_min, label_min08_eaw,   label_min03_eaw ],  
    etank_init_list = [min_init, min_init08, min_init03],
    show=False, plot_name="etankinit_mincuts_eaw", ext="pdf", xlim=[0,200], xsteps=False)  
plotter.plot_avg_train_multirun_error(
    env_run_ids=train_env_run_ids,
    labels=[label_min, label_min08_eaw, label_min03_eaw ],  
    show=False, plot_name="etankinit_mincuts_eaw", ext="pdf", xlim=[0,200], xsteps=False)

###############################################################################################################################

test_env_run_ids = [ "pendulum_f001/etank_min_min", "pendulum_f001/etank_min08eaw_min08eaw", "pendulum_f001/etank_min03eaw_min03eaw" ]
plotter.plot_stat_test_multirun_energy(
    env_run_ids=test_env_run_ids,  
    labels=[label_min, label_min08_eaw, label_min03_eaw ],  
    show=False, plot_name="tankinit_box_mincuts_eaw", ext="pdf", type="boxplot") 
plotter.plot_stat_test_multirun_energy(
    env_run_ids=test_env_run_ids,  
    labels=[label_min, label_min08_eaw,  label_min03_eaw ],  
    show=False, plot_name="tankinit_hist_mincuts_eaw", ext="pdf", type="histplot") 
plotter.plot_stat_test_multirun_energy(
    env_run_ids=test_env_run_ids,  
    labels=[label_min, label_min08_eaw, label_min03_eaw ],  
    show=False, plot_name="tankinit_violin_mincuts_eaw", ext="pdf", type="violinplot")  

plotter.plot_stat_test_multirun_tanklv(
    env_run_ids=test_env_run_ids, 
    etank_init_list = [min_init, min_init08, min_init03],
    labels=[label_min, label_min08_eaw, label_min03_eaw ],  
    show=False, plot_name="tankinit_box_mincuts_eaw", ext="pdf", type="boxplot") 
plotter.plot_stat_test_multirun_tanklv(
    env_run_ids=test_env_run_ids, 
    etank_init_list = [min_init, min_init08, min_init03],
    labels=[label_min, label_min08_eaw,  label_min03_eaw ],  
    show=False, plot_name="tankinit_hist_mincuts_eaw", ext="pdf", type="histplot")  
plotter.plot_stat_test_multirun_tanklv(
    env_run_ids=test_env_run_ids, 
    etank_init_list = [min_init, min_init08, min_init03],
    labels=[label_min, label_min08_eaw,   label_min03_eaw ],  
    show=False, plot_name="tankinit_violin_mincuts_eaw", ext="pdf", type="violinplot") 
