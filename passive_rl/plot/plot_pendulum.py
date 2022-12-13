
from ensurepip import bootstrap
import os
from passive_rl.scripts.pkgpaths import PkgPath
from passive_rl.scripts.plotter import PlotterEBud 
  
out_train_folder = PkgPath.OUT_TRAIN_FOLDER  
out_test_folder = PkgPath.OUT_TEST_FOLDER  
plot_folder = PkgPath.PLOT_FOLDER  

pendulum_env = "pendulum"
pendulum_wind_env = "pendulum_wind"

with open(os.path.join(out_train_folder,pendulum_env,"run_results.txt")) as f:
    lines = [line.rstrip() for line in f]
    inf_init = float(lines[1].split(" ")[2]) 
    min_init = float(lines[2].split(" ")[2])    

label_inf = "$\phi_{\infty}$" 
label_min = "$\phi_{{e^*}}$" 
label_inf_inf = "$\phi_{\infty|\infty}$" 
label_inf_min = "$\phi_{\infty|{e^*}}$" 
label_inf_inf_wind = "$\phi_{\infty|\infty,w}$" 
label_inf_min_wind = "$\phi_{\infty|{e^*},w} $ " 
label_min_min = "$\phi_{{e^*}|{e^*}}$" 
label_min_inf = "$\phi_{{e^*}|\infty}$" 
label_min_wind = "$\phi_{{e^*}|{e^*},w}$" 

plotter = PlotterEBud(out_train_folder = out_train_folder, out_test_folder = out_test_folder, plot_folder = plot_folder) 

smooth = True

 
###############################################################################################################################
###############################################################################################################################
###############################################################################################################################

train_env_run_ids = [  
    "pendulum/etank_inf", 
    "pendulum/etank_min" 
    ] 
labels = [
    label_inf, 
    label_min
    ]
etank_init_list = [
    inf_init, 
    min_init 
    ]
plot_name = "etankinit_infmin"

#####

plotter.multirun_energyexiting_train(env_run_ids = train_env_run_ids, etank_init_list = etank_init_list, 
                                    labels = labels, ylabels = "Energy", show = False, plot_name = plot_name, 
                                    sub_folder_name = pendulum_env, ext = "pdf", xlim = [0,50], xsteps = False, smooth=False)  
plotter.multirun_returns_train(env_run_ids = train_env_run_ids, labels = labels, show = False, plot_name = plot_name, 
                                sub_folder_name = pendulum_env, ext = "pdf", xlim = [0,200], xsteps = False, smooth=smooth) 
 
###############################################################################################################################

test_env_run_ids = [
    "pendulum/etank_inf_inf" ,
    "pendulum_wind/etank_inf_inf",
    "pendulum_wind/etank_inf_min" ,
    ]
labels = [
    label_inf_inf_wind,
    label_inf_min_wind, 
    label_inf_inf
    ]
etank_init_list = [
    inf_init,
    min_init, 
    min_init
    ]
plot_name = "inference_energy_wind"

##### 

plotter.multirun_energyexiting_test(env_run_ids = test_env_run_ids, labels = labels, ylabels = "Energy",  
                                show = False, save=True, sub_folder_name = pendulum_wind_env, plot_name = plot_name, ext = "pdf", 
                                xlim = [0,500], 
                                ylim = [0,20], 
                                xsteps = False, smooth=False, bootstrap=True)
    

###############################################################################################################################

test_env_run_ids = [ 
    "pendulum/etank_min_min" ,
    "pendulum/etank_inf_min" 
    ]
labels = [  
    label_min_min,
    label_inf_min
    ]
etank_init_list = [ 
    inf_init,  
    inf_init
    ]
plot_name = "inference_energy"

##### 

plotter.multirun_energyexiting_test(env_run_ids = test_env_run_ids, labels = labels, ylabels = "Energy",  
                                show = False, save=True, sub_folder_name = pendulum_env, plot_name = plot_name, ext = "pdf", 
                                xlim = [0,500], 
                                ylim = [0,3], 
                                xsteps = False, smooth=False, bootstrap=True)
    

###############################################################################################################################

test_env_run_ids = [ 
    "pendulum/etank_min_min" ,
    "pendulum/etank_inf_min" 
    ]
labels = [  
    label_min_min,
    label_inf_min
    ]
etank_init_list = [ 
    inf_init,  
    inf_init
    ]
plot_name = "inference_poserror"

##### 

plotter.multirun_poserror_test(env_run_ids = test_env_run_ids, labels = labels, ylabels = "Error",  
                                show = False, save=True, sub_folder_name = pendulum_env, plot_name = plot_name, ext = "pdf", 
                                # xlim = [0,500], 
                                # ylim = [0,3], 
                                xsteps = False, smooth=True, bootstrap=True)
    