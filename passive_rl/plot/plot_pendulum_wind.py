
import os
from passive_rl.scripts.pkgpaths import PkgPath
from passive_rl.scripts.plotter import PlotterEBud 
  
out_train_folder = PkgPath.OUT_TRAIN_FOLDER  
out_test_folder = PkgPath.OUT_TEST_FOLDER  
plot_folder = PkgPath.PLOT_FOLDER  

sub_folder_name = "pendulum_wind"

with open(os.path.join(out_train_folder,"run_results.txt")) as f:
    lines = [line.rstrip() for line in f]
    inf_init = float(lines[1].split(" ")[2]) 
    min_init = float(lines[2].split(" ")[2])  
    min_init08 = float(lines[3].split(" ")[2]) # min_init*0.8
    min_init06 = float(lines[4].split(" ")[2]) # min_init*0.6
    min_init05 = float(lines[5].split(" ")[2]) # min_init*0.6
    min_init03 = float(lines[6].split(" ")[2]) # min_init*0.3
    min_init02 = float(lines[7].split(" ")[2]) # min_init*0.3
    min_init01 = float(lines[8].split(" ")[2]) # min_init*0.3

label_inf_inf = "$\phi_\infty$" 
label_inf_min = "$\phi_{\infty|{e^*}}$" 
label_min = "$\phi_{e^*}$"
label_min08 = "$\phi_{e^*_{0.8}}$"
label_min06 = "$\phi_{e^*_{0.6}}$"
label_min05 = "$\phi_{e^*_{0.5}}$"
label_min03 = "$\phi_{e^*_{0.3}}$"
label_min02 = "$\phi_{e^*_{0.2}}$"
label_min01 = "$\phi_{e^*_{0.1}}$"
label_min08_eaw = "$\phi_{e^*_{0.8}}^{aw}$"
label_min06_eaw = "$\phi_{e^*_{0.6}}^{aw}$"
label_min05_eaw = "$\phi_{e^*_{0.5}}^{aw}$"
label_min03_eaw = "$\phi_{e^*_{0.3}}^{aw}$"
label_min02_eaw = "$\phi_{e^*_{0.2}}^{aw}$"
label_min01_eaw = "$\phi_{e^*_{0.1}}^{aw}$"

plotter = PlotterEBud(out_train_folder = out_train_folder, out_test_folder = out_test_folder, plot_folder = plot_folder) 

smooth = True
 
###############################################################################################################################

test_env_run_ids = [
    "pendulum_wind/etank_inf_inf",
    "pendulum_wind/etank_inf_min" ,
    "pendulum_wind/etank_min_min" 
    ]
labels = [
    label_inf_inf,
    label_inf_min, 
    label_min
    ]
etank_init_list = [
    inf_init,
    min_init, 
    min_init 
    ]
plot_name = "tankinit_infmin"

##### 

plotter.multirun_energyexiting_test(env_run_ids = test_env_run_ids, labels = labels, ylabels = "Energy",  
                                show = False, sub_folder_name = sub_folder_name, plot_name = plot_name, ext = "pdf", 
                                xlim = [0,200], ylim = [0,60], xsteps = False, smooth=smooth)
   

# ###############################################################################################################################
# ###############################################################################################################################
# ###############################################################################################################################
  
# test_env_run_ids = [ 
#     "pendulum_wind/etank_min_min", 
#     # "pendulum_wind/etank_min08_min08",
#     "pendulum_wind/etank_min06_min06" , 
#     # "pendulum_wind/etank_min05_min05" , 
#     "pendulum_wind/etank_min03_min03",
#     # "pendulum_wind/etank_min02_min02",
#     "pendulum_wind/etank_min01_min01"
#     ]
# labels = [ 
#     label_min, 
#     # label_min08,  
#     label_min06, 
#     # label_min05, 
#     label_min03, 
#     # label_min02, 
#     label_min01
#     ]
# etank_init_list = [ 
#     min_init, 
#     # min_init08, 
#     min_init06, 
#     # min_init05, 
#     min_init03, 
#     # min_init02, 
#     min_init01 
#     ]
# plot_name = "etankinit_mincuts"
 
# #####
 
# plotter.multirun_energyexiting_test(env_run_ids = test_env_run_ids, labels = labels, ylabels = "Energy",  
#                                 show = False, sub_folder_name = sub_folder_name, plot_name = plot_name, ext = "pdf",
#                                  xlim = [0,200], ylim = [0,20], xsteps = False, smooth=smooth)
   