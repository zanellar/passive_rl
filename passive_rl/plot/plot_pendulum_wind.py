
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

label_inf_inf = "$\phi_\infty$" 
label_inf_min = "$\phi_{\infty|{e^*}}$" 
label_min = "$\phi_{e^*}$" 

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
                                xlim = [0,200], ylim = [0,60], xsteps = False, smooth=False)
    