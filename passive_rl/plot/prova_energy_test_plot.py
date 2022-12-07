
import os
from passive_rl.scripts.pkgpaths import PkgPath
from passive_rl.scripts.plotter import PlotterEBud 
  
out_train_folder = PkgPath.OUT_TRAIN_FOLDER  
out_test_folder = PkgPath.OUT_TEST_FOLDER  
plot_folder = PkgPath.PLOT_FOLDER  

with open(os.path.join(out_train_folder,"run_results.txt")) as f:
    lines = [line.rstrip() for line in f]
    inf_init = float(lines[1].split(" ")[2]) 
    min_init = float(lines[2].split(" ")[2])  
    min_init08 = float(lines[3].split(" ")[2]) # min_init*0.8
    min_init06 = float(lines[4].split(" ")[2]) # min_init*0.6
    min_init03 = float(lines[5].split(" ")[2]) # min_init*0.3

label_inf_inf = "$\phi_\infty$" 
label_inf_min = "$\phi_{\infty|{E^*}}$" 
label_min = "$\phi_{E^*}$"
label_min08 = "$\phi_{E^*_{0.8}}$"
label_min06 = "$\phi_{E^*_{0.6}}$"
label_min03 = "$\phi_{E^*_{0.3}}$"
label_min08_eaw = "$\phi_{E^*_{0.8}}^{aw}$"
label_min06_eaw = "$\phi_{E^*_{0.6}}^{aw}$"
label_min03_eaw = "$\phi_{E^*_{0.3}}^{aw}$"

plotter = PlotterEBud(out_train_folder = out_train_folder, out_test_folder = out_test_folder, plot_folder = plot_folder) 

smooth = False

 
###############################################################################################################################
###############################################################################################################################
###############################################################################################################################
 
###############################################################################################################################

test_env_run_ids = [
    "pendulum/etank_inf_min",
    "pendulum/etank_min_min" 
    ]
labels = [
    label_inf_min, 
    label_min
    ]
etank_init_list = [
    min_init, 
    min_init 
    ]
plot_name = "prova"

#####

plotter.multirun_energyexiting_test(env_run_ids = test_env_run_ids, labels = labels, 
                                show = False, plot_name = plot_name, ext = "pdf", xlim = [0,200], xsteps = False, smooth=smooth)
  