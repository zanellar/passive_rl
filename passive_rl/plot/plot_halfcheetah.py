
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

label_inf_inf = "$\phi_\infty$" 
label_inf_min = "$\phi_{\infty|{E^*}}$" 
label_min = "$\phi_{E^*}$" 

plotter = PlotterEBud(out_train_folder = out_train_folder, out_test_folder = out_test_folder, plot_folder = plot_folder) 

smooth = False

 
###############################################################################################################################
###############################################################################################################################
###############################################################################################################################

train_env_run_ids = [  
    "halfcheetah/etank_inf", 
    "halfcheetah/etank_min" 
    ] 
labels = [
    label_inf_inf, 
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
                                    ext = "pdf", xlim = [0,1000], xsteps = False, smooth=False)  
plotter.multirun_returns_train(env_run_ids = train_env_run_ids, labels = labels, show = False, plot_name = plot_name, 
                                ext = "pdf", xlim = [0,1000], xsteps = False, smooth=smooth) 
 
###############################################################################################################################

test_env_run_ids = [
    "halfcheetah/etank_inf_min",
    "halfcheetah/etank_min_min" 
    ]
labels = [
    label_inf_min, 
    label_min
    ]
etank_init_list = [
    min_init, 
    min_init 
    ]
plot_name = "tankinit_infmin"

#####

plotter.multirun_energytank_test(env_run_ids = test_env_run_ids, labels = labels, xlabels = "Models", 
                                show = False, plot_name = plot_name, ext = "pdf", plot_type = "boxplot")

plotter.multirun_energytank_test(env_run_ids = test_env_run_ids, labels = labels, xlabels = "Models",
                                show = False, plot_name = plot_name, ext = "pdf", plot_type = "violinplot")
  
#####

plotter.multirun_tanklevel_test(env_run_ids = test_env_run_ids, etank_init_list = etank_init_list, labels = labels, xlabels = "Models", 
                                show = False, plot_name = plot_name, ext = "pdf", plot_type = "boxplot") 

plotter.multirun_tanklevel_test(env_run_ids = test_env_run_ids, etank_init_list = etank_init_list, labels = labels, xlabels = "Models", 
                                show = False, plot_name = plot_name, ext = "pdf", plot_type = "violinplot")

#####

plotter.multirun_returns_test(env_run_ids = test_env_run_ids, labels = labels, xlabels = "Models", 
                             show = False, plot_name = plot_name, ext = "pdf", plot_type = "boxplot")

plotter.multirun_returns_test(env_run_ids = test_env_run_ids, labels = labels, xlabels = "Models", 
                            show = False, plot_name = plot_name, ext = "pdf", plot_type = "violinplot")
 