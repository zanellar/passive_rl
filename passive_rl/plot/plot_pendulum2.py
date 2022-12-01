
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

train_env_run_ids = [  
    "pendulum/etank_inf", 
    "pendulum/etank_min" 
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
                                    ext = "pdf", xlim = [0,50], xsteps = False, smooth=False)  
plotter.multirun_returns_train(env_run_ids = train_env_run_ids, labels = labels, show = False, plot_name = plot_name, 
                                ext = "pdf", xlim = [0,200], xsteps = False, smooth=smooth)
plotter.multirun_poserror_train(env_run_ids = train_env_run_ids, labels = labels, show = False, plot_name = plot_name, 
                                ext = "pdf", xlim = [0,200], xsteps = False, smooth=smooth)
 
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

#####

plotter.multirun_poserror_test(env_run_ids = test_env_run_ids, labels = labels, xlabels = "Models", 
                             show = False, plot_name = plot_name, ext = "pdf", plot_type = "boxplot")

plotter.multirun_poserror_test(env_run_ids = test_env_run_ids, labels = labels, xlabels = "Models", 
                            show = False, plot_name = plot_name, ext = "pdf", plot_type = "violinplot") 


###############################################################################################################################
###############################################################################################################################
###############################################################################################################################

train_env_run_ids = [  
    "pendulum/etank_min", 
    "pendulum/etank_min08" ,
    "pendulum/etank_min06" , 
    "pendulum/etank_min03" 
    ]
labels = [
    label_min, 
    label_min08, 
    label_min06, 
    label_min03 
    ]
etank_init_list = [
    min_init,
    min_init08, 
    min_init06, 
    min_init03
    ]
plot_name = "etankinit_mincuts"

#####

plotter.multirun_tanklevel_train( env_run_ids = train_env_run_ids, labels = labels, etank_init_list = etank_init_list, 
                                show = False, plot_name = plot_name, ext = "pdf", xlim = [0,200], xsteps = False, smooth=smooth)

plotter.multirun_returns_train( env_run_ids = train_env_run_ids, labels = labels, show = False, plot_name = plot_name, 
                                ext = "pdf", xlim = [0,200], xsteps = False, smooth=smooth)

plotter.multirun_poserror_train( env_run_ids = train_env_run_ids, labels = labels, show = False, plot_name = plot_name, 
                                ext = "pdf", xlim = [0,200], xsteps = False, smooth=smooth)

###############################################################################################################################

test_env_run_ids = [ 
    "pendulum/etank_min_min", 
    "pendulum/etank_min08_min08",
    "pendulum/etank_min06_min06" , 
    "pendulum/etank_min03_min03"
    ]
labels = [ 
    label_min, 
    label_min08,  
    label_min06, 
    label_min03
    ]
etank_init_list = [ 
    min_init, 
    min_init08, 
    min_init06, 
    min_init03 
    ]
plot_name = "etankinit_mincuts"

#####

plotter.multirun_tanklevel_test(env_run_ids = test_env_run_ids, labels = labels, etank_init_list = etank_init_list, xlabels = "Models", 
                                show = False, plot_name = plot_name, ext = "pdf", plot_type = "boxplot")

plotter.multirun_tanklevel_test(env_run_ids = test_env_run_ids, labels = labels, etank_init_list = etank_init_list, xlabels = "Models", 
                                show = False, plot_name = plot_name, ext = "pdf", plot_type = "violinplot")

#####

plotter.multirun_returns_test(env_run_ids = test_env_run_ids, labels = labels,  xlabels = "Models", 
                             show = False, plot_name = plot_name, ext = "pdf", plot_type = "boxplot") 

plotter.multirun_returns_test(env_run_ids = test_env_run_ids, labels = labels,  xlabels = "Models", 
                               show = False, plot_name = plot_name, ext = "pdf", plot_type = "violinplot")

#####

plotter.multirun_poserror_test(env_run_ids = test_env_run_ids, labels = labels, xlabels = "Models", 
                             show = False, plot_name = plot_name, ext = "pdf", plot_type = "boxplot")

plotter.multirun_poserror_test(env_run_ids = test_env_run_ids, labels = labels, xlabels = "Models", 
                            show = False, plot_name = plot_name, ext = "pdf", plot_type = "violinplot") 

###############################################################################################################################
###############################################################################################################################
###############################################################################################################################


train_env_run_ids = [  
    "pendulum/etank_min", 
    "pendulum/etank_min08eaw" , 
    "pendulum/etank_min06eaw" ,  
    "pendulum/etank_min03eaw"  
    ]
labels = [
    label_min, 
    label_min08_eaw, 
    label_min06_eaw, 
    label_min03_eaw 
    ]
etank_init_list = [
    min_init, 
    min_init08, 
    min_init06, 
    min_init03
    ]
plot_name = "etankinit_mincuts_eaw"

#####

plotter.multirun_tanklevel_train( env_run_ids = train_env_run_ids, labels = labels, etank_init_list = etank_init_list, 
                                show = False, plot_name = plot_name, ext = "pdf", xlim = [0,200], xsteps = False, smooth=smooth)

plotter.multirun_poserror_train( env_run_ids = train_env_run_ids, labels = labels, show = False, plot_name = plot_name, 
                                ext = "pdf", xlim = [0,200], xsteps = False, smooth=smooth)

plotter.multirun_returns_train( env_run_ids = train_env_run_ids, labels = labels, show = False, plot_name = plot_name, 
                                ext = "pdf", xlim = [0,200], xsteps = False, smooth=smooth)

###############################################################################################################################

test_env_run_ids = [ 
    "pendulum/etank_min_min", 
    "pendulum/etank_min08eaw_min08eaw", 
    "pendulum/etank_min06eaw_min06eaw",
    "pendulum/etank_min03eaw_min03eaw" 
    ]
labels = [
    label_min, 
    label_min08_eaw, 
    label_min06_eaw, 
    label_min03_eaw 
    ]
etank_init_list = [
    min_init, 
    min_init08, 
    min_init06, 
    min_init03
    ]
plot_name = "etankinit_mincuts_eaw"

#####

plotter.multirun_tanklevel_test( env_run_ids = test_env_run_ids, etank_init_list = etank_init_list, 
                                labels = labels,   xlabels = "Models", show = False, plot_name = plot_name, ext = "pdf", plot_type = "boxplot") 

plotter.multirun_tanklevel_test( env_run_ids = test_env_run_ids, etank_init_list = etank_init_list, 
                                labels = labels,   xlabels = "Models", show = False, plot_name = plot_name, ext = "pdf", plot_type = "violinplot") 

####

plotter.multirun_poserror_test( env_run_ids = test_env_run_ids, labels = labels,   xlabels = "Models", 
                                show = False, plot_name = plot_name, ext = "pdf", plot_type = "boxplot") 

plotter.multirun_poserror_test( env_run_ids = test_env_run_ids, labels = labels,   xlabels = "Models", 
                                show = False, plot_name = plot_name, ext = "pdf", plot_type = "violinplot")

 
#####

plotter.multirun_returns_test(env_run_ids = test_env_run_ids, labels = labels, xlabels = "Models", 
                             show = False, plot_name = plot_name, ext = "pdf", plot_type = "boxplot")

plotter.multirun_returns_test(env_run_ids = test_env_run_ids, labels = labels, xlabels = "Models", 
                            show = False, plot_name = plot_name, ext = "pdf", plot_type = "violinplot")