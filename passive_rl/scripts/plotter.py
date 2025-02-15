 
import os  
import matplotlib.pyplot as plt
import seaborn as sns  
import pandas as pd 

from passive_rl.scripts.datautils import *  
from passive_rl.scripts.pkgpaths import PkgPath 
from mjrlenvs.scripts.plot.plotter import Plotter

class PlotterEBud(Plotter):
 

    def __init__(self, out_train_folder=None, out_test_folder=None, plot_folder=None) -> None:
        super().__init__(out_train_folder=out_train_folder, out_test_folder=out_test_folder, plot_folder=plot_folder)
     
             
    def multirun_energytank_train(self, env_run_ids, labels=[],  xsteps=False, smooth=False, save=True, show=True, plot_name=None, 
                                sub_folder_name="", ext="pdf", xlim=[None,None], ylim=[None,None], xlabels=None, ylabels=None):
        if plot_name is None:
            plot_name = str(len(env_run_ids))

        run_paths_list = [os.path.join(self.out_train_folder, env_run) for env_run in env_run_ids]

        data = df_multiruns_episodes_energy(run_paths_list=run_paths_list, smooth=smooth, interpolate=xsteps, etype="min", run_label_list=labels )  
        save_path = os.path.join(self.save_multirun_training_plots_path, sub_folder_name, f"{plot_name}_multirun_energytank_train.{ext}") 

        self._line_plot(
            data = data,
            x =  "Episodes", 
            y = "Energy",  
            hue = "Runs", 
            xsteps = xsteps,
            run_paths_list = run_paths_list,
            labels = labels,
            xlabels = xlabels,
            ylabels = ylabels,
            xlim = xlim,
            ylim = ylim,
            show = show,
            save = save,
            save_path = save_path,
            ext = ext
        ) 
  
    def multirun_energyexiting_train(self, env_run_ids, labels=[], etank_init_list=[],  xsteps=False, smooth=False, save=True, show=True, 
                                    plot_name=None, sub_folder_name="", ext="pdf", xlim=[None,None], ylim=[None,None], xlabels=None, ylabels=None):
        if plot_name is None:
            plot_name = str(len(env_run_ids))

        run_paths_list = [os.path.join(self.out_train_folder, env_run) for env_run in env_run_ids]

        data = df_multiruns_episodes_energy(run_paths_list=run_paths_list, smooth=smooth, interpolate=xsteps, etype="min", run_label_list=labels, etank_init_list=etank_init_list) 
        save_path = os.path.join(self.save_multirun_training_plots_path, sub_folder_name, f"{plot_name}_multirun_energyexiting_train.{ext}") 
        
        self._line_plot(
            data = data,
            x =  "Episodes", 
            y = "Exiting",  
            hue = "Runs", 
            xsteps = xsteps,
            run_paths_list = run_paths_list,
            labels = labels,
            xlabels = xlabels,
            ylabels = ylabels,
            xlim = xlim,
            ylim = ylim,
            show = show,
            save = save,
            save_path = save_path,
            ext = ext
        )  

 
    ##############################################################################################################################################################

    def multirun_tanklevel_train(self, env_run_ids, labels=[], etank_init_list=[],  xsteps=False, smooth=False, save=True, show=True, 
                                plot_name=None, sub_folder_name="", ext="pdf", xlim=[None,None], ylim=[None,None], xlabels=None, ylabels=None):
        if plot_name is None:
            plot_name = str(len(env_run_ids))

        run_paths_list = [os.path.join(self.out_train_folder, env_run) for env_run in env_run_ids]

        data = df_multiruns_episodes_energy(run_paths_list=run_paths_list, smooth=smooth, interpolate=xsteps, etype="min", run_label_list=labels, etank_init_list=etank_init_list) 
        save_path = os.path.join(self.save_multirun_training_plots_path, sub_folder_name, f"{plot_name}_multirun_tanklevel_train.{ext}") 
        
        self._line_plot(
            data = data,
            x =  "Episodes", 
            y = "Level",  
            hue = "Runs", 
            xsteps = xsteps,
            run_paths_list = run_paths_list,
            labels = labels,
            xlabels = xlabels,
            ylabels = ylabels,
            xlim = xlim,
            ylim = ylim,
            show = show,
            save = save,
            save_path = save_path,
            ext = ext
        )  

    ##############################################################################################################################################################

    def multirun_poserror_train(self, env_run_ids, labels=[], xsteps=False, smooth=False, save=True, show=True, plot_name=None, 
                                sub_folder_name="", ext="pdf", xlim=[None,None], ylim=[None,None], xlabels=None, ylabels=None):
        if plot_name is None:
            plot_name = str(len(env_run_ids))

        run_paths_list = [os.path.join(self.out_train_folder, env_run) for env_run in env_run_ids]

        data = df_multiruns_episodes_error(run_paths_list=run_paths_list, smooth=smooth, interpolate=xsteps, run_label_list=labels, cumulative_error=True ) 
        save_path = os.path.join(self.save_multirun_training_plots_path, sub_folder_name, f"{plot_name}_multirun_poserror_train.{ext}") 

        self._line_plot(
            data = data,
            x =  "Episodes", 
            y = "Errors",  
            hue = "Runs", 
            xsteps = xsteps,
            run_paths_list = run_paths_list,
            labels = labels,
            xlabels = xlabels,
            ylabels = ylabels,
            xlim = xlim,
            ylim = ylim,
            show = show,
            save = save,
            save_path = save_path,
            ext = ext
        )   
 

    ##############################################################################################################################################################
    ##############################################################################################################################################################
    ##############################################################################################################################################################
    
    def multirun_statplot_energytank_test(self, env_run_ids, labels=[], xlabels=None, ylabels=None, etank_init_list=[], 
                                save=True, show=True, sub_folder_name="", plot_name=None, plot_type="histplot", ext="pdf"): 
        if plot_name == None:
            plot_name = str(len(env_run_ids))
        run_paths_list = [os.path.join(self.out_test_folder, env_run) for env_run in env_run_ids]
        
        data = df_test_multirun_etank(run_paths_list=run_paths_list, etank_init_list=etank_init_list) 
        save_path = os.path.join(self.save_multirun_testing_plots_path, sub_folder_name, f"{plot_name}_{plot_type}_multirun_energytank_test.{ext}")

        self._stat_plot(
            data = data,
            x =  "Runs", 
            y = "Energy",  
            hue = None,   
            plot_type = plot_type,
            labels = labels, 
            xlabels = xlabels,
            ylabels = ylabels,
            show = show,
            save = save,
            save_path = save_path,
            ext = ext
        )    


    ##############################################################################################################################################################
 
    def multirun_energyexiting_test(self, env_run_ids, labels=[], xlabels=None, ylabels=None , xsteps=False, smooth=False, 
                                    save=True, show=True, plot_name=None, sub_folder_name="",   ext="pdf", 
                                    xlim=[None,None], ylim=[None,None], bootstrap=True): 
        if plot_name == None:
            plot_name = str(len(env_run_ids))
        run_paths_list = [os.path.join(self.out_test_folder, env_run) for env_run in env_run_ids]
        
        data = df_test_multirun_energy(run_paths_list=run_paths_list, smooth=smooth, run_label_list=labels ) 
        save_path = os.path.join(self.save_multirun_testing_plots_path, sub_folder_name, f"{plot_name}_multirun_energyexiting_test.{ext}")

        if bootstrap: 
            self._line_plot(
                data = data,
                x =  "Steps", 
                y = "TotalEnergy",  
                hue = "Runs", 
                xsteps = xsteps,
                run_paths_list = run_paths_list,
                labels = labels,
                xlabels = xlabels,
                ylabels = ylabels,
                xlim = xlim,
                ylim = ylim,
                show = show,
                save = save,
                save_path = save_path,
                ext = ext
            )   
        else: 
            maxvals = {}
            minvals = {}
            avgvals = {}
            for run in data["Runs"].unique():
                datarun = data[data["Runs"]==run]
                df = pd.DataFrame( )  
                for i,test in enumerate(datarun["Tests"].unique()): 
                    df.insert(loc=len(df.columns), column=str(i), value=datarun[datarun["Tests"]==test]["TotalEnergy"].reset_index(drop=True))    
                maxvals[run] = df.max(axis=1)
                minvals[run] = df.min(axis=1)
                avgvals[run] = df.mean(axis=1)
                
            # print(df.iloc[498])
            self._line_plot_nobs(
                maxvals = maxvals,
                minvals = minvals,
                avgvals = avgvals, 
                labels = labels,
                xlabels = xlabels,
                ylabels = ylabels,
                xlim = xlim,
                ylim = ylim,
                show = show,
                save = save,
                save_path = save_path,
                ext = ext
            )
            
    ##############################################################################################################################################################
 
    def multirun_poserror_test(self, env_run_ids, labels=[], xlabels=None, ylabels=None , xsteps=False, smooth=False,  
                                save=True, show=True, plot_name=None, sub_folder_name="", ext="pdf", 
                                xlim=[None,None], ylim=[None,None], bootstrap=True): 
        if plot_name == None:
            plot_name = str(len(env_run_ids))
        run_paths_list = [os.path.join(self.out_test_folder, env_run) for env_run in env_run_ids]
        
        data = df_test_multirun_poserror(run_paths_list=run_paths_list, smooth=smooth, run_label_list=labels ) 
        save_path = os.path.join(self.save_multirun_testing_plots_path, sub_folder_name, f"{plot_name}_multirun_poserror_test.{ext}")

        if bootstrap: 
            self._line_plot(
                data = data,
                x =  "Steps", 
                y = "PosError",  
                hue = "Runs", 
                xsteps = xsteps,
                run_paths_list = run_paths_list,
                labels = labels,
                xlabels = xlabels,
                ylabels = ylabels,
                xlim = xlim,
                ylim = ylim,
                show = show,
                save = save,
                save_path = save_path,
                ext = ext
            )   
        else: 
            maxvals = {}
            minvals = {}
            avgvals = {}
            for run in data["Runs"].unique():
                datarun = data[data["Runs"]==run]
                df = pd.DataFrame( )  
                for i,test in enumerate(datarun["Tests"].unique()): 
                    df.insert(loc=len(df.columns), column=str(i), value=datarun[datarun["Tests"]==test]["PosError"].reset_index(drop=True))    
                maxvals[run] = df.max(axis=1)
                minvals[run] = df.min(axis=1)
                avgvals[run] = df.mean(axis=1)
                
            # print(df.iloc[498])
            self._line_plot_nobs(
                maxvals = maxvals,
                minvals = minvals,
                avgvals = avgvals, 
                labels = labels,
                xlabels = xlabels,
                ylabels = ylabels,
                xlim = xlim,
                ylim = ylim,
                show = show,
                save = save,
                save_path = save_path,
                ext = ext
            )
            
 
    ##############################################################################################################################################################
 
    def multirun_statplot_tanklevel_test(self, env_run_ids, labels=[], xlabels=None, ylabels=None, etank_init_list=[], 
                                save=True, show=True, plot_name=None, sub_folder_name="", plot_type="histplot", ext="pdf"): 
        if plot_name == None:
            plot_name = str(len(env_run_ids))
        run_paths_list = [os.path.join(self.out_test_folder, env_run) for env_run in env_run_ids]
        
        data = df_test_multirun_etank(run_paths_list=run_paths_list, etank_init_list=etank_init_list) 
        save_path = os.path.join(self.save_multirun_testing_plots_path, sub_folder_name, f"{plot_name}_{plot_type}_multirun_tanklevel_test.{ext}")

        self._stat_plot(
            data = data,
            x =  "Runs", 
            y = "Level",  
            hue = None,  
            plot_type = plot_type,
            labels = labels, 
            xlabels = xlabels,
            ylabels = ylabels,
            show = show,
            save = save,
            save_path = save_path,
            ext = ext
        )    

 

    ##############################################################################################################################################################
 
    def multirun_statplot_poserror_test(self, env_run_ids, labels=[], xlabels=None, ylabels=None,  save=True, show=True, plot_name=None, 
                                sub_folder_name="", plot_type="histplot", ext="pdf"): 
        if plot_name == None:
            plot_name = str(len(env_run_ids))
        run_paths_list = [os.path.join(self.out_test_folder, env_run) for env_run in env_run_ids]
        
        data = df_test_multirun_errors( run_paths_list=run_paths_list  )
        save_path = os.path.join(self.save_multirun_testing_plots_path, sub_folder_name, f"{plot_name}_{plot_type}_multirun_poserror_test.{ext}")

        self._stat_plot(
            data = data,
            x =  "Runs", 
            y = "Errors",  
            hue = None,  
            plot_type = plot_type,
            labels = labels, 
            xlabels = xlabels,
            ylabels = ylabels,
            show = show,
            save = save,
            save_path = save_path,
            ext = ext
        )    
 