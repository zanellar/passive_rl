 
import numpy as np 
from mjrlenvs.envrl.pendulum import Pendulum  
from passive_rl.scripts.ebudenv import EBudBaseEnv,EBudAwEnv
 
class PendulumEBud(EBudBaseEnv): 

    def __init__(self,    
                max_episode_length = 50, 
                energy_tank_init = 10, # initial energy in the tank
                energy_tank_threshold = 0, # minimum energy in the tank   
                init_joint_config = "random",
                debug = False,
                energy_terminate = False,
                folder_path = None,
                env_name = "pendulum",
                hard_reset = True
                ):
        super(PendulumEBud, self).__init__(
            env = Pendulum(
                env_name = env_name, 
                folder_path = folder_path,
                max_episode_length = max_episode_length, 
                init_joint_config = init_joint_config,
                hard_reset = hard_reset
            ),
            energy_tank_init = energy_tank_init, # initial energy in the tank
            energy_tank_threshold = energy_tank_threshold, # minimum energy in the tank  
            debug = debug,
            energy_terminate = energy_terminate  
        )
 
    def get_joints(self):
        obs = self.env.get_obs()
        joints = np.arcsin(obs[0])
        return joints

    def get_torques(self):
        return self.action
            


class PendulumEBudAw(EBudAwEnv): 

    def __init__(self,    
                max_episode_length = 50, 
                energy_tank_init = 10, # initial energy in the tank
                energy_tank_threshold = 0, # minimum energy in the tank   
                init_joint_config = "random",
                debug = False,
                energy_terminate = False,
                folder_path = None,
                env_name = "pendulum",
                hard_reset = True
                ):
        super(PendulumEBudAw, self).__init__(
            env = Pendulum(
                env_name = env_name, 
                folder_path = folder_path,
                max_episode_length = max_episode_length, 
                init_joint_config = init_joint_config,
                hard_reset = hard_reset
            ),
            energy_tank_init = energy_tank_init, # initial energy in the tank
            energy_tank_threshold = energy_tank_threshold, # minimum energy in the tank  
            debug = debug,
            energy_terminate = energy_terminate  
        )
 
    def get_joints(self):
        obs = self.env.get_obs()
        joints = np.arcsin(obs[0])
        return joints

    def get_torques(self):
        return self.action

    # def upgrade_reward(self,reward):
    #     reward += 0.05*(self.energy_tank-self.energy_tank_init)
    #     return 
            