 
from ntpath import join
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
                hard_reset = True,
                reward_id = 0
                ):
        super(PendulumEBud, self).__init__(
            env = Pendulum(
                env_name = env_name, 
                folder_path = folder_path,
                max_episode_length = max_episode_length, 
                init_joint_config = init_joint_config,
                hard_reset = hard_reset,
                reward_id=reward_id
            ),
            energy_tank_init = energy_tank_init, # initial energy in the tank
            energy_tank_threshold = energy_tank_threshold, # minimum energy in the tank  
            debug = debug,
            energy_terminate = energy_terminate  
        )
 
    def get_joints(self): 
        joints = np.array([self.env.sim.get_state()[0]])
        # sin_q, cos_q, _ = self.env.get_obs()
        # joints = np.arcsin(sin_q) if cos_q>=0 else -np.pi-np.arcsin(sin_q) 
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
                hard_reset = True,
                reward_id = 0
                ):
        super(PendulumEBudAw, self).__init__(
            env = Pendulum(
                env_name = env_name, 
                folder_path = folder_path,
                max_episode_length = max_episode_length, 
                init_joint_config = init_joint_config,
                hard_reset = hard_reset,
                reward_id=reward_id
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

    def upgrade_reward(self,reward):  
        if self.reward_id == 0:
            new_reward = reward - 0.5*(self.energy_tank_init-self.energy_tank)
        elif self.reward_id == 1:
            new_reward = reward/(reward + 0.5*abs(self.energy_tank-self.energy_tank_init))
        return new_reward
            