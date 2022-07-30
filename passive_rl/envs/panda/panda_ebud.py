import gym
import numpy as np
from gym import spaces 
from pandamjrl.envs.reach import EnvReach


class PandaEBud(gym.Env): 

    def __init__(self,    
                max_episode_length = 50, 
                energy_tank_init = 7, # initial energy in the tank
                energy_tank_threshold = 0, # minimum energy in the tank  
                debug = False
                ):
        super(PandaEBud, self).__init__()

        self.debug = debug
        self.t = 0
        self.T = max_episode_length

        ################# Init Learning Environment ####################

        # Vanilla Environment
        self._env = EnvReach(
            max_episode_length=max_episode_length, 
            init_joint_config = [0, -0.5, 0, -2.5, 0, 0, 0, 0, 0]
            )  

        # Vanilla State
        self.state = self._env._get_obs()  
      
        # Observations 
        self.observation_space = self._env.observation_space 

        # Actions  
        self.action_space = self._env.action_space 

        ################# Init Energy-budgeting Framework ####################
 
        self.energy_tank_init = energy_tank_init
        self.energy_tank = energy_tank_init  
        self.energy_tank_threshold = energy_tank_threshold

        self.energy_joints = [0,0,0,0,0,0,0] # list containing the energy going out from the 7 motors
        self.energy_exchanged = 0 # initializing total energy going out at a time step to 0 
 
        self.energy_stop_ct = 0
    
    def _update_energy_tank(self, action, state_new):
        joints = np.arccos(state_new[:7])   
        joints_old = np.arccos(self.state[:7])
        d_joints = joints - joints_old
        self.energy_joints = action*d_joints  
        self.energy_exchanged = sum(self.energy_joints)
        self.energy_tank -= self.energy_exchanged

        is_empty = self.energy_tank <= self.energy_tank_threshold
        return is_empty
  
    def step(self, action):  
  
        # Vanilla Environment Step
        state_new, reward, _done, info = self._env.step(action) 

        # Energy Budgeting  
        energy_done = self._update_energy_tank(action,state_new) 
 
        done = energy_done or _done 

        if energy_done:
            self.energy_stop_ct += 1
         
        info = dict(energy_exchanged = self.energy_exchanged,
                    energy_tank = self.energy_tank, 
                    _info = info)   

        self.state = state_new 
        self.t += 1 
 

        return self.state, reward, done, info

    def reset(self, goal=None):  
        self._env.reset() 
        self.t = 0  
        self.energy_tank = self.energy_tank_init
        self.energy_exchanged = 0.0   
        return self.state 

    def render(self, mode=None): 
        self._env.render()

    def close(self):
        self._env.close()
         