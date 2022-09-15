 
import numpy as np 
import gym
from pyparsing import replaceWith 
 
class EBudBaseEnv(gym.Env): 

    def __init__(self,    
                env, 
                energy_tank_init = 7, # initial energy in the tank
                energy_tank_threshold = 0, # minimum energy in the tank  
                debug = False,
                testing_mode = False 
                ):
        super(EBudBaseEnv, self).__init__()

        self.testing_mode = testing_mode
        self.debug = debug 

        ################# Init Learning Environment ####################

        # Vanilla Environment
        self.env = env
 
        # Observations 
        self.observation_space = self.env.observation_space 

        # Actions  
        self.action_space = self.env.action_space 

        # Initialize    
        self.obs = self.env.get_obs()  
        self.action = np.zeros(self.env.action_space.shape)
        self.reward = None 
        self.done = False
        self.info = {}

        ################# Init Energy-budgeting Framework ####################
 
        self.energy_tank_init = energy_tank_init
        self.energy_tank = energy_tank_init  
        self.energy_tank_threshold = energy_tank_threshold
        self.energy_avaiable = True 

        self.energy_stop_ct = 0

        self.joints = None
        
    def _update_energy_tank(self):
        ''' 
        Etank(t) = Etank(t-1) - Eex(t)
        Ex(t) = sum(tau(t-1,i)*(q(t,i) - q(t-1,i)), i=1,2)
        ''' 

        joints = self.get_joints()
        torques = self.get_torques()

        if self.joints is None:
            self.joints = joints
            return False

        old_joints = self.joints 
        new_joints = joints 
        d_joints = new_joints - old_joints
        self.energy_joints = torques*d_joints  
        self.energy_exchanged = sum(self.energy_joints)
        self.energy_tank -= self.energy_exchanged 
        tank_is_empty = self.energy_tank <= self.energy_tank_threshold
        self.energy_avaiable = not tank_is_empty
        self.joints = new_joints
        return tank_is_empty

    def step(self, action):  
  
        if not self.energy_avaiable: 
            action *= 0  

        # Vanilla Environment Step
        _obs, _reward, _done, _info = self.env.step(action) 
  
        # Energy Budgeting  
        tank_is_empty = self._update_energy_tank() 
     
        if self.testing_mode: 
            done = _done 
        else:
            done = tank_is_empty or _done  
 
        if tank_is_empty:
            self.energy_stop_ct += 1
         
        info = dict(energy_exchanged = self.energy_exchanged,
                    energy_tank = self.energy_tank, 
                    _info = _info)   

        self.action = action
        self.obs = _obs  
        self.reward = _reward
        self.done = done
        self.info = info

        return self.obs, self.reward, self.done, self.info

    def reset(self, goal=None):   
        self.obs = self.env.reset() 
        self.action = np.zeros(self.env.action_space.shape) 
        self.energy_tank = self.energy_tank_init
        self.energy_exchanged = 0.0  
        return self.obs 

    def render(self, mode=None): 
        self.env.render()
   
    def get_sample(self):
        return self.obs, self.action, self.reward, self.done, self.info