# Passive RL #
Framework to train a RL agent with passivity 

## TO DO 

- [ ] Find Agent's parameters for environment in gym-mujoco
- [ ] Test adding obstacles in the xml envs
- [ ] train a manipulator like UR5 or Panda  

## Installation ##

Setup virtual environment

```
conda env create -f rl.yml
```


Add the following line to .bashrc

```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/kvn/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so
```


```
pip install -e .
```




## Test ##