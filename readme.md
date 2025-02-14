# Passive RL #
Within a robotic context, the techniques of passivity-based control and reinforcement learning are merged with the goal of eliminating some of their reciprocal weaknesses, as well as inducing novel promising features in the resulting framework. The contribution is framed in a scenario where passivity-based control is implemented by means of virtual energy tanks, a control technique developed to achieve closed-loop passivity for any arbitrary control input. Albeit the latter result is heavily used, it is discussed why its practical application at its current stage remains rather limited, which makes contact with the highly debated claim that passivity-based techniques are associated with a loss of performance. The use of reinforcement learning allows to learn a control policy that can be passivized using the energy tank architecture, combining the versatility of learning approaches and the system theoretic properties which can be inferred due to the energy tanks. Simulations show the validity of the approach, as well as novel interesting research directions in energy-aware robotics.

 
 
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

Inside the folder:
```
pip install -e .
```


## Mujoco ##

### Contact Forces bug
Issue: https://github.com/openai/mujoco-py/pull/487
Need to modify 'mujoco_py/mjviewer.py' in the conda env as follow:
https://github.com/openai/mujoco-py/pull/487/commits/ab026c1ff8df54841a549cfd39374b312e8f00dd

# Reference #
Zanella, R., Palli, G., Stramigioli, S., & Califano, F. (2024). Learning passive policies with virtual energy tanks in robotics. IET Control Theory & Applications, 18(5), 541-550.

## Cite ##
```
@article{zanella2024learning,
  title={Learning passive policies with virtual energy tanks in robotics},
  author={Zanella, Riccardo and Palli, Gianluca and Stramigioli, Stefano and Califano, Federico},
  journal={IET Control Theory \& Applications},
  volume={18},
  number={5},
  pages={541--550},
  year={2024},
  publisher={Wiley Online Library}
}
```
