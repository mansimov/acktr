### Dependencies
* python 2.7
* tensorflow 0.12

python3 and tensorflow>=1.0 version available at [https://github.com/openai/baselines](https://github.com/openai/baselines)

### Example of training a model

```sh run_train_reacher.sh```

### Example of loading a model (trained from pixels)

Change this line [https://github.com/openai/gym/blob/master/gym/envs/mujoco/mujoco_env.py#L118]( https://github.com/openai/gym/blob/master/gym/envs/mujoco/mujoco_env.py#L118)
from
```self.viewer = mujoco_py.MjViewer()```
to
```self.viewer = mujoco_py.MjViewer(visible=False, init_width=160, init_height=210)```

Then run

```sh run_load_pixels.sh```

Pretrained models available at [http://www.cs.nyu.edu/~mansimov/mujoco-pixels-policies](http://www.cs.nyu.edu/~mansimov/mujoco-pixels-policies)
