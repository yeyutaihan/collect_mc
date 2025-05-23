# Collect Minecraft Video and actions

This repo is forked from https://github.com/wilson1yan/collect-minecraft.git. The code is used to collect Minecraft video and actions.

## Setup

You need to install minerl library. The latest version of minerl is 1.0.2. But this version sometimes crashes, which was also reported in https://github.com/minerllabs/minerl/issues/725.

So I recommend to use the version 0.4.4. But while installing this version, one of the dependencies cannot be accessed. So you need to follow the steps in https://github.com/minerllabs/minerl/issues/744.

You may also need to install a specific version of jdk. You can just install it in your conda environment.

## Usage

```bash
sh xvfb_run.sh python collect.py -o <output_dir>
```

More configuration options can be found in the `collect.py` file. The default configuration is to collect 3000 videos of 1200 frames each. `n_parallel` is the number of parallel processes to run. The default is 24, and I recommend to request 32 CPU cores.

You can also configure the parameters in `SimpleAgent` class in `collect.py`.

The action dimension collected by `collect.py` is 4, which represents `forward`, `backward`, `jump` and `turn`.

The action dimension collected by `mem_collect.py` is 8, which represents `forward`, `backward`, `jump`, `turn`, `x`, `y`, `z` and `yaw`.