# README

# requirements

- python3
- cmake

# install

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## macOS

You may run into issues installing pygame (a pettingzoo dep). You can try to install its dependencies manually:

```
brew install sdl2 sdl2_image sdl2_mixer sdl2_ttf
```

# run

```
python3 chess_project.py
```

after running, see training progress with:

```
tensorboard --logdir=~/ray_results
```

## tips:

- filter tags for "loss" to see policy/total/vf loss
- all checkpointing is built-in; look in the specified output directory (`~/ray_results`). you can use these model weights in subsequent experiments.
