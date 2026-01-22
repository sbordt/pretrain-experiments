













# TODO's

- standardize evals (release evals?)

- when re-starting with long training stuff, will always restart from initial checkpoint again

allow max_steps to depend only on olmo config (just start once and let the olmo train until it ends).

- save init when training from scratch

it is not good that the environment variables that are passed to the training script are in the .sh file. should be in .yaml file

- add option to automatically upload final checkpoint to hf_hub (also for intermediate checkpoints)

add  (asynch?) evals during training.

- refactor num_steps per control and num_steps to work for the different szenatios	

- find a better option to decay / script change than restart with new yaml? although this is probably fine...

- handle config logic with better checks ... perhaps define a large args structure?

- change olmo insertion functions to be more low-level, that is turn off splitting and auto-corrections should be possible

- move olmo insertions fully towards hdf5







