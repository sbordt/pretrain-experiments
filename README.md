# pretrain-experiments




# Setup

[create conda env]

git clone https://github.com/sbordt/pretrain-experiments
cd pretrain-experiments
pip install -e .



## Setup for experiments with OLMo-2 models ()

quick setup:

[in the same env as pretrain-experiments]

git clone https://github.com/sbordt/OLMo
cd OLMo
git checkout pretrain-experiments
pip install -e .[all] 
pip install h5py

then in your configuration files

framework: 
  type: olmo
  repository_path: [path to the repository]

[link to detailed guide]


## Setup for experiments with OLMo-3 models (olmo_core)

Comming soon!
