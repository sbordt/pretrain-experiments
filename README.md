# pretrain-experiments




# Setup

[create conda env]

conda create -n pretrain-experiments python=3.12
conda activate pretrain-experiments

[install pretrain-experiments]

git clone https://github.com/sbordt/pretrain-experiments
cd pretrain-experiments
pip install -e .


## Setup for experiments with OLMo-2 models ()

conda activate pretrain-experiments
git clone https://github.com/sbordt/OLMo
cd OLMo
git checkout pretrain-experiments
pip install -e .[all] 
pip install h5py

[for flash attention:]
pip install psutil ninja packaging 
pip install flash_attn --no-build-isolation

then in your configuration files

framework: 
  type: olmo
  repository_path: [path to the repository]

The example experiments assume that the OLMo folder lies in the location as the pretrain-experiments package.

[link to detailed guide]


## Setup for experiments with OLMo-3 models (olmo_core)

Comming soon!
