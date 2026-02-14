

# cluster

ssh -A sbordt10@134.2.168.205

ssh -A sbordt10@galvani-login.mlcloud.uni-tuebingen.de    
ssh -A sbordt10@galvani-login2.mlcloud.uni-tuebingen.de


# cpu 
salloc --time=12:00:00 --nodes=1 --ntasks=1 --mem=128G --partition=cpu-galvani --cpus-per-task=2

salloc --time=12:00:00 --nodes=1 --ntasks=1 --mem=64G --partition=cpu-ferranti --cpus-per-task=2

# 1 gpu
salloc --time=12:00:00 --nodes=1 --ntasks=1 --mem=128G --partition=a100-galvani --cpus-per-task=6 --gres=gpu:1

salloc --time=12:00:00 --nodes=1 --ntasks=1 --mem=128G --partition=h100-ferranti --cpus-per-task=6 --gres=gpu:1

# 4 gpus

salloc --time=12:00:00 --nodes=1 --ntasks=1 --mem=512G --partition=a100-galvani --cpus-per-task=16 --gres=gpu:4 

srun --pty bash -l
conda activate pretrain-experiments
cd /mnt/lustre/work/luxburg/sbordt10/pretrain-experiments/pretrain-experiments

# galvani storage usage
/usr/sbin/lfs quota -hg 4018 /mnt/lustre

du -sh */


# ferranti image with cuda 12.8 & flash_attn

singularity shell --nv --bind /weka/luxburg/sbordt10:/weka/luxburg/sbordt10 pretrain-experiments.sif
export OLMES_EXECUTABLE=~/venvs/olmes/bin/olmes
python -m pretrain_experiments config/olmo-3.yaml


cat > pretrain-experiments.def << 'EOF'
Bootstrap: docker
From: nvidia/cuda:12.8.0-devel-ubuntu24.04

%post
    apt-get update && apt-get install -y python3.12 python3.12-venv python3-pip git
    update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1
    
    pip install --break-system-packages torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu128
    pip install --break-system-packages packaging psutil ninja h5py
    pip install --break-system-packages https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl 
    

%environment
    export LC_ALL=C
EOF


# olmes installation

pip install --upgrade transformers