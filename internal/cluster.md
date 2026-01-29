

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

# ferranti flash attn install
conda create -n pretrain-experiments python=3.12
conda activate pretrain-experiments

pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130

conda install gxx=13 gcc=13 -c conda-forge
pip install flash-attn --no-build-isolation
