# Slurm Cluster Context & Constraints (u:wiki)

This document outlines the hardware capabilities, software environment, and strict operational constraints of the Slurm cluster. Use this context to generate valid, optimized, and rule-compliant Slurm job scripts (`sbatch`) and commands.

## 1. Hardware & Compute Nodes
The cluster consists of several high-performance nodes equipped with varying CPU and GPU resources.

| Node Name | GPUs | Total GPU Memory | CPU Cores (Logical) | Total RAM | Node Group |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **dgx-h100-em2** | 8x NVIDIA H100 | 640 GB | 224 | ~2 TB | VDA |
| **dgx1** | 8x NVIDIA V100 | 128 GB (16GB each) | 80 | ~500 GB | VDA |
| **galadriel** | 4x NVIDIA H100 | ~374 GB (95GB each)| 192 | ~2 TB | DM |
| **shelob** | 8x NVIDIA H200 | ~1.128 TB (141GB each)| 192 | ~2 TB | DM |
| **vader** | 1x NVIDIA H100 | ~80 GB | 64 | ~500 GB | DM |

## 2. Hard Constraints & Rules

### 🛑 What NOT to do
* **NO Anaconda:** Anaconda/Miniconda is strictly banned due to licensing. Never write scripts that use `conda init`, `conda activate`, or refer to Anaconda. Use `module load miniforge` or `module load python` instead.
* **NO Execution on Login Nodes:** Never execute programs, scripts, or containers directly on login nodes. Always use `sbatch` (preferred) or `srun`.
* **NO Native Docker Commands:** Never use `docker run` or `docker compose`. Use Slurm's Pyxis plugin via `srun/sbatch --container-image=...`.
* **NO Unspecified Memory:** The default memory allocation is **8.192 MB per node**, which is far too low for most tasks. Always explicitly request memory using `--mem=<size>`, `--mem-per-cpu=<size>`, or `--mem-per-gpu=<size>`.

### ⚠️ Mandatory Job Parameters
* `--time=d-hh:mm:ss`: Time limits are strictly **required**. Jobs without a time limit will fail.
* `--gres=gpu:N`: Required if requesting GPUs. **Maximum 4 GPUs per job.**

## 3. Environment & Modules

### Python & Conda (Miniforge)
Environments are managed via environment modules. The default Python version is **3.12.x**.
* **Temporary Environments (Recommended):** Setup in fast `tmpfs` (RAM) and deleted after the job.
* **Permanent Environments:** Persist in `$HOME/venvs/$ENV_NAME`.

**Miniforge Workflow:**
```bash
# Example setup in a Slurm script
source /etc/profile.d/modules.sh
export ENV_MODE="temporary" # Or "permanent"
export ENV_NAME="job_${SLURM_JOB_ID}" # Custom name recommended for permanent
module load miniforge
# Install packages quietly to avoid prompts
conda install -y networkx
python main.py
module 


What is the p_low partition?
It is a special partition available to all users designed specifically to soak up any currently unused or idle resources on the cluster. It allows you to run jobs using maximum available resources without blocking other users.

How it Works (The Catch)
Because p_low is meant for idle resources, jobs submitted to this partition have the lowest priority.

If a user submits a "normal" or higher-priority job that requires the GPU your p_low job is currently using, your job will be interrupted and requeued.

It is highly unpredictable when your job will actually finish, as it is entirely dependent on cluster traffic.

Best Practices for Using p_low
Because your jobs are virtually guaranteed to be interrupted at some point, you should only use p_low for specific types of workloads. The documentation recommends it for expert users running:

Jobs with Checkpointing: Long-running models that periodically save their state (checkpoints). If the job is interrupted and requeued, it can simply load the latest save file and continue from there rather than starting from scratch.