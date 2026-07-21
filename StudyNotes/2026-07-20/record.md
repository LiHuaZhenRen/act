# ACT Reproduction | 2026-07-20

## Goal
Prepare Alibaba Cloud Linux + CUDA + ACT simulation environment.

## Cloud environment
- Platform: Alibaba Cloud PAI DSW
- GPU: NVIDIA A10 24GB
- Driver: 550.163.01
- CUDA shown by nvidia-smi: 12.4
- Base Python: 3.11.15
- Base PyTorch: 2.11.0+cu126
- Base torch.cuda.is_available(): True

## ACT repo
- Repo: https://github.com/LiHuaZhenRen/act
- Local path: `~/projects/act`
- Commit hash: `b90dcd7d5763269dc4b47628141d78ff63a101db`

## Conda environment
- Environment name: `aloha`
- Created from: `conda_env.yaml`
- Python: 3.9
- PyTorch: 2.0.0
- pytorch-cuda: 11.8
- torchvision: 0.15.0
- Status: environment created successfully and activated successfully

## Commands used
```bash
cd ~/projects
git clone https://github.com/LiHuaZhenRen/act
cd act
git rev-parse HEAD
cat conda_env.yaml
conda env create -f conda_env.yaml
conda init bash
source ~/.bashrc
conda activate aloha
conda info --envs
conda list | grep torch
conda env export > ~/projects/act/aloha_environment_backup.yml

