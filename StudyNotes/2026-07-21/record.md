# Record｜2026-07-21

## 1. Generate first simulation episode

### Goal

Complete the first ACT simulation data generation pipeline:

Cloud environment → ACT environment → MuJoCo simulation → HDF5 episode data.

### ACT Simulation Data Generation

Checked script usage:
```bash
python record_sim_episodes.py --help
```

Required arguments:
* `--task_name`

* `--dataset_dir`

* `--num_episodes`


Checked README:
```bash
grep -n "sim_transfer_cube" README.md
```

Selected task:
```
sim_transfer_cube_scripted
```

Because the cloud server has no display:
```bash
export MUJOCO_GL=egl
```

Generated one simulation episode:
```bash
python record_sim_episodes.py \\

 --task_name sim_transfer_cube_scripted \\

 --dataset_dir ./data/sim_transfer_cube_scripted \\

 --num_episodes 1

```

Result:
Successfully generated: ./data/sim_transfer_cube_scripted/episode_0.hdf5
File size:352 MB


### HDF5 Data Structure Inspection

Used h5py to inspect:
```python

with h5py.File(path, "r") as f:

    f.visit(print)
```


Dataset structure:
```
action

observations

├── images

│   └── top

├── qpos

└── qvel
```


Data shapes:
```
action: (400, 14)

observations/qpos: (400, 14)

observations/qvel: (400, 14)

observations/images/top: (400, 480, 640, 3)
```


Interpretation:
* Episode length: 400 timesteps

* Action dimension: 14

* Robot state dimension:

    qpos: 14

    qvel: 14

* Camera:

    400 RGB images

    resolution: 480×640


### Current Status

Completed:

✅ Cloud GPU environment

✅ ACT Conda environment

✅ PyTorch CUDA verification

✅ MuJoCo/dm_control import check

✅ Generated first simulation episode

✅ Verified HDF5 structure

---

## 2. Run minimal ACT training smoke test

### Deepen the understanding of ACT

#### (1) Overall Training Pipeline

完整训练流程：

HDF5 episode
↓
Dataset pipeline
↓
DataLoader
↓
ACT Policy
↓
Forward
↓
Loss computation
↓
Backward
↓
Optimizer update
↓
Checkpoint


#### (3) Dataset Pipeline

数据来源：

HDF5 episode

包含：

/observations/images/<camera>
/observations/qpos
/observations/qvel
/action


读取流程：

constants.py
↓
SIM_TASK_CONFIGS[task_name]
↓
task_config['dataset_dir']
↓
dataset_dir
↓
load_data()
↓
EpisodicDataset()


EpisodicDataset.__getitem__():

读取：
- image
- qpos
- qvel
- action sequence

处理：
- image normalization
- qpos normalization
- action normalization
- padding action sequence


最终返回：

image_data
qpos_data
action_data
is_pad


Batch进入模型：

image:
[B, N_cam, C, H, W]

qpos:
[B, 14]

action:
[B, chunk_size, action_dim]

is_pad:
[B, chunk_size]



#### (3) Training Entry

入口：

imitate_episodes.py / main()


调用流程：

main(args)

↓

load_data()

↓

train_bc(
    train_dataloader,
    val_dataloader,
    config
)



#### (4) Policy Creation

创建流程：

train_bc()

↓

make_policy()

↓

policy.py / ACTPolicy


ACTPolicy作用：

- 包装 ACT model
- 区分 training / inference
- 计算 loss
- 提供 optimizer


真正模型：

policy.py / ACTPolicy

↓

detr/models/detr_vae.py / DETRVAE


其中：

DETRVAE =
CVAE encoder
+
Transformer policy
+
Action decoder


#### (5) Forward Process


Training:

输入：

image
+
qpos
+
expert action sequence


↓

CVAE encoder

expert action sequence
+
qpos

↓

latent z


↓

ACT Transformer decoder

image feature
+
qpos
+
z


↓

predicted action chunk


输出：

a_hat



#### (6) Loss


Loss定义位置：

policy.py


组成：

L1 action reconstruction loss

+
KL divergence loss

↓

ACT total loss


其中：

L1:

预测 action chunk

vs

expert action chunk


KL:

约束 latent z 分布接近 prior



#### (7) Optimization


训练循环：

imitate_episodes.py / train_bc()


流程：

forward

↓

loss

↓

loss.backward()

↓

optimizer.step()


更新：

ACT policy parameters



#### (8) Checkpoint


保存：

policy state_dict

包括：

- backbone parameters
- Transformer parameters
- action head parameters
- latent related parameters


用于之后：

evaluation / rollout
