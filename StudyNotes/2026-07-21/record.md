# Record｜2026-07-21



## Generate first simulation episode



### Goal

Complete the first ACT simulation data generation pipeline:

Cloud environment → ACT environment → MuJoCo simulation → HDF5 episode data.



---



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

&#x20; --task_name sim_transfer_cube_scripted \\

&#x20; --dataset_dir ./data/sim_transfer_cube_scripted \\

&#x20; --num_episodes 1

```



Result:



Successfully generated: ./data/sim_transfer_cube_scripted/episode_0.hdf5



File size:352 MB



---



### HDF5 Data Structure Inspection



Used h5py to inspect:



```python

with h5py.File(path, "r") as f:

&#x20;   f.visit(print)

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

action:

(400, 14)



observations/qpos:

(400, 14)



observations/qvel:

(400, 14)



observations/images/top:

(400, 480, 640, 3)

```



Interpretation:



* Episode length: 400 timesteps

* Action dimension: 14

* Robot state dimension:



&#x20; * qpos: 14

&#x20; * qvel: 14

* Camera:



&#x20; * 400 RGB images

&#x20; * resolution: 480×640



---



### Current Status



Completed:



✅ Cloud GPU environment

✅ ACT Conda environment

✅ PyTorch CUDA verification

✅ MuJoCo/dm_control import check

✅ Generated first simulation episode

✅ Verified HDF5 structure



Next:



1. Inspect image content and action values.

2. Understand ACT dataset loading (`EpisodicDataset`).

3. Run minimal ACT training smoke test.

