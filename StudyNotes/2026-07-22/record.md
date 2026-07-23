# Record | 2026-07-22 

## 1.ACT Smoke Training Success

### Goal

Run the minimal ACT training loop:

HDF5 episode → EpisodicDataset → DataLoader → ACTPolicy → DETRVAE → loss → backward → checkpoint

### Command

```bash
python imitate_episodes.py \
--task_name sim_transfer_cube_scripted \
--ckpt_dir ./smoke_ckpt \
--policy_class ACT \
--kl_weight 10 \
--chunk_size 50 \
--hidden_dim 256 \
--batch_size 2 \
--dim_feedforward 1024 \
--num_epochs 1 \
--lr 1e-5 \
--seed 0

```

## 2.ACT Eval Smoke Test

### Goal

Verify:

checkpoint
→ policy
→ simulation rollout
→ reward


### Command

python imitate_episodes.py \
...
--eval


### Checkpoint

Loaded successfully:

./smoke_ckpt/policy_best.ckpt


### Result

Checkpoint loading:

SUCCESS

Simulation rollout:

SUCCESS


Example:

Rollout 0

episode_return=0

episode_highest_reward=0

env_max_reward=4

Success=False


Video:

Saved:
smoke_ckpt/video0.mp4


### Interpretation

The inference pipeline is working.

Failure is expected because this checkpoint was only trained for smoke testing.

The purpose was verifying:

checkpoint loading,
policy inference,
environment interaction,
and reward evaluation.