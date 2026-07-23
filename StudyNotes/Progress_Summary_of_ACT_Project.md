# ACT 项目阶段性复现总结

## 1. 项目背景与目标

ACT，全称 **Action Chunking with Transformers**，是一个面向机器人操作任务的模仿学习项目。它的核心思想不是让机器人每次只预测一个单步动作，而是根据当前 observation，一次预测未来一段动作序列，即 **action chunk**。这一点已经在前期项目认知记录中明确整理过：ACT 是机器人模仿学习项目，核心是根据当前观察预测未来一段动作序列。

从具身智能路线看，ACT 属于 **imitation learning / behavior cloning** 方向，而不是强化学习或显式 World Model 方向。它不主要依靠 reward 来训练，而是通过 demonstration 数据学习专家动作。训练时，模型学习“在当前图像和机器人状态下，专家接下来会怎么动”；eval 时，再把训练出的 policy 放回仿真环境，看它能否实际完成任务。

ACT 适合作为我的第一个具身智能开源项目，原因有三点。第一，它有明确的论文目标和开源代码结构，适合训练“paper → code → experiment”的连接能力。第二，它同时包含仿真环境、数据生成、HDF5 数据读取、PyTorch 训练、checkpoint 保存和 eval rollout，能够让我经历一次比较完整的机器人学习项目链条。第三，它和我之前了解/简单复现过的 Transformer、ViT、VLA、World Model 都有联系：Transformer 提供了序列建模基础；ViT 帮助我理解 image token / representation 的概念；VLA 让我理解 observation 到 action 的路线；World Model 则提供了另一个对照方向，即预测环境状态变化，而 ACT 当前更偏向“直接学习动作”。

目前我对 ACT 的定位可以概括为：**它不是让模型理解世界如何变化，而是让模型模仿专家在视觉和关节状态条件下如何连续行动。**

## 2. 本阶段完成内容

本阶段的工作不是单纯“跑通一个命令”，而是从一个陌生机器人学习项目中逐步建立项目地图，并完成一次从数据生成到 eval 的最小闭环。

首先，在项目阅读层面，我开始形成阅读开源项目的基本方法：先看 README，找配置文件，理解数据契约，再定位数据生成、数据读取和训练入口。我的 notebook 中已经把这个方法整理为“寻找项目的主线和数据流”，并明确指出 README、`constants.py`、`record_sim_episodes.py`、`utils.py / EpisodicDataset` 和 `imitate_episodes.py` 分别承担入口地图、任务配置、数据生成、数据读取和训练入口的作用。

其次，在环境配置层面，我完成了阿里云 PAI DSW 上的 ACT 运行环境配置。记录中包括云端平台、NVIDIA A10 GPU、驱动、CUDA、PyTorch、Conda 环境 `aloha`、Python 3.9、PyTorch 2.0.0、pytorch-cuda 11.8、torchvision 0.15.0 等信息，并记录了从 clone 仓库到创建 conda 环境、激活环境、导出环境备份的命令。

第三，在 Linux 和命令行能力上，我学习并使用了 `pwd`、`ls`、`cd`、`mkdir -p`、`touch`、`echo`、`cat`、`grep`、`du -sh`、`df -h`、`history` 等基础命令，也整理了 `~`、`.`、`..`、`/`、绝对路径和相对路径等路径概念。 这部分能力虽然基础，但对云端复现非常关键，因为 ACT 的数据、checkpoint、log 都依赖清晰的路径管理。

第四，在仿真数据生成层面，我成功使用 `record_sim_episodes.py` 生成了第一条 `sim_transfer_cube_scripted` episode。记录中写明了通过 `--help` 查看脚本参数、从 README 中搜索任务名、设置 `MUJOCO_GL=egl` 处理云服务器无显示器问题，并最终生成 `episode_0.hdf5`，文件大小约 352 MB。 这一步验证了云端环境、MuJoCo / dm_control、scripted policy、仿真场景和 HDF5 数据保存链路。

第五，在 HDF5 数据检查层面，我使用 `h5py` 查看了生成数据的结构和 shape。记录显示，episode 中包含 `action`、`observations/images/top`、`observations/qpos`、`observations/qvel`；其中 action、qpos、qvel 的 shape 均为 `(400, 14)`，图像为 `(400, 480, 640, 3)`，说明一次 episode 有 400 个 timestep，每个动作和关节状态维度为 14。

第六，在训练层面，我完成了 ACT 的 smoke training。训练目标不是获得高 success rate，而是验证完整训练链路：`HDF5 episode → EpisodicDataset → DataLoader → ACTPolicy → DETRVAE → loss → backward → checkpoint`。这一训练目标和命令已经记录在 2026-07-22 的 smoke training 记录中。

第七，在 eval 层面，我完成了最小 eval 闭环：checkpoint 成功加载，仿真 rollout 成功启动，并保存了 rollout video。记录中显示，`./smoke_ckpt/policy_best.ckpt` 成功加载，simulation rollout 成功，Rollout 0 的结果为 `episode_return=0`、`episode_highest_reward=0`、`env_max_reward=4`、`Success=False`，并保存了 `smoke_ckpt/video0.mp4`。 这说明虽然策略没有完成任务，但 checkpoint → policy → environment → reward 的链路已经跑通。

总体来说，本阶段的能力成长主要体现在：我不只是运行了一个机器人项目，而是开始能追踪一个真实具身智能项目的数据流、训练流和 eval 流。

## 3. ACT 数据流与训练流程

本阶段我理解的 ACT 数据流如下：

```text
record_sim_episodes.py
→ HDF5 episode
→ utils.py / EpisodicDataset
→ DataLoader
→ image / qpos / action / is_pad
→ policy.py / ACTPolicy
→ DETRVAE
→ predicted action chunk
→ loss
→ checkpoint
→ eval rollout
```

第一步，`record_sim_episodes.py` 负责生成仿真数据。它让仿真环境中的 scripted policy 控制机器人执行任务，并把机器人在执行过程中的 observation 和 action 保存为 `.hdf5` 文件。也就是说，HDF5 episode 是后续训练的数据来源，不是普通图片分类中的静态图片文件夹。

第二步，HDF5 episode 保存了整条轨迹。当前检查到的数据包括 `action`、`qpos`、`qvel` 和 top camera 图像。这里的第一维 400 表示一个 episode 中有 400 个时间步，说明 ACT 的数据不是单个样本，而是一段连续机器人操作轨迹。

第三步，`utils.py / EpisodicDataset` 负责从 HDF5 中读取数据。根据记录，`EpisodicDataset.__getitem__()` 会读取 image、qpos、qvel 和 action sequence，并处理 image normalization、qpos normalization、action normalization 和 padding action sequence，最终返回 `image_data`、`qpos_data`、`action_data` 和 `is_pad`。

第四步，DataLoader 把单条样本组织成 batch。当前理解中，进入模型前的 batch 包括：

```text
image:  [B, N_cam, C, H, W]
qpos:   [B, 14]
action: [B, chunk_size, action_dim]
is_pad: [B, chunk_size]
```

其中 image 是相机图像，qpos 是机器人当前关节状态，action 是专家未来动作序列，is_pad 用于标记 padding 部分。

第五步，`policy.py / ACTPolicy` 是 policy 的封装层。它负责包装 ACT 模型、区分 training / inference、计算 loss，并提供 optimizer。真正的模型主体在 `detr/models/detr_vae.py / DETRVAE` 中，包括 CVAE encoder、Transformer policy 和 action decoder。

第六步，训练时的 forward 过程大致是：

```text
image + qpos + expert action sequence
→ CVAE encoder: qpos + expert action sequence → latent z
→ ACT Transformer decoder: image feature + qpos + z → predicted action chunk
→ 输出 a_hat
```

这个理解目前是结构级理解，还没有深入到 DETRVAE 内部每一层张量的精确变化。

第七步，loss 包括 L1 action reconstruction loss 和 KL divergence loss。L1 loss 比较预测 action chunk 与 expert action chunk；KL loss 约束 latent z 的分布接近 prior。记录中已经明确写出 ACT total loss 由 L1 和 KL 两部分构成。

第八步，训练循环执行：

```text
forward
→ loss
→ loss.backward()
→ optimizer.step()
```

optimizer 更新 ACT policy 的模型参数，包括 backbone、Transformer、action head 和 latent 相关参数。checkpoint 保存的是训练后的 policy state_dict，可用于之后 evaluation / rollout。

第九步，eval 阶段加载 checkpoint，并让 policy 在仿真环境中控制机器人。此时没有 expert action，也不再计算训练 loss，而是通过环境返回的 reward / success 判断任务完成情况。我的 eval smoke test 已经验证 checkpoint loading、policy inference、environment interaction 和 reward evaluation 这一链路。

## 4. 我对核心原理的理解

### imitation learning

目前理解：imitation learning 是让模型从专家示教中学习动作策略。ACT 中不是通过 reward 直接训练 policy，而是用 demonstration 中的 expert action 作为监督信号，让模型学习在某个 observation 下专家会如何行动。

掌握程度：初步理解。还没有系统学习 behavior cloning、distribution shift、DAgger 等理论。

### demonstration / episode

目前理解：demonstration 是专家完成任务的一段演示轨迹；episode 是这段轨迹在仿真环境中保存下来的完整数据，包括多个 timestep 的图像、机器人状态和动作。当前生成和检查过的 episode 长度为 400 timestep。

掌握程度：基本理解数据形态，但对 episode 中时间对齐、动作控制频率等细节仍不熟。

### observation

目前理解：observation 是机器人 policy 在某一时刻能看到或读取到的信息。在 ACT 中主要包括图像 image 和机器人关节状态 qpos。

掌握程度：基本理解，但对不同机器人任务中 observation 设计的差异还需要继续学习。

### qpos

目前理解：qpos 是机器人当前关节位置状态。在当前任务中 qpos shape 为 `(400, 14)`，表示每个 timestep 有 14 维关节状态。

掌握程度：知道 qpos 的数据作用，但还没有深入理解每一维分别对应哪个关节。

### action

目前理解：action 是机器人要执行的控制指令。在当前 ACT 数据中 action shape 为 `(400, 14)`，与 qpos 维度一致，表示每个 timestep 的动作维度为 14。

掌握程度：知道 action 是训练标签和控制输出，但对关节空间控制、末端空间控制、位置/速度/力矩控制的区别仍需补充。

### action chunking

目前理解：ACT 的核心不是预测单个下一步动作，而是一次预测未来一段动作序列。这样可以降低有效决策步数，并缓解逐步预测造成的误差累积问题。

掌握程度：概念上理解，但还没有深入分析 temporal aggregation 和 chunk_size 对控制效果的影响。

### Transformer policy

目前理解：Transformer policy 用于根据图像特征、qpos、latent z 等信息预测 action chunk。它和我之前学过的 Transformer / ViT 有联系：都是把输入组织成可被 attention 处理的表示，再输出任务所需的结果。

掌握程度：结构级理解。还没有逐层读完 DETRVAE、Transformer decoder、query embedding、action head 的具体实现。

### CVAE

目前理解：CVAE 在训练阶段使用 expert action sequence 和 qpos 编码 latent z，使模型能够表达 demonstration 中可能存在的动作风格或轨迹差异。推理时没有 expert action，因此使用 prior / 默认 latent。

掌握程度：初步理解。对 VAE/CVAE 的概率建模、reparameterization、KL 项的数学含义仍需补。

### KL loss

目前理解：KL loss 用来约束 latent z 的分布，使其不要偏离 prior 太远。这样推理时使用 prior 或默认 z 才有意义。

掌握程度：知道作用，但数学细节尚未完全掌握。

### L1 loss

目前理解：L1 loss 比较预测 action chunk `a_hat` 与 expert action chunk `a` 的差距，是 ACT imitation learning 中主要的动作重建损失之一。记录中已明确 L1 是 predicted action chunk vs expert action chunk。

掌握程度：基本理解。

### checkpoint

目前理解：checkpoint 保存训练后的 policy 参数。ACT eval 中通过 `--ckpt_dir` 找到 `policy_best.ckpt` 并加载。我的 eval 记录显示 `./smoke_ckpt/policy_best.ckpt` 已成功加载。

掌握程度：基本理解。对 checkpoint 中具体 state_dict key 和模型结构如何对应还未深入。

### rollout

目前理解：rollout 是把训练好的 policy 放到环境中，从 reset 开始让它一步步输出动作并与环境交互，观察一整段任务执行结果。我的 eval 已经完成 rollout，并保存了 video。

掌握程度：基本理解。

### reward / success rate

目前理解：reward 是环境根据任务完成程度给出的评价信号；success rate 是多个 rollout 中成功完成任务的比例。ACT training 不靠 reward 更新模型，但 eval 需要 reward / success 判断 policy 是否真的能完成任务。当前 smoke checkpoint 的 reward 为 0、success 为 False 是正常的，因为训练规模非常小。

掌握程度：初步理解。对 reward 在 `sim_env.py` / `ee_sim_env.py` 中的具体分级定义还需要继续读代码。

## 5. 遇到的问题与解决方式

### Linux 与云端环境问题

最开始在 VMware Ubuntu 中检查时，发现没有 NVIDIA GPU，`nvidia-smi` 不可用，`lspci` 也没有 NVIDIA device，只有 VMware SVGA II Adapter，因此该环境适合 Linux 练习，但不适合 ACT 正式训练。后来切换到阿里云 PAI DSW，并配置了 A10 GPU 环境。

在 DSW 中，还遇到 `/root` 和 `/mnt/workspace` 路径区别的问题。记录中已经明确建议项目代码、数据、checkpoint 和 log 放在 `/mnt/workspace`，因为这是工作盘。

### CUDA / GPU 环境理解问题

本阶段完成了 CUDA / PyTorch 可用性检查，并在 `aloha` 环境中确认 PyTorch、CUDA、torchvision 等关键依赖。云端环境记录显示 GPU 为 NVIDIA A10 24GB，PyTorch 环境为 2.0.0 + pytorch-cuda 11.8。

同时曾遇到 `ImportError: libtorch_cpu.so: undefined symbol: iJIT_NotifyEvent`。根据 notebook 记录，判断该问题发生在 `import torch` 阶段，因此不是模型代码问题，而是 PyTorch 底层库兼容问题；最终通过将 MKL 降到 `mkl=2024.0` 解决。

### Python 文件路径与 HDF5 读取问题

本阶段学习了 Linux 绝对路径、相对路径、当前目录、上一级目录等概念，也学习了如何用 `h5py.File(path, "r")` 和 `f.visit(print)` 查看 HDF5 结构。HDF5 检查结果显示数据结构和 shape 符合 ACT 训练需要。

薄弱点是：目前能够按示例读取 HDF5，但对 HDF5 group / dataset 的抽象、切片读取、批量处理和异常处理还不熟。

### MuJoCo / dm_control / 仿真环境问题

生成仿真 episode 时，由于云服务器没有显示器，出现过 `DISPLAY environment variable is missing` 相关问题。notebook 中记录的解决方式是设置：

```bash
export MUJOCO_GL=egl
```

使用 GPU 离屏渲染，同时不使用 `--onscreen_render`。

此外，在尝试生成更多 episode 时实例曾崩溃，但目前材料中没有完整系统日志，所以没办法判断是否由内存、渲染后端、云实例稳定性或其他资源问题导致。

### PyTorch 训练与 shape 理解问题

本阶段已经能说清楚 ACT batch 进入模型前包含 image、qpos、action、is_pad，并能理解 image、qpos、action 的基本 shape。训练流程也已经整理为 DataLoader → ACTPolicy → Forward → Loss → Backward → Optimizer → Checkpoint。

但对模型内部 shape 的理解仍然有限，尤其是 CNN backbone 输出、Transformer 输入、query embedding、latent z 拼接方式、action head 输出维度等，还没有系统追踪。

### eval / checkpoint 加载问题

本阶段成功完成 eval smoke test。checkpoint `./smoke_ckpt/policy_best.ckpt` 成功加载，simulation rollout 成功，保存了 rollout video。

需要注意的是，eval 命令中的模型结构参数必须与训练时一致，例如 `policy_class`、`chunk_size`、`hidden_dim`、`dim_feedforward` 等，否则可能出现 checkpoint shape mismatch。本阶段没有记录发生 shape mismatch，因此不编造该错误。

### 对 ACT 原理的理解困难

1. 为什么推理时没有 expert action 仍然可以使用 latent z / prior，且把z置为0，如此一来学习latent z 是不是做了无用功；
2. 为什么 imitation learning 训练不用 reward，但 eval 仍然需要 reward / success rate。

目前这些问题已经有初步解释，但还没有从论文公式、CVAE 数学或代码实现层面完全掌握。

## 6. 对 Research Readiness 的提升

### Paper Readiness

这次复现使我更能理解一篇具身智能论文“到底在做什么”。以前读论文时，我可能只停留在 abstract、模型结构图和核心概念上；现在通过 ACT，我能把论文中的 action chunking、Transformer policy、CVAE、demonstration、rollout 等概念对应到具体代码文件、数据结构和命令输出中。

但这个能力仍处在初级阶段。我目前能理解 ACT 的工程主线和基本思想，但还不能独立推导方法细节，也不能完整评价 ACT 与其他 imitation learning 方法的优劣。

### GitHub Readiness

GitHub Readiness 提升比较明显。我已经不再只是 clone 项目后盲目运行，而是开始按照 README、配置文件、数据契约、数据生成入口、数据读取入口、训练入口的顺序建立项目地图。这个方法已经在 notebook 中整理成一套可复用流程。

ACT 的文件分工也已经基本建立：`record_sim_episodes.py` 负责生成仿真数据，`utils.py` 负责数据读取和 DataLoader，`policy.py` 负责 policy 封装和 loss，`imitate_episodes.py` 是训练和评估入口，`detr/` 是模型主体。

不过，目前我对项目的理解仍然依赖导师提示。真正独立阅读陌生大型机器人 repo 时，还需要更多练习。

### Coding Readiness

Coding Readiness 的提升主要体现在三方面。第一，我完成了云端 Linux / Conda / CUDA / PyTorch 环境配置。第二，我能通过终端命令检查路径、文件、环境、依赖和输出。第三，我经历了真实复现中的报错定位，包括 PyTorch / MKL 兼容问题、MuJoCo headless 渲染问题、内部 package 安装问题、episode 数量不足导致 train split 为空等。

但我仍然不是熟练工程实现者。Python 文件读写、HDF5 操作、PyTorch Dataset / DataLoader、模型内部 shape tracing、Linux 环境管理都还需要继续补。

## 7. 当前薄弱点

1. **Python 文件读写**
   能使用简单命令和示例代码，但对路径拼接、异常处理、批量读写还不熟。

2. **h5py / HDF5**
   已经能查看结构和 shape，但还不熟悉 group / dataset 抽象、切片读取和高效数据处理。

3. **PyTorch Dataset / DataLoader**
   已经知道 `EpisodicDataset` 返回 image、qpos、action、is_pad，但还不能完全独立写出一个类似数据集类。

4. **ACT 的 CVAE 与 Transformer 内部**
   知道 CVAE encoder、latent z、Transformer policy、action head 的作用，但还没有逐层读 DETRVAE 的 forward。

5. **imitation learning 理论**
   知道 behavior cloning 和 demonstration 的基本思想，但对 distribution shift、compounding error、DAgger、多模态示教等理论还只是初步接触。

6. **机器人仿真环境**
   已经知道仿真环境包含 robot、world、observation、action interface、reward / success，但对 MuJoCo XML、physics engine、contact、control mode 仍不熟。

7. **GPU / CUDA 使用**
   能检查 GPU 和 PyTorch CUDA，但对 CUDA version、driver、pytorch-cuda、系统库之间的关系还需要继续建立地图。

8. **eval 指标解释**
   能解释 reward / success rate 的基本作用，但还没有具体读完 transfer cube 任务中 reward=1/2/3/4 分别对应什么事件。

9. **实验记录规范**
   已经开始记录命令、结果、错误和解释，但还可以进一步规范：每次实验应固定记录 commit hash、环境、命令、数据版本、输出文件、是否 push、哪些 artifact 不进 Git。

## 8. 下一步计划

考虑在两个方向中选一个：
一是继续深入 ACT 模型结构，重点读 `DETRVAE.forward()`、CVAE encoder、query embedding 和 action head；
二是对比 ACT 与 Diffusion Policy，理解两种 imitation learning policy 的核心差异。
不考虑立刻开启大规模正式训练，因为训练效果提升不一定直接带来理解提升。

## 9. 一句话总结

本阶段我完成了 ACT 从项目阅读、环境配置、仿真数据生成、HDF5 检查、smoke training 到 eval rollout 的最小复现闭环；虽然还没有掌握 ACT 的全部模型细节和 imitation learning 理论，但已经从“会跑单个深度学习例子”推进到“能读懂并复现一个真实具身智能开源项目的主线流程”。
