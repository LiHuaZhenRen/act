# 对 ACT 项目的初步认知

ACT 是一个机器人模仿学习项目，全称是：Action Chunking with Transformers

它的核心思想是：  
模型不是每次只预测一个动作，而是根据当前观察，一次预测未来一段动作序列，也就是 action chunk。

### 1. 项目是否需要真实机器人

初学和复现阶段不需要真实机器人。

这个项目支持仿真环境，可以先在 MuJoCo + dm_control 中运行：

```text
sim_transfer_cube_scripted
sim_insertion_scripted
```

真实机器人部分需要额外安装 ALOHA，因此可以先不碰。

### 2. 项目是否有仿真环境

有。

主要文件是：

```text
sim_env.py
ee_sim_env.py
assets/
```

其中 `assets/` 里包含机器人、桌面、任务场景等 XML 和 STL 文件。

### 3. 项目是否提供数据集

提供。

README 中给了 Google Drive 数据集链接。  
同时项目也可以自己生成仿真数据：

```bash
python record_sim_episodes.py \
  --task_name sim_transfer_cube_scripted \
  --dataset_dir ./data/sim_transfer_cube_scripted \
  --num_episodes 1
```

### 4. 项目是否需要 GPU

训练基本需要 GPU。

代码里大量使用：

```python
.cuda()
```

所以默认环境是 NVIDIA GPU + CUDA。  
如果只是阅读代码，不需要 GPU。  
如果要正式训练和评估，最好有 GPU。

### 5. 安装依赖是否复杂

偏复杂，涉及：

```text
PyTorch
CUDA
MuJoCo
dm_control
h5py
opencv
einops
DETR 本地安装
```

相比普通 Python 项目，机器人仿真和深度学习项目的环境搭建更容易遇到版本问题。

### 6. ACT 项目的主要文件分工

```text
README.md
项目说明和运行命令

constants.py
任务配置、数据目录、episode 长度、相机配置、机器人常量

record_sim_episodes.py
生成仿真演示数据

scripted_policy.py
手写规则策略，用来控制机器人完成任务

sim_env.py
关节空间控制的仿真环境

ee_sim_env.py
末端执行器空间控制的仿真环境

utils.py
数据读取、归一化、DataLoader 构造

policy.py
ACTPolicy / CNNMLPPolicy 的封装，负责调用模型并计算 loss

imitate_episodes.py
训练和评估入口

detr/
ACT 模型本体，基于 DETR/Transformer 修改
```

### 7. ACT 项目的数据流

目前理解的数据流是：

```text
scripted_policy.py
-> 控制机器人在 ee_sim_env 中完成任务
-> 得到一条演示轨迹

record_sim_episodes.py
-> 重放轨迹到 sim_env
-> 保存 qpos / qvel / images / action 到 HDF5

utils.py
-> 从 HDF5 中读取 episode
-> 随机选择一个时间点
-> 构造 image、qpos、未来 action chunk、is_pad

imitate_episodes.py
-> 创建 dataloader
-> 创建 policy
-> 训练模型

policy.py + detr/
-> 输入当前图像和 qpos
-> 输出未来一段 action
-> 计算 L1 loss 和 KL loss

eval
-> 加载 checkpoint
-> 让模型在仿真环境中控制机器人
-> 根据 reward 计算成功率
```

### 8. 第一个复现目标

第一个可执行目标不是直接训练，而是：

```text
生成并可视化 1 条仿真 episode
```

因为这个目标可以同时验证：

```text
环境是否安装成功
MuJoCo 是否能运行
仿真场景是否能加载
scripted policy 是否能工作
HDF5 数据是否能保存
visualize 脚本是否能读取数据
```

实现顺序：

```text
1. 生成 1 条 scripted 仿真数据
2. 可视化 episode
3. 打开 HDF5 检查数据结构
4. 用少量数据试跑训练
5. 正式训练 ACT
6. eval 测试成功率
```

### 9.Environment Check: VMware Ubuntu

Commands:

```bash
nvidia-smi
lspci | grep -i nvidia
lspci | grep -i vga
python3 --version
```

Results:

- `nvidia-smi` not found.
- No NVIDIA device found by `lspci`.
- VGA device is `VMware SVGA II Adapter`.
- System Python is 3.12.3.
- PyTorch is not installed.

Conclusion:

This VMware Ubuntu environment is useful for Linux practice, but it cannot access NVIDIA GPU/CUDA directly, so it is not suitable for formal ACT training.
```
