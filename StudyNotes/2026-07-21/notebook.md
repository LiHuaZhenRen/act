# Notebook｜2026-07-20 

---

# 1. Linux命令行写入文件

## 方法1：使用重定向写入

```bash
cat > filename.md <<'EOF'
content
EOF
````

含义：

* `cat > filename.md`：
  创建文件并写入内容
* `EOF`：
  作为输入结束标志

注意：

当终端出现：

```text
...
```

说明当前命令还没有结束，需要输入结束标志。

---

## 方法2：简单写入

```bash
echo "content" > filename.md
```

适合写入少量内容。

---

# 2. Linux路径与文件位置问题

DSW中：

```text
/root
```

和：

```text
/mnt/workspace
```

不是同一个位置。

推荐：

* 项目代码
* 数据
* checkpoint
* log

放在：

```text
/mnt/workspace
```

因为这是工作盘。

---

# 3. PyTorch import报错分析

错误：

```
ImportError:
libtorch_cpu.so: undefined symbol: iJIT_NotifyEvent
```

## 判断流程

第一步：

发现错误发生在：

```python
import torch
```

说明不是模型代码问题，而是PyTorch环境问题。

第二步：

观察：

```
undefined symbol
```

含义：动态链接库找不到需要的函数。

通常意味着：

```
Python包版本冲突
↓
底层C/C++库不兼容
```

---

## 本次问题原因

环境：

```
PyTorch 2.0.0
CUDA 11.8
MKL 2024.2.2
```

MKL版本过新，旧PyTorch + 新MKL底层库不兼容

---

## 解决原则

不要直接：

* 重装整个环境
* 升级PyTorch
* 修改CUDA

采用：

```
定位冲突包
↓
最小修改
↓
保持项目原始环境
```

最终：

```bash
conda install mkl=2024.0
```

解决。

---

# 4. 如何从README和--help找到运行命令

不要直接复制别人命令。

标准流程：

## Step 1：找到功能对应脚本

目标：

生成仿真数据

因此寻找：

```
record_sim_episodes.py
```

---

## Step 2：查看脚本参数

执行：

```bash
python record_sim_episodes.py --help
```

得到：

```
--task_name
--dataset_dir
--num_episodes
--onscreen_render
```

因此知道：

运行必须提供：

```text
任务名称
数据保存路径
episode数量
```

---

## Step 3：从README寻找具体参数

查看：

```bash
grep -n "sim_transfer_cube" README.md
```

得到：

```
sim_transfer_cube_scripted
```

因此：

```bash
--task_name sim_transfer_cube_scripted
```

---

## Step 4：最小化测试

科研复现不要一开始跑大实验。

选择：

```bash
--num_episodes 1
```

先验证：

* 环境
* 数据生成
* 文件输出

---

## Step 5：处理云服务器无GUI问题

错误：

```
DISPLAY environment variable is missing
```

原因：

服务器没有显示器。

解决：

使用：

```bash
export MUJOCO_GL=egl
```

表示：

使用GPU离屏渲染。

同时不要：

```bash
--onscreen_render
```

---

最终命令：

```bash
export MUJOCO_GL=egl

python record_sim_episodes.py \
--task_name sim_transfer_cube_scripted \
--dataset_dir ./data/sim_transfer_cube_scripted \
--num_episodes 1
```

---

# 5. Linux终端中运行Python代码

方法1：

进入交互模式：

```bash
python
```

出现：

```
>>>
```

可以直接输入Python代码。

退出：

```python
exit()
```

或者：

```
Ctrl+D
```

---

方法2：

直接运行一段代码：

```bash
python -c "import torch; print(torch.__version__)"
```

适合简单测试。

---

方法3：

多行脚本：

```bash
python <<'PY'
代码
PY
```

适合快速实验。

---

# 6. 生成第一条ACT仿真episode

目标：

验证：

```
仿真环境
↓
专家策略
↓
数据保存
```

执行：

```bash
python record_sim_episodes.py \
--task_name sim_transfer_cube_scripted \
--dataset_dir ./data/sim_transfer_cube_scripted \
--num_episodes 1
```

结果：

生成：

```
data/
└── sim_transfer_cube_scripted/
    └── episode_0.hdf5
```

说明：

ACT数据生成流程成功。

---

# 7. HDF5数据检查

## 为什么检查？

确认：

```
生成的数据
↓
是否符合训练需要
```

---

## 查看结构

Python：

```python
import h5py

with h5py.File(path,"r") as f:
    f.visit(print)
```

得到：

```
action

observations
├── images
│   └── top
├── qpos
└── qvel
```

---

## 查看shape

```python
print(f["action"].shape)
print(f["observations/qpos"].shape)
print(f["observations/images/top"].shape)
```

结果：

```
action:
(400,14)

qpos:
(400,14)

qvel:
(400,14)

image:
(400,480,640,3)
```

理解：

第一维：

```
400 timestep
```

表示一次episode包含400个时间状态。
