# Notebook｜2026-07-20

## Linux commands learned

- `pwd`: 打印当前所在文件夹的完整路径
- `ls`: 查看当前目录下的文件和文件夹
- `cd`: 切换目录
- `mkdir -p`: 创建目录，如果父目录不存在也一起创建
- `touch`: 创建空文件
- `echo`: 写入 （echo "hello linux" > README.md）
- `cat`: 查看文件内容，或向文件写入内容
- `grep`: 搜索文本（grep -r "hello" . # grep搜索文字；-r 即recursive，递归搜索所有子文件；. 即当前目录）
- `du -sh`: 查看目录或文件大小
- `df -h`: 查看磁盘使用情况
- `history`: 查看历史命令
	history | grep pip # 定向查找pip命令
	history | head -20 # 查看最初20条命令
	history | tail -20 # 查看最后20条命令

## Path concepts

- `~`: 当前用户的 home 目录。（eg. ls ~ 列出主目录中的所有文件）
- `.`: 当前目录
- `..`: 上一级目录
- `/`: Linux 根目录
- 绝对路径：从 `/` 开始，例如 `/root/projects/act`
- 相对路径：相对于当前目录，例如 `../record.md`

## Conda concepts

- `conda env create -f conda_env.yaml`: 根据环境文件创建环境
- `conda info --envs`: 查看已有环境
- `conda list | grep torch`: 查看和 torch 相关的包

