# Piper IK 与数据集轨迹回放

本项目围绕 `piper_description/urdf/piper_description.urdf` 提供三类可视化工具，并把数据集读取和 IK 逻辑拆成独立工具模块：

- `scripts/piper_ik_visualizer.py`: 手动控制 XYZ/Yaw/Pitch/Roll 的 Pinocchio IK 可视化界面。
- `scripts/replay_piper_dataset_web3d.py`: 读取数据集，在网页中用 MeshCat 三维显示机械臂轨迹回放。
- `scripts/replay_piper_target_axes_web3d.py`: 读取数据集，只显示目标姿态 3D 坐标轴移动路线，不做机械臂逆解。
- `scripts/piper_dataset_reader.py`: Piper 数据集读取、episode 定位和目标位姿轨迹构建工具。
- `scripts/piper_ik_utils.py`: Piper Pinocchio/Pink 逆解、FK、关节限位和夹爪映射工具。

默认数据集为：

```text
20260412_panda_dual_pika_32_27/
```

默认 URDF 为：

```text
piper_description/urdf/piper_description.urdf
```

## 环境安装

```bash
cd /Users/bingcm/program/piper_ik
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-ik.txt
```

依赖主要包括：

- `pin`: Pinocchio Python 包
- `meshcat`: 三维网页可视化
- `pyarrow`: 读取 parquet 数据集
- `scipy`: 姿态旋转计算

## 3D 数据集回放网页

启动：

```bash
source .venv/bin/activate
python scripts/replay_piper_dataset_web3d.py
```

默认地址：

```text
3D control panel: http://127.0.0.1:8020/
MeshCat viewer:    http://127.0.0.1:7060/static/
```

控制页会嵌入 MeshCat 三维视图，并提供：

- `Play/Pause`
- `Prev/Next`
- 帧滑块
- 播放速度
- Episode 下拉选择
- 夹爪开合显示
- 当前帧 IK 收敛状态与误差
- 末端轨迹线、采样点、ghost 机械臂姿态和当前机械臂姿态

网页中可以直接用 `Episode` 下拉框切换同一数据集根目录下的不同 episode。也可以在启动时指定初始 episode：

```bash
python scripts/replay_piper_dataset_web3d.py \
  --dataset-root 20260412_panda_dual_pika_32_27 \
  --episode-index 10
```

也可以直接指定单个 parquet 文件：

```bash
python scripts/replay_piper_dataset_web3d.py \
  --file 20260412_panda_dual_pika_32_27/data/chunk-000/episode_000010.parquet
```

`--episode-index` 会根据 `meta/info.json` 里的 `data_path` 和 `chunks_size` 自动定位文件。`--file` 和 `--episode-index` 只能二选一；两个都不传时默认使用 episode 0。

当前数据集每一帧是相对上一帧的位姿增量，所以默认使用：

```text
--delta-mode incremental
```

也就是：

```text
pose[k + 1] = pose[k] ⊕ delta[k]
```

如果你想把每行解释为相对 episode 起点的位姿，可以显式使用：

```bash
python scripts/replay_piper_dataset_web3d.py --delta-mode from-start
```

## 目标姿态坐标轴轨迹网页

这个界面不做机械臂逆解，也不显示整机模型，只显示目标姿态的 3D 坐标轴和移动路线：

```bash
source .venv/bin/activate
python scripts/replay_piper_target_axes_web3d.py
```

默认地址：

```text
Target axes control panel: http://127.0.0.1:8030/
MeshCat viewer:             http://127.0.0.1:7070/static/
```

界面支持：

- `Play/Pause`
- `Prev/Next`
- 帧滑块
- Episode 下拉选择
- 当前目标姿态 `xyz/rpy`
- 轨迹线、采样点、ghost 坐标轴和当前坐标轴

默认从 `--initial-joints` 通过 URDF FK 得到起始末端姿态，然后按数据集的相对位姿逐帧累加。也可以直接给起始目标姿态，这样不会读取 URDF 做 FK：

```bash
python scripts/replay_piper_target_axes_web3d.py \
  --initial-pose 0.035 0.000 0.273 -0.018 1.571 -0.018
```

常用参数：

```bash
python scripts/replay_piper_target_axes_web3d.py \
  --dataset-root 20260412_panda_dual_pika_32_27 \
  --episode-index 10 \
  --axis-length 0.07
```

## 初始位姿调整

离线 IK 默认以以下 6 个关节角作为初始关节姿态：

```text
0.000, 0.004, -0.281, -0.000, 0.364, 0.000
```

可以通过命令行覆盖：

```bash
python scripts/replay_piper_dataset_web3d.py \
  --initial-joints 0.000 0.004 -0.281 -0.000 0.364 0.000
```

如果只想在这个初始姿态对应的末端位姿上做 XYZ 偏移或额外抬升，使用以下参数，不需要改代码：

```bash
python scripts/replay_piper_dataset_web3d.py \
  --initial-x-offset -0.2 \
  --initial-y-offset 0.0 \
  --initial-z-offset 0.0 \
  --z-lift 0.03
```

参数含义：

```text
--initial-x-offset   初始 x 偏移，单位米；向后 20cm 可设为 -0.2
--initial-y-offset   初始 y 偏移，单位米
--initial-z-offset   初始 z 偏移，单位米
--z-lift             初始 z 抬升，单位米；默认 0.0
--initial-joints     离线 IK 初始关节姿态，6 个关节角
```

如果你的坐标系里“向后”是 X 正方向，把 `--initial-x-offset -0.2` 改成 `--initial-x-offset 0.2`。

对应代码位置：

- `scripts/piper_dataset_reader.py` 的 `build_pose_plan()`
- `scripts/replay_piper_dataset_web3d.py` 的 argparse 参数定义

## 手动 IK 控制界面

启动：

```bash
python scripts/piper_ik_visualizer.py
```

默认会打开本地控制面板，使用 Pinocchio 做 IK，使用 MeshCat 显示模型。界面支持：

- X/Y/Z 增量移动
- Yaw/Pitch/Roll 增量旋转
- 键盘快捷键
- 当前末端姿态与目标姿态显示

常用参数：

```bash
python scripts/piper_ik_visualizer.py --ee-frame gripper_base
python scripts/piper_ik_visualizer.py --control-port 8011
python scripts/piper_ik_visualizer.py --meshcat-port 7050
python scripts/piper_ik_visualizer.py --open-meshcat
```

## 数据列说明

默认读取右臂：

```text
observation.state.arm.right.end_effector_pose
observation.state.arm.right.end_effector_value
```

`end_effector_pose` 为 6 维：

```text
[dx, dy, dz, droll, dpitch, dyaw]
```

脚本默认将每一帧解释为相对上一帧的位姿增量：

```text
--delta-mode incremental
```

左臂可通过参数切换列名，例如：

```bash
python scripts/replay_piper_dataset_web3d.py \
  --pose-column observation.state.arm.left.end_effector_pose \
  --gripper-column observation.state.arm.left.end_effector_value
```

3D 网页会把 `end_effector_value` 映射到 URDF 夹爪关节：

```text
joint7: 0.0 -> 0.035
joint8: 0.0 -> -0.035
```

默认按 `0=闭合, 1=张开` 显示。如果数据集语义相反，启动时加：

```bash
python scripts/replay_piper_dataset_web3d.py --invert-gripper
```

## 端口与进程

常用端口：

```text
8011 / 7050: 手动 IK 控制界面
8020 / 7060+: 数据集 3D 回放网页
8030 / 7070+: 目标姿态坐标轴轨迹网页
```

如果 MeshCat 默认端口被占用，脚本会自动使用后续可用端口，并在终端日志中打印实际地址。

当前后台 3D 回放进程如果由本会话启动，可用以下命令停止：

```bash
kill -INT $(cat /tmp/piper_dataset_web3d_8020.pid)
```

手动 IK 控制界面后台进程：

```bash
kill -INT $(cat /tmp/piper_ik_visualizer_8011.pid)
```

## 常见问题

### 打开 127.0.0.1 没有数据

必须带端口，例如：

```text
http://127.0.0.1:8020/
```

不要只打开：

```text
http://127.0.0.1/
```

### MeshCat 页面空白或端口变化

如果 `7060` 被占用，脚本可能自动切到 `7061`、`7062` 等端口。以终端打印的 `MeshCat:` 地址或控制页中的 iframe 地址为准。

### 离线 IK 部分帧不收敛

离线预览从 URDF 的默认初始姿态开始，和真实机械臂当前姿态可能不同。可以调整：

```bash
--initial-x-offset
--initial-y-offset
--initial-z-offset
--z-lift
--pos-scale
--rot-scale
```

网页回放使用 URDF 的离线初始姿态作为累加起点。
