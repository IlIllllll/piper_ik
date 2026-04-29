# UMI Arms State Visualizer

这个目录独立于 Piper 的 `scripts/`，用于查看 `log_umi_arms_state.py` 生成的 UMI 左右臂日志。

启动示例数据：

```bash
python umi_visualizer/umi_state_visualizer.py
```

查看真实日志：

```bash
python umi_visualizer/umi_state_visualizer.py \
  --log-file /tmp/umi_arms.log
```

实时监听 UMI WebSocket 数据。默认会连接 `ws://192.168.150.27:8100/ws`，
并订阅 `state` topic；网页通过 WebSocket 自动更新，不需要手动刷新：

```bash
python umi_visualizer/umi_state_visualizer.py \
  --live
```

指定上游 WebSocket：

```bash
python umi_visualizer/umi_state_visualizer.py \
  --live \
  --ws-url ws://192.168.150.27:8100/ws
```

如果仍需要兼容 UDP 广播，可以显式切换入口。下面命令会在本机 `0.0.0.0:8211`
接收 UDP 包，并只接受源 IP 为 `192.168.150.27` 的数据：

```bash
python umi_visualizer/umi_state_visualizer.py \
  --live \
  --live-transport udp \
  --data-bind-host 0.0.0.0 \
  --data-source-host 192.168.150.27 \
  --data-port 8211
```

页面默认监听：

```text
http://127.0.0.1:8060/
```

支持内容：

- `robot_target`、`UMI raw`、`B feedback` 三类日志源切换
- 左右手轨迹显示
- XY / XZ / YZ 投影
- 时间轴播放与逐帧查看
- x/y/z/roll/pitch/yaw/grip 曲线
