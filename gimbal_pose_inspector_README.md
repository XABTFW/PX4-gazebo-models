# Gazebo 云台位姿只读验证

`gimbal_pose_inspector.py` 读取 Gazebo 实时位姿，验证相机安装、坐标变换和理想投影。
它不连接 PX4 串口、uORB 或 MAVLink，也不发送云台/飞机控制指令。
它不产生识别、识别框、锁定或跟随状态。

## 启动

先运行带相机的模型，并在同一个 Gazebo world 中加载用于观察的目标：

```bash
make px4_sitl gz_fpv_quadtailsitter_gimbal
```

在第二个终端中查询实际实体名：

```bash
/usr/bin/python3 Tools/simulation/gimbal_pose_inspector.py --world default --list
```

从列表中选出本机和目标的模型名，替换下面的 `本机模型名` 和 `目标模型名`。
使用 Gazebo 的实际名称（可能有 `_0`、`_1` 后缀），不要使用 `4023_...` 的 airframe 文件名。
目标必须与本机位于同一 world。脚本也必须使用相同的 `GZ_PARTITION`。

```bash
/usr/bin/python3 Tools/simulation/gimbal_pose_inspector.py \
  --world default \
  --observer 本机模型名 \
  --target 目标模型名 \
  --output /tmp/gimbal_geometry.jsonl
```

`--output` 不允许覆盖已有文件；再次运行需换一个文件名。省略它时 JSONL 输出到终端。
按 Ctrl+C 退出。可用 `--duration 10` 记录十秒；`--rate 10` 指墙钟日志采样频率。
依赖本机安装的 `gz.transport13` 和 `gz.msgs10` Python 绑定，应使用 `/usr/bin/python3`。

## 数据依据和坐标约定

- 从 `/world/<world>/scene/info` 只读服务获得父子层级及实体 ID。
  重名链节通过 `模型名::链节名` 区分，不按裸 `camera_link` 名字猜测本机。
- 从 `/world/<world>/pose/info` 获取 `Pose_V`。模型/链节位姿相对于父实体；
  沿父子层级组合变换才能得到世界位姿。
  每次计算使用同一条消息中的全部所需位姿，缺少任一层级就报告无效，不拼接历史位姿。
- 相机参数默认读取仓库的
  `Tools/simulation/gz/models/fpv_quadtailsitter_gimbal/model.sdf`：
  1280×720，水平视场 2.0 rad，near/far 为 0.05/15000 m，传感器相对
  `camera_link` 的位置为 `(0.031, 0, 0)` m。运行模型必须与此 SDF 配置一致；
  若使用另一份 SDF，显式传入 `--camera-sdf`。这些参数来自文件，未与渲染器标定。
- 相机采用 Gazebo FLU 坐标：`+X` 前、`+Y` 左、`+Z` 上。
  链节的实时姿态来自 Gazebo，零位安装姿态不替代运动中的真实反馈。
- 观察点是目标**模型原点**，不是飞机几何中心、图像识别中心或包围框。
- 理想针孔投影使用方形像素：`f = width / (2*tan(hfov/2))`，
  `u = (width-1)/2 - f*y/x`，`v = (height-1)/2 - f*z/x`。
  像素中心采用左上角 `(0,0)` 约定。带畸变或自定义镜头/内参的 SDF 会被拒绝。

## JSONL 字段

第一行 `type=metadata` 记录模型名、SDF 路径、相机参数、传感器安装位姿和计算假设。
后续 `type=sample` 包含：

| 字段 | 含义 |
| --- | --- |
| `valid` / `reason` | 计算是否有效；无效时记录原因，不输出假角度和假距离 |
| `sim_time_s` / `sim_epoch` | Gazebo 时间；仿真时间回退时 epoch 递增 |
| `pose_age_wall_s` | 消息到达后经过的墙钟秒数 |
| `source_sequence` | 接收到的消息序号；重复日志行可能引用同一条消息 |
| `distance_m` | 相机传感器原点到目标模型原点的三维距离 |
| `target_in_camera_flu_m` | 目标模型原点在相机 FLU 坐标中的位置 |
| `azimuth_right_deg` | 右偏为正，`atan2(-y,x)` |
| `elevation_up_deg` | 上偏为正，`atan2(z,hypot(x,y))` |
| `target_origin_pixel_uv` | 理想像素投影；在相机后方时为 null；可位于画幅外 |
| `target_origin_in_frustum` | 目标原点是否位于画幅和 near/far 裁剪范围内 |
| `camera_in_observer_rpy_flu_deg` | 相机相对本机模型 FLU 坐标的 roll/pitch/yaw；不是关节角或 DYT 框架角 |
| `camera_world_position_m` | 相机传感器世界位置 |
| `camera_world_quaternion_wxyz` | 相机传感器到世界的旋转，四元数 w/x/y/z 顺序 |

消息超过 `--max-age`（默认 0.5 秒）、仿真时间停止推进、实体缺失、四元数异常、
非有限值或相机与目标重合时均输出 `valid=false`。
三维位置和角度都是仿真真值及其几何计算结果，没有添加“仿真传感器噪声”。

`target_origin_in_frustum=true` 只表示原点通过几何视锥检查。
脚本不检查遮挡、图像是否发布、光照、目标可辨识尺寸或分类结果；不能把这个标志当成视觉检测成功。
Euler 角在俯仰 ±90° 附近具有表示奇异性，姿态比较应优先使用四元数。

## 验证

```bash
/usr/bin/python3 Tools/simulation/test_gimbal_pose_inspector.py -v
```

自动检查覆盖：层级组合、嵌套模型、重名链节、前后/上下/左右投影、裁剪边界、
该 SDF 的朝上相机和传感器偏移、异常数据、过期、暂停和时间重置。
这些是离线测试，不代表已完成你正在运行的 SITL 联调。

父子位姿语义依据 Gazebo Sim 8 的
[SceneBroadcaster 实现](https://github.com/gazebosim/gz-sim/blob/gz-sim8/src/systems/scene_broadcaster/SceneBroadcaster.cc)，
Python 接口依据
[Gazebo Transport 13 官方文档](https://gazebosim.org/api/transport/13/python.html)。
