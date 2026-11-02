# Gazebo 相机目标检测、锁定、跟踪与采集

默认是独立、只读的相机工具，订阅 `gz.msgs.Image`。
显式使用 `--dyt-model` 可在 SITL 发布 `dyt_target`，
再加 `--follow-gimbal` 可控制仿真云台将图像目标保持在中心。
它实现真实画面预览、运动目标检测、单目标锁定与持续跟踪，以及无叠字 PNG 保存。
仅依赖现有 OpenCV / NumPy，不需要下载模型权重。

## 运行

本机已检查 `/usr/bin/python3` 能导入 OpenCV、NumPy、Gazebo Transport 13 和 Msgs 10。
先启动带相机模型：

```bash
make px4_sitl gz_fpv_quadtailsitter_gimbal
```

另开终端查询实际 topic：

```bash
/usr/bin/python3 Tools/simulation/gz_camera_reader.py --list-topics
```

选择该飞机 `camera_link` 下相机的图像 topic；不要选择 `/camera_info` 或 IMU topic。
用 `gz topic -i -t 实际图像topic` 检查消息类型为 `gz.msgs.Image`。
不要直接假设飞机的实例后缀、world 名或相机 topic。
两个终端应使用相同的 `GZ_PARTITION`。

将下面的 `实际图像topic` 替换为查到的 topic，预览：

```bash
/usr/bin/python3 Tools/simulation/gz_camera_reader.py --topic 实际图像topic
```

默认自动锁定最大的运动候选区域。黄色框表示运动候选，绿色框和十字表示有效锁定目标。
检测器先预热 10 个不同时间戳的处理帧，期间不自动锁定。
鼠标左键拖出矩形可选定或替换目标；松开后，在下一个新时间戳图像上锁定该区域。
按 R 清除锁定并重新预热；按 Q / Esc，关闭窗口或按 Ctrl+C 退出。
在有图形会话的终端运行预览；无图形环境用 `--headless`。

只显示运动候选、由鼠标框选目标：

```bash
/usr/bin/python3 Tools/simulation/gz_camera_reader.py --topic 实际图像topic --manual-lock
```

无窗口模式指定首帧目标区域（像素坐标 `x,y,width,height`）：

```bash
/usr/bin/python3 Tools/simulation/gz_camera_reader.py \
  --topic 实际图像topic --headless --manual-lock --roi 200,120,80,60
```

框选区域必须位于画面内，宽高至少 4 像素，并包含非黑色外观。
非法区域会输出 `selection_error`，该次选择不生效。
尽量紧贴目标框选，避免包含大面积背景。

记录 10 秒到一个**不存在的新目录**：

```bash
/usr/bin/python3 Tools/simulation/gz_camera_reader.py \
  --topic 实际图像topic \
  --headless \
  --duration 10 \
  --rate 10 \
  --record-dir /tmp/gz_camera_capture
```

目录中包含 `frame_000000001.png` 等文件及 `frames.jsonl`。
程序不会覆盖已有采集目录；重复运行请换目录。
不指定目录时，只预览/输出元数据，不保存图片。

## DYT 输入与摄像头跟随（SITL 试用）

### 双机协同：系统 ID 4 跟随系统 ID 1

模型名、PX4 实例号和 MAVLink 系统 ID 是不同的标识。
本配置中带朝上相机的 `fpv_quadtailsitter_gimbal` 使用系统 ID 4，
`x500` 使用系统 ID 1。启动脚本支持通过 `PX4_PARAM_MAV_SYS_ID` 显式指定系统 ID；
未指定时仍使用实例号加一。修改运行中的 `MAV_SYS_ID` 不能同步更新所有模块缓存，
请结束原来的两架仿真后，重新启动。

在仓库根目录先编译：

```bash
make px4_sitl_default
```

终端一启动带相机的跟随机（保留原实例 0）：

```bash
PX4_PARAM_MAV_SYS_ID=4 \
PX4_PARAM_CRDZ_ACT_AUX=-1 PX4_PARAM_CRDZ_ACT_BTN=1 \
PX4_SYS_AUTOSTART=4023 PX4_SIM_MODEL=gz_fpv_quadtailsitter_gimbal \
GZ_IP=127.0.0.1 ./build/px4_sitl_default/bin/px4
```

待 Gazebo 启动后，终端二启动目标机（保留原实例 2）：

```bash
PX4_PARAM_MAV_SYS_ID=1 \
PX4_GZ_MODEL_POSE="5,0,0.3,0,0,0" \
PX4_SYS_AUTOSTART=4001 PX4_SIM_MODEL=gz_x500 \
GZ_IP=127.0.0.1 ./build/px4_sitl_default/bin/px4 -i 2
```

`PX4_GZ_MODEL_POSE` 的格式是 `x,y,z,roll,pitch,yaw`，位置单位米，角度单位弧度。
上述目标机从 Gazebo 世界坐标 `(5,0,0.3)` 生成，避免与原点处的跟随机重叠。
该参数只影响新模型生成；同名模型已存在时，需要先删除旧模型或重新启动 Gazebo。

两架使用同一 `GZ_PARTITION`。自启动按系统 ID 选择角色：ID 1 为 `broadcast`，
ID 4 为 `rendezvous -t 1`，并启动 `dyt_guidance`。
分别用 `param show MAV_SYS_ID`、`listener vehicle_status -n 1` 和
`cooperative_rendezvous status` 检查身份一致。
跟随机应显示 `vehicle=4 role=rendezvous target=1`，目标机应显示
`vehicle=1 role=broadcast`。

QGC 选择系统 ID 4 的飞机发送摇杆输入。跟随机起飞后切换 Position 模式，
松开再按配置的按钮（`CRDZ_ACT_BTN=1` 对应 `buttons=0x0002`），
检查 `dyt_guidance_status.midcourse_switch_requested` 和
`cooperative_rendezvous_status.active`。有目标位置数据不等于已经激活跟随。
图像框选与下文相机跟随可以独立启动；协同中段跟随使用目标机广播的位置。

目前支持 `fpv_quadtailsitter_gimbal` 的三轴云台和相机 SDF。
重新编译并启动 `make px4_sitl gz_fpv_quadtailsitter_gimbal`，查询实际模型名和图像 topic。
将下面两个占位符替换为查询结果；两个进程必须处于同一 `GZ_PARTITION`。

```bash
/usr/bin/python3 Tools/simulation/gimbal_pose_inspector.py --list
/usr/bin/python3 Tools/simulation/gz_camera_reader.py --list-topics
```

PX4 控制台先停止原来的 mount 输入，避免 RC / MAVLink 与图像跟随同时控制云台。
新版 `--follow-gimbal` 会以带超时的控制权心跳让 Gazebo bridge 暂停 PX4 云台指令，
退出或心跳中断 0.5 仿真秒后恢复 PX4 输入。旧二进制仍需要先执行 `gimbal stop`。
启动 DYT 时禁用自动激活和辅助开关，仅观察目标输入：

```text
gimbal stop
param set DYTG_MODE 0
param set DYTG_AUTO_EN 0
param set DYTG_ACT_AUX -1
param set DYTG_ACT_BTN -1
param set DYTG_DLY_MS 0
param set DYTG_GMB_DLY 0
param set DYT_VIS_W 1280
param set DYT_VIS_H 720
dyt_guidance start
```

启动预览、鼠标框选目标并让相机跟随（无需解锁飞机）：

```bash
/usr/bin/python3 Tools/simulation/gz_camera_reader.py \
  --topic 实际图像topic \
  --dyt-model 本机模型名 \
  --world default \
  --manual-lock --follow-gimbal --rate 60
```

不同 world 请替换 `--world`。不加 `--follow-gimbal` 时仅发送目标数据。
无图形环境可使用 `--headless --roi x,y,width,height` 代替鼠标框选。
默认自动运动检测可能把相机运动产生的背景变化识别为目标，首次联调建议手动框选。

默认 `--rate 60` 是处理轮询上限，相机 SDF 仍输出 30 Hz 图像；重复时间戳不会重复发布。
DYT 模式拒绝 `--rate < 30`。每帧日志的 `dyt_timing` 记录真实仿真帧间隔、
图像时间与匹配位姿的偏差、图像仿真年龄。Gazebo 已携带曝光和采样时间戳，
4023 机型将 `DYTG_DLY_MS`、`DYTG_GMB_DLY` 的默认值设为 0，避免再减一次硬件延迟；
已保存非零值的现有实例请按上面的参数命令改为 0。脚本按 DYT 默认新鲜度要求检查：
帧间隔不超过 0.12 秒，图像年龄不超过 0.25 秒。超限会输出 `dyt_state` 错误并发送无效数据，
不会修改图像时间戳或放宽飞控参数来假装有效。机器负载或仿真加速仍可能造成超限，
应以日志实测为准。`--record-dir` 的 PNG 写盘也计入处理时间。

跟踪使用原始目标外观、前景及其边界掩码，并搜索尺度变化；外观变化时增加旋转搜索，
避免固定大小模板在飞机接近、转向时丢锁。锁定后停止无关的全图运动检测。
稳定跟踪时每侧搜索余量为目标尺寸的 35%（限制在 24–64 像素），
避免靠近后目标变大引起搜索计算量急剧增长；仅在当前角度的匹配分数不足时做旋转搜索。
搜索图像最长边进一步限制到 256 像素，模板按相同比例变换，尺度及旋转候选都在该尺寸内匹配。
匹配峰值使用二次曲线细化位置，再映射回原图坐标，降低缩小计算后的坐标跳动。
低置信度时增加 ±4% 的尺度以及 ±2°、±4°、±8° 的旋转候选；找到可靠匹配后提前结束回退搜索。
原始预览、框选、目标框尺寸及导引视线角仍使用原图坐标；曝光时间戳和有效性检查保持原有要求。
启动日志的 `matching_max_side_px` 可确认已加载该版本。
日志中的 `tracking_time_wall_s` 和 `processing_time_wall_s` 可用于检查实际处理负载。
`--rate 60` 只是轮询上限，无法保证耗时过长的跟踪计算达到 60 Hz。
短暂图像或位姿中断期间保留目标模板，但不发布有效测量；在丢失超时内图像恢复可继续原目标，
超时后仍需重新框选。完全遮挡、离开视场或无法区分的外观不能保证持续锁定。

云台控制使用与图像曝光时间最接近的历史位姿（偏差不超过 0.02 秒），
采用随仿真时间计算的比例速度、约两像素死区和每轴 0.5 rad/s 上限。
死区内保持已有关节指令，避免反复追逐像素量化误差；朝上时降低绕光轴的冗余 yaw 修正。
原始预览保持真实画面，不用显示滤波掩盖实际跟踪误差。

在 PX4 控制台检查：

```text
listener dyt_target -n 5
dyt_guidance status
```

数据路径为 `gz.msgs.Image → OpenCV 跟踪 → /model/本机模型名/dyt/target
→ gz_bridge → uORB dyt_target → dyt_guidance`。
这个试用的相机跟随由 Python 根据像素偏差控制 Gazebo 云台关节，
并非 `dyt_guidance` 的飞机轨迹跟随，也没有发送飞机激活或 Offboard 命令。
不要运行串口 `dyt_gimbal start`，以免两个数据源同时发布 `dyt_target`。

LOS 是理想针孔相机的右/下正角度，弧度单位；相机姿态是相对于模型的 FRD 姿态，
已包含朝上安装角。使用 `DYTG_LXSIGN/LYSIGN/RSIGN/PSIGN/YSIGN=1`、
`DYTG_ROFF/POFF/YOFF=0`、`DYTG_MNT_EN=0` 避免重复应用安装角或符号。
只读取本机及云台位姿作为姿态反馈，不读取目标模型真值来伪造图像测量或距离。
`range_m` 为 NaN，图像时间戳作为 `timestamp_sample`，延迟超过 0.5 仿真秒的包被拒绝。
图像和匹配位姿时间差超过 0.02 秒、帧间隔超限、图像冻结/中断、失配或丢失时停止发送新的跟随指令。
云台保持最后位置；失配数据标为无效，停止发送后由 DYT 原有新鲜度检查拒绝旧数据。
云台每轴修正限制为 0.5 rad/s，roll/pitch 同时受 SDF 机械限位约束。

退出用 Q / Esc / Ctrl+C。恢复原 mount 控制用 `gimbal start`。
本入口只用于 SITL 相机联调，尚未验证真实硬件。

## 检测和跟踪

- 检测采用 MOG2 背景差分和形态学去噪，过滤小区域及超过半幅画面的运动轮廓。
  `--min-target-area` 设置最小轮廓面积，默认 100 像素。
- 自动锁定最大的候选框，也可用鼠标或 `--roi` 锁定静止目标。
  锁定后使用原始 BGR 外观模板在上次位置附近搜索，目标停止运动仍可跟踪。
  不随每次匹配更新模板，以减少外观漂移。
- `--match-threshold` 设置接受匹配的最低分数，默认 0.8，范围 `(0,1]`。
  分数是 `1 - TM_SQDIFF_NORMED`，限制到 `[0,1]`，不是分类概率。
- 状态为 `detecting`（等待选择）、`locked`（刚锁定）、`tracking`（有效匹配）、
  `searching`（短时失配）和 `lost`（超时丢失）。短时失配时扩大局部搜索区域，
  在 `--lost-timeout` 内可恢复同一 ID，默认 2 **仿真秒**。
  超时后保留丢失状态，不自动换目标；需重新框选或按 R。
- 仿真时间回退或图像尺寸改变会清除跟踪和背景模型，手动模式需再次选择目标。
  短暂图像过期、时间冻结、解码或位姿异常期间保留模板，期间观测无效。
  `--roi` 只应用于初始选择，不在重置后自动重放。
- 这是运动区域检测，不具备 x500、人、车等语义分类能力。运动相机、云台转动、光照变化
  会产生误检；建议这些场景用手动框选。跟踪支持尺度变化和图像平面旋转；剧烈外观改变、
  快速移出搜索范围或长时间遮挡仍会导致丢失。外观相似的多个目标仍可能混淆身份。

每条 `frame` 日志的 `target` 包含：

| 字段 | 含义 |
| --- | --- |
| `state` / `valid` | 跟踪状态；只在锁定/有效匹配时有效 |
| `target_id` | 当前选择的 ID，每次新锁定递增，重置不复用 ID |
| `bbox_xywh` | 有效目标的左上角、宽高（像素）；失配时为 `null` |
| `center_px` | 有效目标中心（像素），失配时不提供 |
| `center_error_normalized` | 中心相对画面中心的偏差，右/下为正，画面边缘对应 ±1 |
| `match_score` | 匹配分数；刚锁定为 1，未匹配时为 `null` |
| `candidates_xywh` | 当前帧运动候选框，不代表已识别的物体类别 |

同时间戳的重复接收消息沿用该图像时间戳的跟踪结果，不重复推进检测器。
使用日志时同时检查 `stream_state`、`target.valid` 和图像时间戳；
中断后历史帧的目标位置不能作为实时观测。这些像素偏差不等于角度、距离或控制指令。

## 图像数据与限制

- 图像只来自订阅消息的 `data` 字段；按 `step` 处理每行末尾的 padding。
  支持 RGB8、BGR8、RGBA8、BGRA8 和灰度8；其它像素格式明确报错。
- PNG 保存的是转换后的 BGR8 像素，没有叠加文字、检测框或几何生成内容。
  RGBA/BGRA 的 alpha 被丢弃，灰度被展开成三通道；它不是原始 protobuf / 字节流备份。
- `stamp_ns` 是图像消息的仿真时间戳，保留整数纳秒；不使用墙钟来冒充图像时间。
  无时间戳或时间戳非法时，图像被判为无效。
- `receive_age_wall_s` 只表示本机收到消息后经过的时间，不是相机到程序的端到端延迟。
- 超过 `--max-age`（默认0.5墙钟秒）未收到画面，或图像时间戳停止推进，会清空显示并报告异常。
  黑色异常提示画面只用于窗口展示，绝不保存为相机帧。
- 队列只保留最新收到的消息；`--rate` 是处理/保存的上限，不是相机帧率。
  处理不及时会丢弃中间消息，`skipped_received_frames` 记录相邻处理之间略过的接收消息数。
  同一条消息不会反复保存。仿真时间回退时 `sim_epoch` 增加。
- 元数据日志包含 topic、来源格式、分辨率、行跨度、消息序号、仿真时间和保存文件名。

## 已完成验证

```bash
/usr/bin/python3 Tools/simulation/test_gz_camera_reader.py -v
```

离线测试包括通道顺序、行 padding、错误长度/格式、时间戳、过期、冻结、时间重置、
PNG 像素一致性、无重复保存、目录防覆盖及无窗口模式。
新增合成图像测试覆盖运动候选预热、自动锁定、移动/静止跟踪、遮挡恢复、
超时丢失不自动换目标、外观不匹配、手动锁定、非法 ROI、分辨率变化，
以及主循环中的重复时间戳、仿真重置和图像中断。
DYT 入口已通过 `make px4_sitl_default` 编译和独立分区的真实 Gazebo 图像联调。
本次稳定性验证使用独立、不解锁的 Gazebo 实例和真实相机画面：

- 静止 x500：20 秒 / 606 帧全部有效；居中后目标中心标准差为 0 像素，
  最大帧间隔 0.036 秒，最大图像年龄 0.048 秒。
- x500 移动、距离变化及转向：移除前统计区间 1016 帧全部通过跟踪及 DYT 时间检查；
  最大帧间隔 0.068 秒，图像年龄 0.092 秒，匹配位姿偏差 0.008 秒。
- 移除 x500 后统计的 170 帧全部无效，最终进入 `lost`，没有把天空误报为有效目标。
- 图像闭环期间向 PX4 云台注入 30 度 pitch 请求，目标仍保持居中，验证控制权互斥。
  单独心跳测试验证：租约有效时 PX4 关节指令被抑制；停止心跳后新 PX4 输入恢复发布。
- 40 项离线测试覆盖缩放、旋转、短暂中断保留身份、天空误锁拒绝、云台死区，
  以及朝上相机跨欧拉角奇异点时的姿态等价性。
- 处理负载回归（灰度外观亮度变化，15 帧）：128 像素目标的中位跟踪耗时
  从约 92 ms 降为 6 ms，256 像素目标从约 377 ms 降为 16 ms；所有测试帧保持有效。
  实际耗时仍应以运行日志中的耗时与时间校验结果为准。
- 近距离计算上限回归：在 1280×720 图像内移动 473×647 像素目标框，
  比较同一输入的更新前后耗时，并检查原图中心误差、目标尺寸和丢失后的无效状态。
  本机 12 帧移动测试中位耗时约 91 ms → 8 ms，最大中心误差 1 像素；
  接近同时转向测试 12/12 帧有效，最大耗时约 42 ms、最大中心误差约 3 像素。
  测试数值记录在 `/tmp/dyt_bounded_benchmark.json`；机器负载不同会改变耗时。

测试没有使用目标位姿来生成跟踪观测；目标位姿仅由测试程序改变以制造运动。
这些结果验证了相机脚本与 Gazebo 云台，不代表已验证所有飞行机动、遮挡和真实硬件。

DYT 专项离线测试（像素方向、时间戳、失效、图像/位姿时间差、云台奇异位形和限位）：

```bash
/usr/bin/python3 Tools/simulation/test_gz_dyt_bridge.py -v
```

图像字段依据本机 `gz.msgs.Image` 定义，显示/保存接口参照
[OpenCV 官方文档](https://docs.opencv.org/4.10.0/d5/d98/tutorial_mat_operations.html)。
