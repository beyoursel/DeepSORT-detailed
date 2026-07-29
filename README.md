# DeepSORT / ByteTrack Multi-Object Tracking (C++)

基于 C++ 的多目标跟踪项目，检测器支持 **YOLOv5 / YOLOv8 ONNX**，跟踪器支持 **DeepSORT / ByteTrack**，推理后端通过统一的 `IBackend` 抽象支持 **ONNXRuntime CPU / GPU** 切换。

---

## 1. 特性

- **配置驱动**：模型路径、输入源、推理后端、跟踪算法参数全部通过 `config/config.yaml` 配置，无需改代码即可切换。
- **多后端支持**：ONNXRuntime CPU / GPU 可切换，且支持“全局默认 + 模块独立覆盖”。
- **多检测网络**：已接入 YOLOv5 / YOLOv8，后续可按 `DetectorFactory` 扩展 YOLOv9 / YOLOv10 等。
- **多跟踪器**：DeepSORT / ByteTrack 一键切换。
- **可视化**：同时显示绿色检测框（类别 + 置信度）和红色跟踪框（跟踪 ID）。
- **耗时打印**：终端输出 `det_pre / det_infer / det_post / reid / track` 各阶段耗时，便于性能分析。

---

## 2. 项目结构

```
.
├── backend/               # 推理后端抽象
│   ├── include/IBackend.h
│   ├── include/BackendFactory.h
│   └── src/ONNXRuntimeBackend.cpp
├── config/                # 配置解析
│   ├── AppConfig.cpp
│   └── config.yaml
├── detector/              # 检测器
│   ├── include/
│   ├── src/
│   └── YOLOv5/            # YOLOv5 / YOLOv8 实现
├── tracker/               # 跟踪器
│   ├── include/           # ITracker 统一接口 + TrackerFactory
│   ├── deepsort/
│   └── bytetrack/
├── lib/                     # 第三方库
│   ├── onnxruntime-linux-x64-1.12.1/       # CPU 库
│   └── onnxruntime-linux-x64-gpu-1.16.3/   # GPU 库
├── models/                  # 模型文件
│   ├── yolov5s.onnx
│   ├── yolov8s.onnx
│   └── feature.onnx       # DeepSORT ReID
├── scripts/
│   └── extract_frame.py   # 视频抽帧脚本
├── main.cpp                 # 视频跟踪主程序
├── test_detector.cpp        # 单帧检测测试程序
└── CMakeLists.txt
```

---

## 3. 环境依赖

- Ubuntu 18.04 / 20.04 / 22.04
- GCC 9+
- CMake 3.5+
- OpenCV 4.x
- Eigen3
- yaml-cpp
- ONNXRuntime（已放在 `lib/` 下，无需编译）

安装 OpenCV 依赖和 yaml-cpp：

```bash
sudo apt-get update
sudo apt-get install -y libeigen3-dev libyaml-cpp-dev
```

> 如需 GPU 推理，需额外准备 CUDA 11.8 + cuDNN 8.6（或兼容 ONNXRuntime 1.16.3 的版本）。

### ONNXRuntime 下载

当前项目使用以下两个预编译版本（已放在 `lib/` 目录下），如需重新下载：

| 版本 | 适用场景 | 下载链接 |
|---|---|---|
| 1.12.1 CPU | 仅 CPU 推理 | [onnxruntime-linux-x64-1.12.1.tgz](https://github.com/microsoft/onnxruntime/releases/download/v1.12.1/onnxruntime-linux-x64-1.12.1.tgz) |
| 1.16.3 GPU | CUDA 11.8 GPU 推理 | [onnxruntime-linux-x64-gpu-1.16.3.tgz](https://github.com/microsoft/onnxruntime/releases/download/v1.16.3/onnxruntime-linux-x64-gpu-1.16.3.tgz) |

解压后放到项目根目录的 `lib/` 下：

```bash
# CPU 版本
wget https://github.com/microsoft/onnxruntime/releases/download/v1.12.1/onnxruntime-linux-x64-1.12.1.tgz
tar -xzf onnxruntime-linux-x64-1.12.1.tgz -C lib/

# GPU 版本（CUDA 11.8）
wget https://github.com/microsoft/onnxruntime/releases/download/v1.16.3/onnxruntime-linux-x64-gpu-1.16.3.tgz
tar -xzf onnxruntime-linux-x64-gpu-1.16.3.tgz -C lib/
```

> 也可以从官方 Release 页面查找其他版本：[https://github.com/microsoft/onnxruntime/releases](https://github.com/microsoft/onnxruntime/releases)

---

## 4. 编译

```bash
cd /media/taole/mydisk/DL_PROJECT/DeepSORT-detailed
mkdir -p build && cd build

# GPU 后端（默认）
cmake .. -DUSE_ONNXRUNTIME_GPU=ON
make -j$(nproc)

# CPU 后端（使用更小的 CPU 专用库）
# cmake .. -DUSE_ONNXRUNTIME_GPU=OFF
# make -j$(nproc)
```

编译产物：

- `./build/DeepSORT`：视频跟踪主程序
- `./build/test_detector`：单帧检测测试程序

---

## 5. 配置说明

所有参数集中在 `config/config.yaml`：

```yaml
# 全局推理后端
backend:
  type: "onnxruntime_gpu"   # onnxruntime_cpu | onnxruntime_gpu
  device_id: 0

# 输入输出
dataset:
  coco_labels: "./coco_80_labels_list.txt"
input:
  source: "./test_video.avi"   # 视频或图片路径
  type: "video"                # video | image
output:
  video: "out.avi"
  image: "out1.jpg"
  fourcc: "MJPG"
  fps: 10
  width: 1920
  height: 1080
  show: true        # 实时预览窗口；无显示环境设为 false
  result: ""        # MOT 格式结果文件，留空不输出

# 检测器配置
detector:
  type: "yolov8"                # yolov5 | yolov8
  backend: "onnxruntime_gpu"  # 不写则继承全局 backend.type
  model_path: "./models/yolov8s.onnx"
  input_width: 640
  input_height: 640
  confidence_threshold: 0.25
  nms_threshold: 0.4

# 跟踪器选择
tracker:
  type: "bytetrack"             # deepsort | bytetrack

# DeepSORT 配置
deepsort:
  feature_model_path: "./models/feature.onnx"
  backend: "onnxruntime_gpu"   # 不写则继承全局 backend.type
  feature_dim: 512
  max_cosine_distance: 0.2
  nn_budget: 100
  max_iou_distance: 0.7
  max_age: 30
  n_init: 3

# ByteTrack 配置
bytetrack:
  fps: 20
  track_buffer: 30
  track_thresh: 0.5
  high_thresh: 0.6
  match_thresh: 0.8
```

### 5.1 后端切换

- 全局 `backend.type` 是默认值。
- `detector.backend` / `deepsort.backend` 可单独覆盖，用于异构部署。例如：
  - 检测器跑 GPU，`backend.type = onnxruntime_gpu`。
  - ReID 跑 CPU，`deepsort.backend = onnxruntime_cpu`。
- 不写子模块的 `backend`，则默认继承全局设置。

### 5.2 跟踪器切换

直接改 `tracker.type`：

```yaml
tracker:
  type: "deepsort"   # 或 bytetrack
```

### 5.3 检测器切换

直接改 `detector.type`：

```yaml
detector:
  type: "yolov5"     # 或 yolov8
  model_path: "./models/yolov5s.onnx"
```

### 5.4 Headless 模式与结构化结果输出

在服务器 / 容器等无显示环境中运行时，关闭预览窗口并输出 MOT 格式结果：

```yaml
output:
  show: false              # 不弹窗、不调用 imshow/waitKey
  video: ""                # 可选：置空则不写视频文件
  result: "results.txt"    # MOTChallenge 格式：<frame>,<id>,<x>,<y>,<w>,<h>,<conf>,-1,-1,-1
```

`result` 文件每行对应一个已确认的跟踪框，可直接交给下游系统或用于跟踪指标评测。

---

## 6. 运行

### 6.1 视频跟踪

```bash
cd /media/taole/mydisk/DL_PROJECT/DeepSORT-detailed
./build/DeepSORT
# 或指定其他配置文件
./build/DeepSORT ./config/test_deepsort.yaml
```

运行后：

- 实时显示窗口：标题为 `Detector: yolov5` 或 `Detector: yolov8`，并同时显示绿色检测框和红色跟踪框。
- 保存输出视频：`output.video` 指定的文件（默认 `out.avi`）。
- 终端会打印每帧耗时，例如：

```text
[TIME] det_pre: 8.31 ms
[TIME] det_infer: 9.77 ms
[TIME] det_post: 2.26 ms
[TIME] track: 0.52 ms
[FRAME] frame=2 dets=5 det_total=20.39ms track_total=0.53ms
```

### 6.2 单帧检测测试

```bash
./build/test_detector ./config/test_detector.yaml
# 或 CPU 版本
./build/test_detector ./config/test_detector_cpu.yaml
```

终端会输出检测三阶段耗时：

```text
[TIME] det_pre: 11.33 ms
[TIME] det_infer: 9.89 ms
[TIME] det_post: 2.06 ms
```

---

## 7. 关于 GPU 首帧延迟

使用 `onnxruntime_gpu` 时，**第一帧推理通常需要 2~3 秒**，这是因为首次 `Run()` 会完成：

- CUDA context 创建
- cuDNN / cuBLAS handle 初始化
- GPU 显存池分配
- kernel 编译 / graph 构建
- 模型权重上传

这属于**正常预热**，不是性能问题。从第二帧开始会进入稳定状态：

| 后端 | 稳定单帧检测耗时（YOLOv8s @ 640x640） |
|---|---|
| ONNXRuntime CPU | ~300 ms |
| ONNXRuntime GPU | ~10 ms |

> 做性能 benchmark 时，建议丢弃前 1~2 帧，从第 3 帧开始统计。

---

## 8. 导出 YOLOv8 ONNX 模型

使用 Ultralytics 官方仓库：

```bash
pip install ultralytics
yolo export model=yolov8s.pt format=onnx opset=12 imgsz=640
```

导出后的 `yolov8s.onnx` 直接放入 `models/` 目录即可使用。

---

## 9. 抽帧脚本

从视频中抽取一帧保存为图片：

```bash
python3 scripts/extract_frame.py \
    --input ./test_video.avi \
    --output ./test_frame.jpg \
    --time 1.0
```

也支持按百分比抽取：

```bash
python3 scripts/extract_frame.py \
    --input ./test_video.avi \
    --output ./test_frame.jpg \
    --percent 0.5
```

---

## 10. 扩展新检测网络

检测器通过 `DetectorFactory` 创建，新增网络只需：

1. 在 `detector/include/` 下继承 `IDetector` 实现新检测器。
2. 在 `DetectorFactory::create()` 中注册新类型。
3. 在 `config.yaml` 中设置 `detector.type` 即可切换。

后端通过 `BackendFactory` 创建，新增推理框架（如 TensorRT、OpenVINO）只需实现 `IBackend` 接口并注册。

跟踪器通过 `TrackerFactory` 创建，新增跟踪算法（如 OC-SORT、BoT-SORT）只需：

1. 继承 `ITracker`（`tracker/include/ITracker.h`）实现 `update(frame, detections)`。
2. 在 `TrackerFactory::create()` 中注册新类型。
3. 在 `config.yaml` 中设置 `tracker.type` 即可切换。

---

## 11. 参考

- DeepSORT：[Simple Online and Realtime Tracking with a Deep Association Metric](https://arxiv.org/pdf/1703.07402.pdf)
- ByteTrack：[Multi-Object Tracking by Associating Every Detection Box](http://arxiv.org/abs/2110.06864)
- YOLOv5 / YOLOv8：[ultralytics/yolov5](https://github.com/ultralytics/yolov5) / [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)
- 原项目参考：[shaoshengsong/DeepSORT](https://github.com/shaoshengsong/DeepSORT)
