# yolo_ort_ros — YOLOv11 ONNXRuntime (CUDA) ROS 2 Node for Jetson

`yolo_ort_ros` provides a ROS 2 (Humble) node that runs a YOLOv11 ONNX model using **ONNX Runtime** with **CUDAExecutionProvider** on NVIDIA Jetson devices (e.g., Orin Nano).  
It is designed as a simple, low-latency “latest-frame” inference node for camera topics.

✅ Subscribes to: `sensor_msgs/msg/Image`  
✅ Publishes detections: `vision_msgs/msg/Detection2DArray`  
✅ Publishes centroids (optional compatibility output): `std_msgs/msg/String`  

---

## Features

- **GPU-accelerated inference** using ONNX Runtime CUDA EP on Jetson.
- **Latest-frame processing** (drops older frames automatically to reduce latency).
- **Letterbox preprocessing** (resize + padding to model input size).
- **Two output formats**:
  - `vision_msgs/Detection2DArray` for ROS-native downstream use.
  - `std_msgs/String` containing `[(cx, cy), ...]` (same style as many quick Python prototypes).

---

## Requirements

### System / Platform
- Ubuntu 22.04 (JetPack 6.x recommended on Jetson)
- ROS 2 Humble installed via apt
- OpenCV installed (system OpenCV is fine)
- `cv_bridge` and `vision_msgs`

### Python
- Python 3.10
- ONNX Runtime **GPU** build compatible with Jetson / JetPack (see installation below)

---

## Model export (Ultralytics)

Export an end-to-end ONNX model (with NMS included) using Ultralytics:

```bash
yolo export model=best_model.pt format=onnx imgsz=640 opset=17 simplify=True nms=True batch=1
