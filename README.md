# yolo_ort_ros
yolo_ort_ros is a lightweight ROS 2 (Humble) node for running YOLOv11 ONNX models on NVIDIA Jetson using ONNX Runtime with CUDA acceleration. The node subscribes to sensor_msgs/Image, performs GPU inference, and publishes detections as vision_msgs/Detection2DArray plus an optional centroid list in a backward-compatible std_msgs/String format
