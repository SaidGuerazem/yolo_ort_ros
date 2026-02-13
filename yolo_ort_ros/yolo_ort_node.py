#!/usr/bin/env python3
import threading
import queue
import time
from dataclasses import dataclass

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import Image
from std_msgs.msg import String
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose, BoundingBox2D
from cv_bridge import CvBridge

import onnxruntime as ort


@dataclass
class LetterboxMeta:
    scale: float
    pad_x: int
    pad_y: int
    src_w: int
    src_h: int


def letterbox(bgr: np.ndarray, S: int) -> tuple[np.ndarray, LetterboxMeta]:
    h, w = bgr.shape[:2]
    r = min(S / w, S / h)
    new_w, new_h = int(round(w * r)), int(round(h * r))
    resized = cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    dw, dh = S - new_w, S - new_h
    left, top = dw // 2, dh // 2

    out = cv2.copyMakeBorder(
        resized, top, dh - top, left, dw - left,
        borderType=cv2.BORDER_CONSTANT, value=(114, 114, 114)
    )
    meta = LetterboxMeta(scale=r, pad_x=left, pad_y=top, src_w=w, src_h=h)
    return out, meta


class YoloOrtNode(Node):
    def __init__(self):
        super().__init__("yolo_ort_node")

        self.declare_parameter("image_topic", "/flir_camera/image_raw")
        self.declare_parameter("centroids_topic", "/flir_camera/centroids")
        self.declare_parameter("detections_topic", "/flir_camera/detections")
        self.declare_parameter("model_path", "")
        self.declare_parameter("imgsz", 640)
        self.declare_parameter("conf_thres", 0.6)
        self.declare_parameter("normalize_to_engine", True)  # match your old behavior
        self.declare_parameter("use_gpu", True)
        self.declare_parameter("queue_size", 1)  # latest-only

        self.image_topic = self.get_parameter("image_topic").value
        self.centroids_topic = self.get_parameter("centroids_topic").value
        self.detections_topic = self.get_parameter("detections_topic").value
        self.model_path = self.get_parameter("model_path").value
        self.S = int(self.get_parameter("imgsz").value)
        self.conf = float(self.get_parameter("conf_thres").value)
        self.normalize_to_engine = bool(self.get_parameter("normalize_to_engine").value)
        self.use_gpu = bool(self.get_parameter("use_gpu").value)
        self.queue_size = int(self.get_parameter("queue_size").value)

        if not self.model_path:
            raise RuntimeError("model_path param is empty (set it to your .onnx)")

        self.bridge = CvBridge()

        # ---- ONNX Runtime session (CUDA if available) ----
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.intra_op_num_threads = 1
        so.inter_op_num_threads = 1

        providers = ort.get_available_providers()
        self.get_logger().info(f"ORT providers available: {providers}")

        sess_providers = ["CPUExecutionProvider"]
        if self.use_gpu and ("CUDAExecutionProvider" in providers):
            sess_providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        self.session = ort.InferenceSession(self.model_path, sess_options=so, providers=sess_providers)
        self.input_name = self.session.get_inputs()[0].name

        self.get_logger().info(f"Using providers: {self.session.get_providers()}")
        self.get_logger().info(f"Model input: {self.input_name}")

        # pubs/subs
        self.pub_centroids = self.create_publisher(String, self.centroids_topic, 10)
        self.pub_dets = self.create_publisher(Detection2DArray, self.detections_topic, 10)
        self.sub = self.create_subscription(Image, self.image_topic, self.image_cb, qos_profile_sensor_data)

        # latest-only queue + worker
        self.q = queue.Queue(maxsize=max(1, self.queue_size))
        self.stop = False
        self.worker = threading.Thread(target=self.worker_loop, daemon=True)
        self.worker.start()

        self.get_logger().info(f"Subscribed: {self.image_topic}")
        self.get_logger().info(f"Publishing centroids: {self.centroids_topic}")
        self.get_logger().info(f"Publishing detections: {self.detections_topic}")

    def destroy_node(self):
        self.stop = True
        try:
            self.q.put_nowait(None)
        except Exception:
            pass
        super().destroy_node()

    def image_cb(self, msg: Image):
        try:
            bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            if bgr is None or bgr.size == 0:
                return
            # latest-only behavior
            try:
                self.q.put_nowait((bgr, msg.header.stamp))
            except queue.Full:
                try:
                    _ = self.q.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self.q.put_nowait((bgr, msg.header.stamp))
                except queue.Full:
                    pass
        except Exception as e:
            self.get_logger().error(f"cv_bridge error: {e}")

    def worker_loop(self):
        while rclpy.ok() and not self.stop:
            item = None
            try:
                item = self.q.get(timeout=0.2)
            except queue.Empty:
                continue
            if item is None:
                continue

            bgr, stamp = item

            try:
                lb, meta = letterbox(bgr, self.S)

                # Fast blob creation in C++ (OpenCV): NCHW float32, RGB, /255
                blob = cv2.dnn.blobFromImage(
                    lb, scalefactor=1.0/255.0, size=(self.S, self.S),
                    mean=(0, 0, 0), swapRB=True, crop=False
                )

                # Run ONNX
                out = self.session.run(None, {self.input_name: blob})

                # Expect end2end output: (N,6) or (1,N,6)
                det = out[0]
                det = np.asarray(det)
                if det.ndim == 3:
                    det = det[0]
                if det.size == 0:
                    self.publish_none(stamp)
                    continue

                # det rows: x1,y1,x2,y2,score,cls
                centroids = []
                det_msg = Detection2DArray()
                det_msg.header.stamp = stamp

                for row in det:
                    x1, y1, x2, y2, score, cls = row[:6]
                    score = float(score)
                    if score < self.conf:
                        continue

                    # Map from letterbox coords -> original
                    x1 = (float(x1) - meta.pad_x) / meta.scale
                    y1 = (float(y1) - meta.pad_y) / meta.scale
                    x2 = (float(x2) - meta.pad_x) / meta.scale
                    y2 = (float(y2) - meta.pad_y) / meta.scale

                    x1 = max(0.0, min(x1, meta.src_w - 1.0))
                    y1 = max(0.0, min(y1, meta.src_h - 1.0))
                    x2 = max(0.0, min(x2, meta.src_w - 1.0))
                    y2 = max(0.0, min(y2, meta.src_h - 1.0))

                    cx = 0.5 * (x1 + x2)
                    cy = 0.5 * (y1 + y2)

                    if self.normalize_to_engine:
                        ex = cx * meta.scale + meta.pad_x
                        ey = cy * meta.scale + meta.pad_y
                        centroids.append((ex / self.S, ey / self.S))
                    else:
                        centroids.append((cx / meta.src_w, cy / meta.src_h))

                    # vision_msgs detection
                    d2 = Detection2D()
                    bb = BoundingBox2D()
                    bb.center.position.x = float(cx)
                    bb.center.position.y = float(cy)
                    bb.size_x = float(max(0.0, x2 - x1))
                    bb.size_y = float(max(0.0, y2 - y1))
                    d2.bbox = bb

                    hyp = ObjectHypothesisWithPose()
                    hyp.hypothesis.class_id = str(int(cls))
                    hyp.hypothesis.score = float(score)
                    d2.results.append(hyp)

                    det_msg.detections.append(d2)

                if len(centroids) == 0:
                    self.publish_none(stamp)
                else:
                    s = String()
                    s.data = str([(float(a), float(b)) for a, b in centroids])
                    self.pub_centroids.publish(s)
                    self.pub_dets.publish(det_msg)

            except Exception as e:
                self.get_logger().error(f"Inference error: {e}")

    def publish_none(self, stamp):
        s = String()
        s.data = "None"
        self.pub_centroids.publish(s)

        d = Detection2DArray()
        d.header.stamp = stamp
        self.pub_dets.publish(d)


def main():
    rclpy.init()
    node = YoloOrtNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

