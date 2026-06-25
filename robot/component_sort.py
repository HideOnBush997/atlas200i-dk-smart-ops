#!/usr/bin/env python3
# coding: utf-8

import os
import sys
import json
from pathlib import Path
from time import sleep

import Arm_Lib
import cv2 as cv
import numpy as np
import rclpy
import torch
from ament_index_python.packages import get_package_share_directory
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from ais_bench.infer.interface import InferSession

from dofbot_info.srv import Kinemarics

from .utils.dofbot_config import ArmCalibration, read_XYT
from .utils.npu_utils import get_labels_from_txt, xyxy2xywh, draw_bbox
from .utils.det_utils import letterbox, scale_coords, nms


CFG = {
    "conf_thres": 0.4,
    "iou_thres": 0.45,
    "input_shape": [640, 640],
}

PLACE_POSITIONS = {
    "battery": [135, 43, 50, 41, 90, 30],
    "capacitor": [150, 58, 35, 30, 90, 30],
    "cement_resistor": [170, 65, 28, 23, 90, 30],
    "relay": [10, 65, 28, 23, 90, 30],
    "switch1": [30, 58, 35, 30, 90, 30],
    "switch2": [45, 43, 50, 41, 90, 30],
}

GRIP_OPEN = 55
GRIP_CLOSE = 135
READY = [90, 80, 50, 50, 265, GRIP_CLOSE]
WARMUP_BUFFER = 3
GRIP_ANGLE_FILE = "component_grip_angles.json"


def adaptive_grip_angles(width, height):
    min_side = min(float(width), float(height))
    open_angle = int(max(55, min(105, 35 + min_side * 0.45)))
    # No force feedback is available. Small parts need a tighter close angle;
    # large parts close less to avoid pushing them away.
    close_angle = int(max(125, min(155, 160 - min_side * 0.18)))
    return open_angle, close_angle


def load_component_grip_angles(cfg_dir):
    grip_file = Path(cfg_dir) / GRIP_ANGLE_FILE
    if not grip_file.exists():
        return {}
    try:
        data = json.loads(grip_file.read_text(encoding="utf-8"))
        return {str(name): int(angle) for name, angle in data.items()}
    except Exception as exc:
        print("Cannot read component grip angles:", grip_file, exc, flush=True)
        return {}


def infer_component_image(img_bgr, model, class_names, cfg):
    img, scale_ratio, pad_size = letterbox(img_bgr, new_shape=cfg["input_shape"])
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img, dtype=np.float32) / 255.0
    output = model.infer([img])[0]
    if isinstance(output, np.ndarray):
        output = torch.from_numpy(output)

    boxout = nms(output, conf_thres=cfg["conf_thres"], iou_thres=cfg["iou_thres"])
    pred_all = boxout[0].numpy()
    if len(pred_all) > 0:
        scale_coords(
            cfg["input_shape"],
            pred_all[:, :4],
            img_bgr.shape,
            ratio_pad=(scale_ratio, pad_size),
        )
    drawed_res = draw_bbox(pred_all, img_bgr, (0, 255, 0), 2, class_names)
    return pred_all, class_names, drawed_res


def open_camera(preferred=0):
    candidates = [preferred, 0, 1, 2, 3, 4]
    for video in sorted(Path("/dev").glob("video*")):
        try:
            candidates.append(int(video.name.replace("video", "")))
        except ValueError:
            pass

    tried = []
    for index in dict.fromkeys(candidates):
        capture = cv.VideoCapture(index, cv.CAP_V4L2)
        if not capture.isOpened():
            capture.release()
            tried.append(f"{index}:closed")
            continue

        capture.set(cv.CAP_PROP_BUFFERSIZE, 1)
        capture.set(cv.CAP_PROP_FRAME_WIDTH, 640)
        capture.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
        ok = False
        for _ in range(12):
            ret, frame = capture.read()
            if ret and frame is not None:
                ok = True
                break
            sleep(0.05)
        if ok:
            print("Open camera index:", index, flush=True)
            return capture
        capture.release()
        tried.append(f"{index}:no_frame")

    raise RuntimeError("Cannot open camera. Tried " + ", ".join(tried))
    raise RuntimeError("Cannot open camera")


def capture_latest_frame(capture):
    frame = None
    ret = False
    for _ in range(4):
        ok, current = capture.read()
        if ok and current is not None:
            ret = True
            frame = current
        sleep(0.03)
    print("component capture latest frame:", ret, flush=True)
    return ret, frame


def package_share_dirs():
    share_root = Path(get_package_share_directory("dofbot_garbage_yolov5"))
    return share_root / "config", share_root / "model"


class ComponentSorter:
    def __init__(self):
        rclpy.init(args=sys.argv)
        cfg_dir, model_dir = package_share_dirs()
        self.model_path = str(model_dir / "yolov5s_component.om")
        self.label_path = str(model_dir / "component_names.txt")
        if not os.path.exists(self.model_path):
            self.model_path = str(model_dir / "yolov5s_bs1.om")
        if not os.path.exists(self.label_path):
            self.label_path = str(model_dir / "coco_names.txt")

        self.model_path = os.path.realpath(self.model_path)
        self.label_path = os.path.realpath(self.label_path)
        self.model = InferSession(0, self.model_path)
        self.labels_dict = get_labels_from_txt(self.label_path)
        self.arm = Arm_Lib.Arm_Device()
        self.node = rclpy.create_node("dofbot_component_sort")
        self.client = self.node.create_client(Kinemarics, "trial_service")
        self.node_pub = rclpy.create_node("dofbot_component_img_node")
        self.image_pub = self.node_pub.create_publisher(Image, "cam_data", 10)
        self.bridge = CvBridge()
        self.frame = None
        self.offset = 0.0
        self.x_offset = 0.0
        self.grip_angles = load_component_grip_angles(cfg_dir)
        with open(cfg_dir / "offset.txt", "r") as f:
            self.offset = float(f.readline())
            self.x_offset = float(f.readline())
        print("component model:", self.model_path, flush=True)
        print("component labels:", self.label_path, flush=True)
        print("x_offset:", self.x_offset, "y_offset:", self.offset, flush=True)
        print("component grip angles:", self.grip_angles, flush=True)

    def detect(self, image):
        self.frame = cv.resize(image, (640, 480))
        pred, names, drawed_res = infer_component_image(self.frame, self.model, self.labels_dict, CFG)
        self.frame = drawed_res
        self.image_pub.publish(self.bridge.cv2_to_imgmsg(drawed_res, encoding="bgr8"))

        components = []
        gn = torch.tensor([640, 480, 640, 480])
        for *xyxy, conf, cls in reversed(pred):
            name = names[int(cls)]
            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
            point_x = int(xywh[0] * 640)
            point_y = int(xywh[1] * 480)
            width = abs(float(xyxy[2]) - float(xyxy[0]))
            height = abs(float(xyxy[3]) - float(xyxy[1]))
            posxy = (
                round((point_x - 320) / 4000, 5),
                round(((480 - point_y) / 3000) * 0.8 + 0.19, 5),
            )
            cv.circle(self.frame, (point_x, point_y), 5, (0, 0, 255), -1)
            cv.putText(self.frame, f"{name} {float(conf):.2f}", (point_x, max(20, point_y - 8)),
                       cv.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
            components.append((name, posxy, point_x, point_y, width, height, float(conf)))
        components.sort(key=lambda item: item[3], reverse=True)
        print("components:", components, flush=True)
        return self.frame, components

    def joints_for(self, posxy):
        print("posxy is:", posxy, flush=True)
        self.client.wait_for_service()
        request = Kinemarics.Request()
        request.tar_x = posxy[0] + self.x_offset
        request.tar_y = posxy[1] + self.offset
        request.kin_name = "ik"
        future = self.client.call_async(request)
        rclpy.spin_until_future_complete(self.node, future)
        response = future.result()
        if not response:
            return None
        joints = [response.joint1, response.joint2, response.joint3, response.joint4, response.joint5]
        if joints[2] < 0:
            joints[1] += joints[2] / 2
            joints[3] += joints[2] * 3 / 4
            joints[2] = 0
        print("ik joints:", joints, flush=True)
        return joints

    def move_component(self, component):
        name, posxy, _, _, width, height, _ = component
        if name not in PLACE_POSITIONS:
            print("Unknown component class, skip:", name, flush=True)
            return False
        joints = self.joints_for(posxy)
        if not joints:
            print("IK failed, skip:", name, flush=True)
            return False

        grip_open, grip_close = adaptive_grip_angles(width, height)
        if name in self.grip_angles:
            grip_close = self.grip_angles[name]
            print("configured gripper close:", name, grip_close, "open:", grip_open, flush=True)
        else:
            print("adaptive gripper open/close:", grip_open, grip_close, "size:", width, height, flush=True)
        pickup = [joints[0], joints[1], joints[2], joints[3], 265, grip_open]
        place = PLACE_POSITIONS[name].copy()
        place[5] = grip_close
        lift = [place[0], 80, 50, 50, 265, grip_open]
        carry_ready = READY.copy()
        carry_ready[5] = grip_close

        print("=== move component", name, "===", flush=True)
        self.arm.Arm_serial_servo_write6_array(carry_ready, 1000)
        sleep(1)
        self.arm.Arm_serial_servo_write(6, grip_open, 500)
        sleep(0.5)
        self.arm.Arm_serial_servo_write6_array([int(v) for v in pickup], 1000)
        sleep(1)
        self.arm.Arm_serial_servo_write(6, grip_close, 700)
        sleep(0.7)
        self.arm.Arm_serial_servo_write(6, grip_close, 500)
        sleep(0.8)
        self.arm.Arm_serial_servo_write6_array(carry_ready, 1000)
        sleep(1)
        self.arm.Arm_serial_servo_write6_array([int(v) for v in place], 1000)
        sleep(1)
        self.arm.Arm_serial_servo_write(6, grip_open, 700)
        sleep(0.7)
        self.arm.Arm_serial_servo_write6_array([int(v) for v in lift], 1000)
        sleep(1)
        return True


def confirm_components(frame, components):
    cv.imwrite("/tmp/component_sort_detect.jpg", frame)
    if os.environ.get("DISPLAY"):
        cv.imshow("component_sort_detect", frame)
        cv.waitKey(1)
    print("detect image: /tmp/component_sort_detect.jpg", flush=True)
    if not components:
        return False
    for idx, component in enumerate(components, 1):
        name, posxy, px, py, width, height, conf = component
        print(f"{idx}. {name} conf={conf:.2f} pixel=({px},{py}) posxy={posxy} size={width:.0f}x{height:.0f}", flush=True)
    print("Auto confirm pickup.", flush=True)
    return True


def main(args=None):
    sorter = ComponentSorter()
    calibration = ArmCalibration()
    cfg_dir, _ = package_share_dirs()
    xy, _ = read_XYT(str(cfg_dir / "XYT_config.txt"))
    dp = np.fromfile(str(cfg_dir / "dp.bin"), dtype=np.int32).reshape(4, 2)

    home = [xy[0], xy[1], 0, 0, 90, 30]
    view = [xy[0], xy[1], 0, 0, 90, 30]
    sorter.arm.Arm_serial_servo_write6_array(view, 1000)
    sleep(1)

    capture = open_camera(0)
    warmup = 0
    camera_fail_count = 0
    try:
        while True:
            ret, img = capture_latest_frame(capture)
            if not ret:
                camera_fail_count += 1
                if camera_fail_count >= 8:
                    print("component camera read failed repeatedly, reopen camera", flush=True)
                    capture.release()
                    sleep(0.5)
                    capture = open_camera(0)
                    camera_fail_count = 0
                continue
            camera_fail_count = 0
            img = cv.resize(img, (640, 480))
            img = calibration.perspective_transform(dp, img)
            frame, components = sorter.detect(img)
            cv.imwrite("/tmp/component_sort_debug_frame.jpg", frame)
            if os.environ.get("DISPLAY"):
                cv.imshow("component_sort_detect", frame)
                key = cv.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
            print("warmup:", warmup, flush=True)
            if components:
                warmup += 1
            else:
                warmup = 0
            if warmup > WARMUP_BUFFER:
                if confirm_components(frame, components):
                    for component in components:
                        sorter.move_component(component)
                    sorter.arm.Arm_serial_servo_write6_array(home, 1000)
                    sleep(1)
                warmup = 0
    finally:
        capture.release()
        cv.destroyAllWindows()


if __name__ == "__main__":
    main()
