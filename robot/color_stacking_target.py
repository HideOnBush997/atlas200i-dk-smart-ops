#!/usr/bin/env python3
# coding: utf-8
import sys
import os
from time import sleep

import torch
import rclpy
import Arm_Lib
import cv2 as cv
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from pathlib import Path
from ament_index_python.packages import get_package_share_directory

from dofbot_info.srv import Kinemarics
from .stacking_grap import stacking_grap


rclpy.init(args=sys.argv)


class stacking_GetTarget:
    def __init__(self, test_mode=False):
        if test_mode:
            FILE = Path(__file__).resolve()
            lib_root = os.path.dirname(FILE.parents[0])
            cfg_folder = os.path.join(lib_root, "config")
        else:
            share_root = get_package_share_directory("robot_arm_color_stacking")
            cfg_folder = os.path.join(share_root, "config")

        self.offset_cfg_path = os.path.join(cfg_folder, "offset.txt")
        self.color_cfg_path = os.path.join(cfg_folder, "color_stacking.txt")
        self.test_mode = test_mode

        self.image = None
        self.color_status = True
        self.xy = [90, 135]
        self.arm = Arm_Lib.Arm_Device()
        self.grap = stacking_grap()
        self.node = rclpy.create_node("dofbot_stacking")
        self.node_pub = rclpy.create_node("dofbot_img_node")
        self.client = self.node.create_client(Kinemarics, "trial_service")
        self.image_pub = self.node_pub.create_publisher(Image, "cam_data", 10)
        self.bridge = CvBridge()

        self.offset = -1
        self.x_offset = -1
        with open(self.offset_cfg_path, "r") as f:
            self.offset = float(f.readline())
            self.x_offset = float(f.readline())
            print("y_offset is", self.offset)
            print("x_offset is", self.x_offset)

        self.color_ranges = self.read_color_ranges(self.color_cfg_path)
        print("color ranges:", self.color_ranges)
        print("finish init..")

    def read_color_ranges(self, path):
        ranges = {}
        with open(path, "r") as f:
            for line in f:
                line = line.strip().strip(",")
                if not line or ":" not in line:
                    continue
                name, value = line.split(":", 1)
                name = name.strip().strip('"').lower()
                nums = [int(x.strip()) for x in value.strip().strip("[]").split(",")]
                if len(nums) == 6:
                    ranges[name] = (np.array(nums[:3], dtype=np.uint8), np.array(nums[3:], dtype=np.uint8))
        return ranges

    def target_run(self, msg, xy=None):
        if xy is not None:
            self.xy = xy
        if any(v is not None for v in msg.values()):
            self.arm.Arm_Buzzer_On(1)
            sleep(0.5)

        msg_list = sorted(list(msg.items()), key=lambda x: x[1][1])
        num = 1
        for name, pos in msg_list:
            print("stack pos:", pos, flush=True)
            print("stack color:", name, "stack index:", num, flush=True)
            try:
                joints = self.server_joint(pos)
                print("stack joints:", joints, flush=True)
                if not joints:
                    print("skip stack target because joints empty", flush=True)
                    continue
                self.grap.arm_run(str(num), joints)
                num += 1
            except Exception as e:
                print("stack target failed:", e, flush=True)

        self.arm.Arm_serial_servo_write(1, 90, 1000)
        sleep(1)
        joints_0 = [self.xy[0], self.xy[1], 0, 0, 90, 30]
        self.arm.Arm_serial_servo_write6_array(joints_0, 1000)
        sleep(1)

    def select_color(self, image, garbage_index=0):
        self.image = cv.resize(image, (640, 480))
        msg = self.get_pos_by_color()
        return self.image, msg

    def get_pos_by_color(self):
        img = self.image.copy()
        hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        kernel = np.ones((5, 5), np.uint8)
        msg = {}
        roi_x1, roi_y1, roi_x2, roi_y2 = 170, 120, 470, 380
        cv.rectangle(self.image, (roi_x1, roi_y1), (roi_x2, roi_y2), (255, 255, 255), 2)

        for color_name, (lower, upper) in self.color_ranges.items():
            mask = cv.inRange(hsv, lower, upper)
            # Red may wrap around hue=180; include high-red range as well.
            if color_name == "red":
                lower2 = np.array([170, max(80, int(lower[1])), max(80, int(lower[2]))], dtype=np.uint8)
                upper2 = np.array([179, int(upper[1]), int(upper[2])], dtype=np.uint8)
                mask = cv.bitwise_or(mask, cv.inRange(hsv, lower2, upper2))
            mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel, iterations=1)
            mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel, iterations=2)
            contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            candidates = []
            for contour in contours:
                area = cv.contourArea(contour)
                if area < 900:
                    continue
                x, y, w, h = cv.boundingRect(contour)
                if w < 20 or h < 20:
                    continue
                center_x = int(x + w / 2)
                center_y = int(y + h / 2)
                if not (roi_x1 <= center_x <= roi_x2 and roi_y1 <= center_y <= roi_y2):
                    continue
                aspect = w / float(h)
                # Ignore long background patches on the mat; target blocks are roughly square/box-like.
                if aspect < 0.45 or aspect > 2.2:
                    continue
                candidates.append((area, x, y, w, h))
            if not candidates:
                continue
            area, x, y, w, h = max(candidates, key=lambda item: item[0])
            point_x = int(x + w / 2)
            point_y = int(y + h / 2)
            cv.rectangle(self.image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv.circle(self.image, (point_x, point_y), 5, (0, 0, 255), -1)
            cv.putText(self.image, f"{color_name} {int(area)}", (x, max(20, y - 8)), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            a = round(((point_x - 320) / 4000), 5)
            b = round(((480 - point_y) / 3000) * 0.8 + 0.19, 5)
            msg[color_name] = (a, b)

        if msg:
            cv.putText(self.image, "Color Result, Waiting for Robot Arm Finish..", (25, 45), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        data = self.bridge.cv2_to_imgmsg(self.image, encoding="bgr8")
        self.image_pub.publish(data)
        cv.imwrite('/tmp/color_debug_frame.jpg', self.image)
        cv.imwrite('/tmp/color_sort_debug_frame.jpg', self.image)
        print("color msg is:", msg)
        return msg

    def server_joint(self, posxy):
        self.client.wait_for_service()
        request = Kinemarics.Request()
        request.tar_x = posxy[0] + self.x_offset
        request.tar_y = posxy[1] + self.offset
        request.kin_name = "ik"
        try:
            self.future = self.client.call_async(request)
            rclpy.spin_until_future_complete(self.node, self.future)
            response = self.future.result()
            if response:
                joints = [0.0, 0.0, 0.0, 0.0, 0.0]
                joints[0] = response.joint1
                joints[1] = response.joint2
                joints[2] = response.joint3
                joints[3] = response.joint4
                joints[4] = response.joint5
                if joints[2] < 0:
                    joints[1] += joints[2] * 3 / 5
                    joints[3] += joints[2] * 3 / 5
                    joints[2] = 0
                return joints
        except Exception:
            print("arg error")
