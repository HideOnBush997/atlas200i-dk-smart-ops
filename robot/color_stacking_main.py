#!/usr/bin/env python3
# coding: utf-8

import os
from time import sleep

import Arm_Lib
import cv2 as cv
from pathlib import Path
import numpy as np
from ament_index_python.packages import get_package_share_directory


def open_camera(preferred=0):
    for index in [preferred, 1, 2, 3]:
        capture = cv.VideoCapture(index)
        if capture.isOpened():
            capture.set(cv.CAP_PROP_BUFFERSIZE, 1)
            capture.set(cv.CAP_PROP_FRAME_WIDTH, 640)
            capture.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
            print("Open camera index:", index)
            return capture
        capture.release()
    raise RuntimeError("Cannot open USB camera. Check /dev/video* or close other camera programs.")

from .utils.dofbot_config import ArmCalibration, read_XYT
from .utils.stacking_target import stacking_GetTarget


CLAMP_OPEN = 60
CLAMP_CLOSE = 135
# Fixed poses copied from the calibrated color stacking workflow.
# Pose format: [S1, S2, S3, S4, S5]. Gripper is appended when moving.
P_HOME = [88, 123, 0, 0, 90]
P_TOP = [90, 80, 50, 50, 270]
P_PICKUP = [92, 48, 35, 30, 270]

DROP_POSES = {
    "yellow": [64, 18, 64, 56, 270],
    "red": [119, 15, 66, 56, 270],
    "green": [139, 61, 16, 29, 270],
    "blue": [44, 64, 15, 28, 270],
}


def move_pose(arm, pose, move_time=1000, gripper=None):
    target = list(pose)
    if gripper is None:
        gripper = CLAMP_CLOSE
    target.append(gripper)
    print("move:", target, flush=True)
    arm.Arm_serial_servo_write6_array([int(v) for v in target], move_time)
    sleep(move_time / 1000.0 + 0.15)


def clamp(arm, close):
    angle = CLAMP_CLOSE if close else CLAMP_OPEN
    print("clamp:", angle, flush=True)
    arm.Arm_serial_servo_write(6, angle, 500)
    sleep(0.6)


def lift_up(arm):
    arm.Arm_serial_servo_write(2, 90, 1500)
    arm.Arm_serial_servo_write(3, 90, 1500)
    arm.Arm_serial_servo_write(4, 90, 1500)
    sleep(1.6)


def select_pick_color(msg):
    if not msg:
        return None
    # The pickup point is fixed near the image center; if more than one color is
    # visible, use the detected block closest to that fixed pickup area.
    known = [(name, pos) for name, pos in msg.items() if name in DROP_POSES]
    if not known:
        return None
    known.sort(key=lambda item: abs(item[1][0]))
    return known[0][0]


def fixed_color_sort_once(arm, color_name):
    drop_pose = DROP_POSES.get(color_name)
    if drop_pose is None:
        print("unknown color, skip:", color_name, flush=True)
        return
    print("=== fixed color sort:", color_name, "===", flush=True)
    move_pose(arm, P_TOP, 1000, CLAMP_OPEN)
    move_pose(arm, P_PICKUP, 1000, CLAMP_OPEN)
    clamp(arm, True)
    move_pose(arm, P_TOP, 1000, CLAMP_CLOSE)
    move_pose(arm, drop_pose, 1000, CLAMP_CLOSE)
    clamp(arm, False)
    sleep(0.2)
    lift_up(arm)
    move_pose(arm, P_HOME, 1000, CLAMP_OPEN)


def capture_and_detect(capture, target):
    frame = None
    ret = False
    # Some USB cameras on this board do not behave well after repeated grab().
    # Read several frames and use the newest successful one to avoid stale buffer frames.
    for _ in range(4):
        ok, current = capture.read()
        if ok:
            ret = True
            frame = current
        sleep(0.03)
    print("capture latest frame:", ret, flush=True)
    if not ret:
        return None, {}
    frame = cv.resize(frame, (640, 480))
    return target.select_color(frame)


def main(args=None):
    # 创建获取目标实例
    target = stacking_GetTarget()
    # 创建相机标定实例
    calibration = ArmCalibration()
    # 初始化一些参数
    dp = []
    xy = [90, 135]
    msg = {}
    WARMUP_BUFFER = 1

    # 后续作为ROS参数
    DP_PRINT = False

    share_root = get_package_share_directory("robot_arm_color_stacking")
    cfg_folder = os.path.join(share_root, "config")
    dp_cfg_path = os.path.join(cfg_folder, "dp.bin")

    # XYT参数路径
    # revise
    XYT_path = os.path.join(cfg_folder, "XYT_config.txt")

    try:
        xy, _ = read_XYT(XYT_path)
    except Exception:
        print("No XYT_config file!!!")

    print("Read xy is", xy)

    warm_up_count = 0
    last_num = 0
    last_count = 0

    # 创建机械臂驱动实例
    arm = Arm_Lib.Arm_Device()
    joints_0 = P_HOME + [CLAMP_OPEN]
    joints_1 = [P_HOME[0], P_HOME[1], 50, 50, 90, CLAMP_OPEN]

    # 重置机械臂位置
    print("Start Reset Robot Arm Position, Please Wait..")
    arm.Arm_serial_servo_write6_array(joints_1, 1000)
    sleep(2)
    arm.Arm_serial_servo_write6_array(joints_0, 1000)
    sleep(2)
    print("Finish Robot Arm Position Reset!")

    # 打开摄像头
    capture = open_camera(0)
    camera_fail_count = 0
    # 当摄像头正常打开的情况下循环执行
    while True:
        _, msg = capture_and_detect(capture, target)
        if msg == {}:
            msg = {}
            print("read latest camera frame failed or no color", flush=True)
            camera_fail_count += 1
            if camera_fail_count >= 8:
                print("camera read failed repeatedly, reopen camera", flush=True)
                capture.release()
                sleep(0.5)
                capture = open_camera(0)
                camera_fail_count = 0
            sleep(0.1)
            continue
        camera_fail_count = 0

        dp = np.fromfile(dp_cfg_path, dtype=np.int32)
        if DP_PRINT:
            print("dp dtype:", dp.dtype)
            print(dp.shape)
            print(dp)
        dp = dp.reshape(4, 2)

        # Color stacking uses the full camera view so blocks around the mat stay visible.
        # img = calibration.perspective_transform(dp, img)

        print("Model is warming up at stage:", warm_up_count)
        if warm_up_count != 0 and last_num == warm_up_count:
            last_count += 1
            if last_count > 5:
                warm_up_count = 0
                last_count = 0
        last_num = warm_up_count

        if len(msg) != 0:
            warm_up_count += 1
            if warm_up_count > WARMUP_BUFFER:
                sleep(0.2)
                img, msg = capture_and_detect(capture, target)
                color_name = select_pick_color(msg)
                print("fixed pickup color:", color_name, flush=True)
                if color_name is not None:
                    arm.Arm_Buzzer_On(1)
                    sleep(0.3)
                    fixed_color_sort_once(arm, color_name)
                warm_up_count = 0
                last_num = 0
                last_count = 0

    cv.destroyAllWindows()
    capture.release()


if __name__ == "__main__":
    main()
