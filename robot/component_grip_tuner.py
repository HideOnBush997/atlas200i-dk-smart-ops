#!/usr/bin/env python3
# coding: utf-8

import json
import os
from pathlib import Path
from time import sleep

import cv2 as cv
import numpy as np

from .component_sort import (
    ComponentSorter,
    ArmCalibration,
    READY,
    adaptive_grip_angles,
    open_camera,
    package_share_dirs,
    read_XYT,
)


GRIP_ANGLE_FILE = "component_grip_angles.json"
SOURCE_CONFIG_DIR = Path(
    "/home/HwHiAiUser/E2ESamples/ros2_robot_arm/ros2_ws/src/"
    "dofbot_garbage_yolov5/dofbot_garbage_yolov5/config"
)


def grip_file_paths(cfg_dir):
    return [Path(cfg_dir) / GRIP_ANGLE_FILE, SOURCE_CONFIG_DIR / GRIP_ANGLE_FILE]


def load_angles(cfg_dir):
    for path in grip_file_paths(cfg_dir):
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                return {str(name): int(angle) for name, angle in data.items()}
            except Exception as exc:
                print("read grip angle file failed:", path, exc, flush=True)
    return {}


def save_angles(cfg_dir, angles):
    text = json.dumps(angles, indent=2, sort_keys=True)
    for path in grip_file_paths(cfg_dir):
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text + "\n", encoding="utf-8")
            print("saved grip angles:", path, flush=True)
        except Exception as exc:
            print("save grip angles failed:", path, exc, flush=True)


def draw_overlay(frame, component, angles):
    if not component:
        cv.putText(frame, "no component detected", (10, 28), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return frame

    name, posxy, px, py, width, height, conf = component
    saved = angles.get(name, "none")
    cv.circle(frame, (int(px), int(py)), 10, (255, 0, 255), 3)
    cv.putText(frame, f"{name} conf={conf:.2f} saved_angle={saved}", (10, 28),
               cv.FONT_HERSHEY_SIMPLEX, 0.68, (0, 0, 255), 2)
    cv.putText(frame, "terminal: angle number = test+save | h home | p print | q quit", (10, 56),
               cv.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
    return frame


def reset_arm(sorter, home):
    print("reset arm:", home, flush=True)
    sorter.arm.Arm_serial_servo_write6_array([int(v) for v in home], 1000)
    sleep(1)


def test_grip_with_angle(sorter, component, close_angle, home):
    name, posxy, px, py, width, height, conf = component
    joints = sorter.joints_for(posxy)
    if not joints:
        print("IK failed; cannot test grip.", flush=True)
        return False

    open_angle, _ = adaptive_grip_angles(width, height)
    pickup_open = [joints[0], joints[1], joints[2], joints[3], 265, open_angle]
    lift = [90, 80, 50, 50, 265, close_angle]

    print(
        f"test grip label={name} pixel=({px},{py}) posxy={posxy} "
        f"open={open_angle} close={close_angle}",
        flush=True,
    )
    sorter.arm.Arm_serial_servo_write6_array([90, 80, 50, 50, 265, open_angle], 1000)
    sleep(1)
    sorter.arm.Arm_serial_servo_write6_array([int(v) for v in pickup_open], 1000)
    sleep(1)
    sorter.arm.Arm_serial_servo_write(6, int(close_angle), 700)
    sleep(0.8)
    sorter.arm.Arm_serial_servo_write6_array([int(v) for v in lift], 1000)
    sleep(1)
    sorter.arm.Arm_serial_servo_write(6, int(open_angle), 700)
    sleep(0.8)
    reset_arm(sorter, home)
    return True


def read_command(prompt):
    try:
        return input(prompt).strip()
    except EOFError:
        return "q"


def main(args=None):
    cfg_dir, _ = package_share_dirs()
    sorter = ComponentSorter()
    calibration = ArmCalibration()
    xy, _ = read_XYT(str(cfg_dir / "XYT_config.txt"))
    dp = np.fromfile(str(cfg_dir / "dp.bin"), dtype=np.int32).reshape(4, 2)
    angles = load_angles(cfg_dir)

    home = [xy[0], xy[1], 0, 0, 90, 30]
    reset_arm(sorter, home)
    capture = open_camera(0)

    try:
        while capture.isOpened():
            ret, img = capture.read()
            if not ret:
                continue
            img = cv.resize(img, (640, 480))
            img = calibration.perspective_transform(dp, img)
            frame, components = sorter.detect(img)
            component = components[0] if components else None
            frame = draw_overlay(frame, component, angles)
            cv.imwrite("/tmp/component_grip_tuner.jpg", frame)
            if os.environ.get("DISPLAY"):
                cv.imshow("component_grip_tuner", frame)
                cv.waitKey(1)

            if component:
                name = component[0]
                print("detected:", name, "current saved angle:", angles.get(name), flush=True)
            else:
                print("no component detected. Put exactly one component in view.", flush=True)

            cmd = read_command("Input close angle, h=home, p=print, q=quit, Enter=rescan: ")
            if cmd == "":
                continue
            if cmd.lower() == "q":
                break
            if cmd.lower() == "h":
                reset_arm(sorter, home)
                continue
            if cmd.lower() == "p":
                print("current grip angles:", angles, flush=True)
                continue
            if not component:
                print("No detected component; cannot save angle.", flush=True)
                continue
            try:
                close_angle = int(float(cmd))
            except ValueError:
                print("Invalid angle. Input a number such as 135.", flush=True)
                continue
            close_angle = max(0, min(180, close_angle))
            if test_grip_with_angle(sorter, component, close_angle, home):
                angles[component[0]] = close_angle
                save_angles(cfg_dir, angles)
                print(f"saved: {component[0]} -> {close_angle}", flush=True)
    finally:
        capture.release()
        cv.destroyAllWindows()
        reset_arm(sorter, home)


if __name__ == "__main__":
    main()
