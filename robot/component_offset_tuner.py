#!/usr/bin/env python3
# coding: utf-8

import os
from pathlib import Path
from time import sleep

import cv2 as cv
import numpy as np

from .component_sort import ComponentSorter, ArmCalibration, open_camera, package_share_dirs, read_XYT


OFFSET_STEP = 0.001
SOURCE_OFFSET = Path(
    "/home/HwHiAiUser/E2ESamples/ros2_robot_arm/ros2_ws/src/"
    "dofbot_garbage_yolov5/dofbot_garbage_yolov5/config/offset.txt"
)


def save_offsets(cfg_dir, sorter):
    text = f"{sorter.offset}\n{sorter.x_offset}\n"
    for path in [cfg_dir / "offset.txt", SOURCE_OFFSET]:
        try:
            path.write_text(text, encoding="utf-8")
            print("saved offset:", path, text.strip().replace("\n", ", "), flush=True)
        except Exception as exc:
            print("save offset failed:", path, exc, flush=True)


def draw_overlay(frame, sorter, components, selected_index):
    lines = [
        f"x_offset={sorter.x_offset:.4f} y_offset={sorter.offset:.4f}",
        "a/d x- x+ | w/s y+ y- | m move | y save | h home | q quit",
    ]
    if components:
        selected_index %= len(components)
        name, posxy, px, py, width, height, conf = components[selected_index]
        lines.append(f"selected {selected_index + 1}/{len(components)} {name} pos={posxy} conf={conf:.2f}")
        cv.circle(frame, (int(px), int(py)), 10, (255, 0, 255), 3)
    else:
        lines.append("no component detected")

    y = 26
    for line in lines:
        cv.putText(frame, line, (10, y), cv.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 255), 2)
        y += 24
    return frame


def move_to_target(sorter, component):
    name, posxy, px, py, width, height, conf = component
    print("move target:", name, "pixel:", (px, py), "posxy:", posxy, flush=True)
    joints = sorter.joints_for(posxy)
    if not joints:
        print("IK failed; cannot move.", flush=True)
        return

    pickup = [joints[0], joints[1], joints[2], joints[3], 265, 60]
    print("move only, no grip:", pickup, flush=True)
    sorter.arm.Arm_serial_servo_write6_array([90, 80, 50, 50, 265, 60], 1000)
    sleep(1)
    sorter.arm.Arm_serial_servo_write6_array([int(v) for v in pickup], 1000)
    sleep(1)


def main(args=None):
    cfg_dir, _ = package_share_dirs()
    sorter = ComponentSorter()
    calibration = ArmCalibration()
    xy, _ = read_XYT(str(cfg_dir / "XYT_config.txt"))
    dp = np.fromfile(str(cfg_dir / "dp.bin"), dtype=np.int32).reshape(4, 2)
    capture = open_camera(0)
    selected_index = 0
    auto_move = True

    home = [xy[0], xy[1], 0, 0, 90, 30]
    sorter.arm.Arm_serial_servo_write6_array(home, 1000)
    sleep(1)

    try:
        while capture.isOpened():
            ret, img = capture.read()
            if not ret:
                continue
            img = cv.resize(img, (640, 480))
            img = calibration.perspective_transform(dp, img)
            frame, components = sorter.detect(img)
            frame = draw_overlay(frame, sorter, components, selected_index)
            cv.imwrite("/tmp/component_offset_tuner.jpg", frame)
            if os.environ.get("DISPLAY"):
                cv.imshow("component_offset_tuner", frame)
                key = cv.waitKey(30) & 0xFF
            else:
                key = 255

            if key == ord("q"):
                break
            if key == ord("a"):
                sorter.x_offset -= OFFSET_STEP
                auto_move = True
            elif key == ord("d"):
                sorter.x_offset += OFFSET_STEP
                auto_move = True
            elif key == ord("w"):
                sorter.offset += OFFSET_STEP
                auto_move = True
            elif key == ord("s"):
                sorter.offset -= OFFSET_STEP
                auto_move = True
            elif key == ord("y"):
                save_offsets(cfg_dir, sorter)
            elif key == ord("m") and components:
                auto_move = True
            elif key == ord("h"):
                print("return home:", home, flush=True)
                sorter.arm.Arm_serial_servo_write6_array([int(v) for v in home], 1000)
                sleep(1)

            if components and auto_move:
                selected_index %= len(components)
                move_to_target(sorter, components[selected_index])
                auto_move = False

    finally:
        capture.release()
        cv.destroyAllWindows()
        sorter.arm.Arm_serial_servo_write6_array(home, 1000)


if __name__ == "__main__":
    main()
