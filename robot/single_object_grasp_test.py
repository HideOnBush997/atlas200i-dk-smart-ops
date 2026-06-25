#!/usr/bin/env python3
# coding: utf-8

import math
import os
from pathlib import Path
from time import sleep

import cv2 as cv
import numpy as np

from component_sort import ComponentSorter, ArmCalibration, open_camera, package_share_dirs, read_XYT


WINDOW_NAME = "single_object_grasp_test"
ROI_PAD = 28


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def pixel_to_posxy(px, py):
    return (
        round((px - 320) / 4000, 5),
        round(((480 - py) / 3000) * 0.8 + 0.19, 5),
    )


def refine_grasp_pose(frame, component):
    name, posxy, px, py, width, height, conf = component
    x1 = clamp(int(px - width / 2) - ROI_PAD, 0, frame.shape[1] - 1)
    y1 = clamp(int(py - height / 2) - ROI_PAD, 0, frame.shape[0] - 1)
    x2 = clamp(int(px + width / 2) + ROI_PAD, 0, frame.shape[1] - 1)
    y2 = clamp(int(py + height / 2) + ROI_PAD, 0, frame.shape[0] - 1)
    roi = frame[y1:y2, x1:x2].copy()
    if roi.size == 0:
        return px, py, 0.0, None, (x1, y1, x2, y2)

    mask = np.zeros(roi.shape[:2], np.uint8)
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)

    rx = max(1, int(roi.shape[1] * 0.08))
    ry = max(1, int(roi.shape[0] * 0.08))
    rw = max(1, int(roi.shape[1] * 0.84))
    rh = max(1, int(roi.shape[0] * 0.84))
    rect = (rx, ry, rw, rh)

    try:
        cv.grabCut(roi, mask, rect, bgd, fgd, 5, cv.GC_INIT_WITH_RECT)
        fg = np.where((mask == cv.GC_FGD) | (mask == cv.GC_PR_FGD), 255, 0).astype("uint8")
    except cv.error:
        gray = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)
        _, fg = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    kernel = np.ones((5, 5), np.uint8)
    fg = cv.morphologyEx(fg, cv.MORPH_OPEN, kernel, iterations=1)
    fg = cv.morphologyEx(fg, cv.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv.findContours(fg, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if not contours:
        return px, py, 0.0, fg, (x1, y1, x2, y2)

    contour = max(contours, key=cv.contourArea)
    area = cv.contourArea(contour)
    if area < 80:
        return px, py, 0.0, fg, (x1, y1, x2, y2)

    hull = cv.convexHull(contour)
    m = cv.moments(hull)
    if m["m00"] > 0:
        cx = int(m["m10"] / m["m00"]) + x1
        cy = int(m["m01"] / m["m00"]) + y1
    else:
        cx = px
        cy = py

    angle = 0.0
    pts = hull.reshape(-1, 2).astype(np.float32)
    if len(pts) >= 5:
        mean = np.mean(pts, axis=0)
        cov = np.cov((pts - mean).T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        axis = eigvecs[:, np.argmax(eigvals)]
        angle = math.degrees(math.atan2(float(axis[1]), float(axis[0])))
    elif len(pts) >= 2:
        rect = cv.minAreaRect(hull)
        angle = float(rect[2])

    if angle < -90:
        angle += 180
    if angle > 90:
        angle -= 180

    return cx, cy, round(angle, 1), fg, (x1, y1, x2, y2)


def draw_overlay(frame, component, refined, selected_name):
    cx, cy, angle, fg, roi_box = refined
    x1, y1, x2, y2 = roi_box
    name, posxy, px, py, width, height, conf = component

    cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
    cv.circle(frame, (px, py), 6, (0, 255, 255), -1)
    cv.circle(frame, (cx, cy), 8, (255, 255, 0), 2)
    cv.line(frame, (cx - 16, cy), (cx + 16, cy), (255, 255, 0), 2)
    cv.line(frame, (cx, cy - 16), (cx, cy + 16), (255, 255, 0), 2)
    cv.putText(
        frame,
        f"{name} conf={conf:.2f} px=({cx},{cy}) angle={angle:.1f}",
        (10, 28),
        cv.FONT_HERSHEY_SIMPLEX,
        0.68,
        (0, 0, 255),
        2,
    )
    cv.putText(
        frame,
        f"label_center=({px},{py}) entity_center=({cx},{cy})",
        (10, 54),
        cv.FONT_HERSHEY_SIMPLEX,
        0.58,
        (0, 0, 255),
        2,
    )
    cv.putText(
        frame,
        "g grasp test | m move only | h home | q quit",
        (10, 80),
        cv.FONT_HERSHEY_SIMPLEX,
        0.58,
        (0, 0, 255),
        2,
    )
    if fg is not None and fg.size > 0:
        fg_vis = cv.cvtColor(fg, cv.COLOR_GRAY2BGR)
        fh, fw = fg_vis.shape[:2]
        small_w = min(160, fw)
        small_h = int(fh * small_w / fw)
        fg_vis = cv.resize(fg_vis, (small_w, small_h))
        frame[frame.shape[0] - small_h - 10:frame.shape[0] - 10, 10:10 + small_w] = fg_vis
    return frame


def move_to_pose(sorter, posxy, grip_open=65):
    joints = sorter.joints_for(posxy)
    if not joints:
        return None
    pickup = [joints[0], joints[1], joints[2], joints[3], 265, grip_open]
    sorter.arm.Arm_serial_servo_write6_array([90, 80, 50, 50, 265, grip_open], 1000)
    sleep(1)
    sorter.arm.Arm_serial_servo_write6_array([int(v) for v in pickup], 1000)
    sleep(1)
    return pickup


def grasp_test(sorter, component, home):
    name, posxy, px, py, width, height, conf = component
    cx, cy, angle, fg, roi_box = refine_grasp_pose(sorter.frame, component)
    posxy = pixel_to_posxy(cx, cy)
    grip_open, grip_close = sorter_adaptive_angles(width, height)
    print(
        f"target={name} entity_center=({cx},{cy}) angle={angle:.1f}deg "
        f"posxy={posxy} grip_open={grip_open} grip_close={grip_close}",
        flush=True,
    )
    joints = sorter.joints_for(posxy)
    if not joints:
        print("IK failed.", flush=True)
        return

    pickup = [joints[0], joints[1], joints[2], joints[3], 265, grip_open]
    sorter.arm.Arm_serial_servo_write6_array([90, 80, 50, 50, 265, grip_open], 1000)
    sleep(1)
    sorter.arm.Arm_serial_servo_write6_array([int(v) for v in pickup], 1000)
    sleep(1)
    sorter.arm.Arm_serial_servo_write(6, grip_close, 700)
    sleep(0.8)
    sorter.arm.Arm_serial_servo_write6_array([90, 80, 50, 50, 265, grip_close], 1000)
    sleep(1)
    sorter.arm.Arm_serial_servo_write(6, grip_open, 700)
    sleep(0.8)
    sorter.arm.Arm_serial_servo_write6_array(home, 1000)
    sleep(1)


def sorter_adaptive_angles(width, height):
    min_side = min(float(width), float(height))
    open_angle = int(max(55, min(105, 35 + min_side * 0.45)))
    close_angle = int(max(125, min(155, 160 - min_side * 0.18)))
    return open_angle, close_angle


def main(args=None):
    sorter = ComponentSorter()
    calibration = ArmCalibration()
    cfg_dir, _ = package_share_dirs()
    xy, _ = read_XYT(str(cfg_dir / "XYT_config.txt"))
    dp = np.fromfile(str(cfg_dir / "dp.bin"), dtype=np.int32).reshape(4, 2)

    home = [xy[0], xy[1], 0, 0, 90, 30]
    sorter.arm.Arm_serial_servo_write6_array(home, 1000)
    sleep(1)

    capture = open_camera(0)
    try:
        while capture.isOpened():
            ret, img = capture.read()
            if not ret:
                continue
            img = cv.resize(img, (640, 480))
            img = calibration.perspective_transform(dp, img)
            frame, components = sorter.detect(img)
            if components:
                component = max(components, key=lambda item: float(item[6]))
                refined = refine_grasp_pose(frame, component)
                sorter.frame = draw_overlay(frame.copy(), component, refined, component[0])
                cx, cy, angle, _, _ = refined
                print(
                    f"best={component[0]} pixel=({cx},{cy}) angle={angle:.1f}deg conf={component[6]:.2f}",
                    flush=True,
                )
            else:
                sorter.frame = frame
                cv.putText(
                    sorter.frame,
                    "no target",
                    (10, 28),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.68,
                    (0, 0, 255),
                    2,
                )

            cv.imwrite("/tmp/single_object_grasp_test.jpg", sorter.frame)
            if os.environ.get("DISPLAY"):
                cv.imshow(WINDOW_NAME, sorter.frame)
                key = cv.waitKey(30) & 0xFF
            else:
                key = 255

            if key == ord("q"):
                break
            if key == ord("h"):
                sorter.arm.Arm_serial_servo_write6_array(home, 1000)
                sleep(1)
            elif key == ord("m") and components:
                component = max(components, key=lambda item: float(item[6]))
                cx, cy, _, _, _ = refine_grasp_pose(frame, component)
                posxy = pixel_to_posxy(cx, cy)
                print(f"move-only pixel=({cx},{cy}) posxy={posxy}", flush=True)
                grip_open, _ = sorter_adaptive_angles(component[4], component[5])
                move_to_pose(sorter, posxy, grip_open=grip_open)
            elif key == ord("g") and components:
                component = max(components, key=lambda item: float(item[6]))
                grasp_test(sorter, component, home)
    finally:
        capture.release()
        cv.destroyAllWindows()
        sorter.arm.Arm_serial_servo_write6_array(home, 1000)


if __name__ == "__main__":
    main()
