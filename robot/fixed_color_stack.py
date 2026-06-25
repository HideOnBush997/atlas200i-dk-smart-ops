#!/usr/bin/env python3
# coding: utf-8

"""
Fixed color block stack-then-sort workflow.

Start state:
    Four color blocks are placed on their matching color areas.

Phase 1:
    Move blocks to the middle stack from bottom to top:
    blue -> green -> red -> yellow.

Phase 2:
    Move blocks from the middle stack back to their color areas:
    yellow -> red -> green -> blue.

This program uses fixed arm poses only. It does not use camera, model, or
dofbot_server.
"""

from time import sleep

from Arm_Lib import Arm_Device


CLAMP_OPEN = 60
CLAMP_CLOSE = 135

# p=[S1, S2, S3, S4, S5]. The gripper angle is controlled separately.
P_HOME = [90, 130, 0, 0, 90]
P_TOP = [90, 80, 50, 50, 270]

P_LAYER_1 = [92, 48, 35, 30, 270]
P_LAYER_2 = [94, 64, 23, 34, 270]
P_LAYER_3 = [94, 61, 44, 17, 270]
P_LAYER_4 = [91, 73, 40, 17, 270]

P_YELLOW = [64, 18, 64, 56, 270]
P_RED = [119, 15, 66, 56, 270]
P_GREEN = [139, 61, 16, 29, 270]
P_BLUE = [44, 64, 15, 28, 270]

STACK_TO_MIDDLE_PLAN = [
    ("blue", P_BLUE, P_LAYER_1),
    ("green", P_GREEN, P_LAYER_2),
    ("red", P_RED, P_LAYER_3),
    ("yellow", P_YELLOW, P_LAYER_4),
]

SORT_BACK_PLAN = [
    ("yellow", P_LAYER_4, P_YELLOW),
    ("red", P_LAYER_3, P_RED),
    ("green", P_LAYER_2, P_GREEN),
    ("blue", P_LAYER_1, P_BLUE),
]


arm = None


def clamp(close):
    angle = CLAMP_CLOSE if close else CLAMP_OPEN
    print("clamp", "close" if close else "open", angle, flush=True)
    arm.Arm_serial_servo_write(6, angle, 500)
    sleep(0.6)


def move_pose(pose, move_time=1000, gripper=None):
    target = list(pose)
    if gripper is None:
        gripper = CLAMP_CLOSE
    target.append(gripper)
    print("move", target, flush=True)
    arm.Arm_serial_servo_write6_array([int(v) for v in target], move_time)
    sleep(move_time / 1000.0 + 0.15)


def lift_up():
    print("lift up", flush=True)
    arm.Arm_serial_servo_write(2, 90, 1500)
    arm.Arm_serial_servo_write(3, 90, 1500)
    arm.Arm_serial_servo_write(4, 90, 1500)
    sleep(1.6)


def move_block(label, from_pose, to_pose):
    print("=== move block", label, "===", flush=True)
    move_pose(P_TOP, 1000, CLAMP_OPEN)
    move_pose(from_pose, 1000, CLAMP_OPEN)
    clamp(True)
    move_pose(P_TOP, 1000, CLAMP_CLOSE)
    move_pose(to_pose, 1000, CLAMP_CLOSE)
    clamp(False)
    sleep(0.2)
    lift_up()
    move_pose(P_HOME, 1100, CLAMP_OPEN)
    sleep(0.5)


def run_plan(title, plan):
    print("\n###", title, "###", flush=True)
    for label, from_pose, to_pose in plan:
        move_block(label, from_pose, to_pose)


def main(args=None):
    global arm
    arm = Arm_Device()
    sleep(0.2)

    print("Start fixed color stack workflow.", flush=True)
    print("Initial blocks: blue/green/red/yellow on matching color areas.", flush=True)
    clamp(False)
    move_pose(P_HOME, 1000, CLAMP_OPEN)
    sleep(1)

    run_plan("Phase 1: stack blocks to middle", STACK_TO_MIDDLE_PLAN)
    run_plan("Phase 2: sort stacked blocks back", SORT_BACK_PLAN)

    clamp(False)
    move_pose(P_HOME, 1000, CLAMP_OPEN)
    print("Fixed color stack workflow finished.", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Program closed.", flush=True)
        if arm is not None:
            clamp(False)
            move_pose(P_HOME, 1000, CLAMP_OPEN)
