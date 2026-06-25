#!/usr/bin/env python3
# coding: utf-8

from time import sleep

from Arm_Lib import Arm_Device


HOME_POSE = [88, 123, 0, 0, 90, 30]


def main():
    arm = Arm_Device()
    sleep(0.2)
    print("reset arm to home:", HOME_POSE, flush=True)
    arm.Arm_serial_servo_write6_array(HOME_POSE, 1500)
    sleep(1.8)
    print("reset finished", flush=True)


if __name__ == "__main__":
    main()
