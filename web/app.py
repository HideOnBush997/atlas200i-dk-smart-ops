#!/usr/bin/env python3
# coding: utf-8

import os
import signal
import subprocess
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

from flask import Flask, jsonify, render_template, send_file


BASE_DIR = Path("/home/HwHiAiUser/E2ESamples/ros2_robot_arm")
WS_DIR = BASE_DIR / "ros2_ws"
DEBUG_IMAGES = [
    Path("/tmp/component_sort_debug_frame.jpg"),
    Path("/tmp/component_sort_detect.jpg"),
    Path("/tmp/single_object_grasp_test.jpg"),
    Path("/tmp/color_debug_frame.jpg"),
    Path("/tmp/color_sort_debug_frame.jpg"),
    Path("/tmp/color_sort_debug_frame_latest.jpg"),
]

COMMANDS = {
    "server": {
        "name": "逆解服务",
        "cmd": "ros2 run dofbot_moveit dofbot_server",
        "match": "dofbot_server",
        "cwd": BASE_DIR,
    },
    "component_sort": {
        "name": "元器件分拣",
        "cmd": "ros2 run dofbot_garbage_yolov5 component_sort",
        "match": "component_sort",
        "cwd": BASE_DIR,
    },
    "color_stack": {
        "name": "色块堆叠",
        "cmd": "ros2 run robot_arm_color_stacking fixed_color_stack",
        "match": "fixed_color_stack",
        "cwd": BASE_DIR,
    },
    "color_sort": {
        "name": "色块分拣",
        "cmd": "ros2 run robot_arm_color_stacking color_stacking",
        "match": "color_stacking",
        "cwd": BASE_DIR,
    },
    "reset": {
        "name": "机械臂复位",
        "cmd": "python3 web/arm_reset.py",
        "match": "arm_reset.py",
        "cwd": BASE_DIR,
    },
}
TASK_KEYS = ["component_sort", "color_stack", "color_sort"]

app = Flask(__name__)
processes = {}
events = []
CN_TZ = timezone(timedelta(hours=8))


def add_event(message):
    events.append({"time": datetime.now(CN_TZ).strftime("%H:%M:%S"), "message": message})
    del events[:-80]


def kill_matching(match):
    if not match:
        return
    try:
        output = subprocess.check_output(["ps", "-ef"], text=True)
    except Exception:
        return
    for line in output.splitlines():
        if match not in line or "grep" in line or "plink" in line:
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        try:
            pid = int(parts[1])
        except ValueError:
            continue
        if pid == os.getpid():
            continue
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def log_path(key):
    return Path(f"/tmp/web_panel_task_{key}.log")


def launch_shell(key, command, cwd):
    wrapped = f"cd {cwd} && . setenv.sh && {command}"
    log_file = log_path(key).open("ab")
    return subprocess.Popen(
        ["bash", "-lc", wrapped],
        stdout=log_file,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
    )


def is_running(key):
    proc = processes.get(key)
    return bool(proc and proc.poll() is None)


def stop_process(key):
    proc = processes.get(key)
    if not proc or proc.poll() is not None:
        processes.pop(key, None)
        return False
    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    try:
        proc.wait(timeout=3)
    except subprocess.TimeoutExpired:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    processes.pop(key, None)
    return True


def latest_image():
    existing = [path for path in DEBUG_IMAGES if path.exists() and path.stat().st_size > 0]
    if not existing:
        return None
    return max(existing, key=lambda path: path.stat().st_mtime)


def image_info(path):
    if not path:
        return {"image": False, "image_mtime": 0}
    try:
        return {"image": True, "image_mtime": path.stat().st_mtime}
    except OSError:
        return {"image": False, "image_mtime": 0}


def latest_logs():
    lines = []
    for key, item in COMMANDS.items():
        path = log_path(key)
        if not path.exists():
            continue
        try:
            tail = path.read_text(encoding="utf-8", errors="ignore").splitlines()[-5:]
        except Exception:
            continue
        for line in tail:
            if line.strip():
                lines.append({"time": item["name"], "message": line[-120:]})
    return lines[-12:]


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/start/<key>", methods=["POST"])
def start(key):
    item = COMMANDS.get(key)
    if not item:
        return jsonify({"ok": False, "message": "未知任务"}), 404
    if is_running(key):
        return jsonify({"ok": True, "message": f"{item['name']} 已在运行"})
    if key != "server":
        for old_key in TASK_KEYS:
            stop_process(old_key)
            kill_matching(COMMANDS[old_key].get("match"))
    kill_matching(item.get("match"))
    log_path(key).write_text("", encoding="utf-8")
    processes[key] = launch_shell(key, item["cmd"], item["cwd"])
    add_event(f"启动 {item['name']}")
    return jsonify({"ok": True, "message": f"已启动 {item['name']}"})


@app.route("/api/stop/<key>", methods=["POST"])
def stop(key):
    item = COMMANDS.get(key)
    if not item:
        return jsonify({"ok": False, "message": "未知任务"}), 404
    stopped = stop_process(key)
    kill_matching(item.get("match"))
    add_event(f"停止 {item['name']}" if stopped else f"{item['name']} 未运行")
    return jsonify({"ok": True, "message": "已停止" if stopped else "未运行"})


@app.route("/api/reset", methods=["POST"])
def reset_arm():
    for key in TASK_KEYS:
        stop_process(key)
        kill_matching(COMMANDS[key].get("match"))
    stop_process("reset")
    kill_matching(COMMANDS["reset"].get("match"))
    log_path("reset").write_text("", encoding="utf-8")
    processes["reset"] = launch_shell("reset", COMMANDS["reset"]["cmd"], COMMANDS["reset"]["cwd"])
    add_event("机械臂复位")
    return jsonify({"ok": True, "message": "正在复位"})


@app.route("/api/emergency", methods=["POST"])
def emergency():
    for key in list(processes):
        stop_process(key)
    for item in COMMANDS.values():
        kill_matching(item.get("match"))
    add_event("急停：已停止面板启动的全部任务")
    return jsonify({"ok": True, "message": "急停完成"})


@app.route("/api/status")
def status():
    running = {key: is_running(key) for key in COMMANDS}
    current = next((COMMANDS[key]["name"] for key, val in running.items() if val), "待机")
    image = latest_image()
    info = image_info(image)
    merged_events = events[-8:] + latest_logs()
    return jsonify({
        "running": running,
        "current": current,
        "events": merged_events[-12:],
        "image": info["image"],
        "image_mtime": info["image_mtime"],
        "timestamp": datetime.now(CN_TZ).strftime("%Y/%m/%d %H:%M:%S"),
    })


@app.route("/api/frame")
def frame():
    image = latest_image()
    if not image:
        return ("", 404)
    response = send_file(str(image), mimetype="image/jpeg", max_age=0)
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


if __name__ == "__main__":
    add_event("控制面板启动")
    app.run(host="0.0.0.0", port=5002, threaded=True)
