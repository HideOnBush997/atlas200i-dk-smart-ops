#!/usr/bin/env python3
# coding: utf-8

"""
Board-side speech recognition wrapper based on the Atlas demo.

Flow:
1. Record speech from microphone to a wav file.
2. Validate wav format and loudness.
3. Reuse the board demo recognizer to obtain text.
4. Save the recognized text for frontend or control logic.
"""

from __future__ import annotations

import argparse
import audioop
import os
import re
import subprocess
import sys
import wave
from pathlib import Path


DEFAULT_DEMO_ROOT = Path("/home/HwHiAiUser/samples/notebooks/09-speech-recognition")
DEFAULT_WAV = DEFAULT_DEMO_ROOT / "command.wav"
DEFAULT_TEXT = DEFAULT_DEMO_ROOT / "asr_result.txt"
DEFAULT_SCRIPT = DEFAULT_DEMO_ROOT / "run_asr.py"


def run_cmd(command, cwd=None, env=None):
    result = subprocess.run(
        command,
        cwd=str(cwd) if cwd else None,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return result.returncode, result.stdout


def record_wav(device, duration, wav_path, sample_rate):
    command = [
        "arecord",
        "-D",
        device,
        "-f",
        "S16_LE",
        "-r",
        str(sample_rate),
        "-c",
        "1",
        "-d",
        str(duration),
        str(wav_path),
    ]
    print("record command:", " ".join(command), flush=True)
    code, output = run_cmd(command)
    print(output, end="", flush=True)
    if code != 0:
        raise RuntimeError("record failed")


def inspect_wav(wav_path):
    with wave.open(str(wav_path), "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_rate = wav_file.getframerate()
        sample_width = wav_file.getsampwidth()
        frames = wav_file.getnframes()
        data = wav_file.readframes(frames)

    duration = 0.0 if sample_rate == 0 else frames / float(sample_rate)
    rms = audioop.rms(data, sample_width) if data else 0
    peak = audioop.max(data, sample_width) if data else 0

    info = {
        "channels": channels,
        "sample_rate": sample_rate,
        "sample_width": sample_width,
        "frames": frames,
        "duration": duration,
        "rms": rms,
        "peak": peak,
    }
    print("wav info:", info, flush=True)
    return info


def extract_text(output):
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    for index, line in enumerate(lines):
        if "识别结果" in line:
            if index + 1 < len(lines):
                return lines[index + 1]
            return ""
    for line in reversed(lines):
        if line.startswith("[INFO]") or line.startswith("202"):
            continue
        return line
    return ""


def recognize_with_demo(wav_path, demo_root, script_path):
    if not script_path.exists():
        raise FileNotFoundError(f"ASR demo script not found: {script_path}")

    env = os.environ.copy()
    ascend_pythonpath = "/usr/local/Ascend/thirdpart/aarch64/acllite"
    env["PYTHONPATH"] = (
        ascend_pythonpath
        if not env.get("PYTHONPATH")
        else ascend_pythonpath + ":" + env["PYTHONPATH"]
    )

    command = ["/usr/local/miniconda3/bin/python", str(script_path), str(wav_path)]
    code, output = run_cmd(command, cwd=demo_root, env=env)
    print(output, end="", flush=True)
    if code != 0:
        raise RuntimeError("ASR demo failed")

    text = extract_text(output)
    return text


def save_text(text_path, text):
    text_path.write_text(text.strip() + "\n", encoding="utf-8")
    print("saved text:", text_path, flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Atlas demo speech recognition wrapper")
    parser.add_argument("--device", default="plughw:0,0", help="arecord device, e.g. plughw:0,0")
    parser.add_argument("--duration", type=int, default=5, help="record seconds")
    parser.add_argument("--sample-rate", type=int, default=16000, help="wav sample rate")
    parser.add_argument("--wav", type=Path, default=DEFAULT_WAV, help="recorded wav output path")
    parser.add_argument("--text", type=Path, default=DEFAULT_TEXT, help="recognized text output path")
    parser.add_argument("--demo-root", type=Path, default=DEFAULT_DEMO_ROOT, help="speech demo directory")
    parser.add_argument("--script", type=Path, default=DEFAULT_SCRIPT, help="speech demo runner script")
    parser.add_argument("--skip-record", action="store_true", help="recognize an existing wav file")
    parser.add_argument("--min-rms", type=int, default=200, help="minimum loudness threshold")
    return parser.parse_args()


def main():
    args = parse_args()
    args.wav.parent.mkdir(parents=True, exist_ok=True)
    args.text.parent.mkdir(parents=True, exist_ok=True)

    if not args.skip_record:
        record_wav(args.device, args.duration, args.wav, args.sample_rate)

    info = inspect_wav(args.wav)
    if info["channels"] != 1:
        raise RuntimeError("wav must be mono")
    if info["sample_rate"] not in (16000, 48000):
        raise RuntimeError("wav sample rate must be 16000 or 48000")
    if info["rms"] < args.min_rms:
        raise RuntimeError(f"wav loudness too low, rms={info['rms']}")

    text = recognize_with_demo(args.wav, args.demo_root, args.script)
    save_text(args.text, text)
    print("recognized text:", text, flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("voice_asr_demo error:", exc, flush=True)
        sys.exit(1)
