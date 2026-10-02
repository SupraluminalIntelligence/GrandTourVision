"""Capture real visionOS simulator framebuffers during native UI-test holds.

No UI event synthesis. Run beside the XCTest walkthrough. Uses official simctl.
"""
import argparse
import os
from pathlib import Path
import subprocess
import signal
import time

parser = argparse.ArgumentParser()
parser.add_argument("--udid", required=True)
parser.add_argument("--log", required=True, type=Path)
parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "Screenshots")
parser.add_argument("--focused", action="store_true")
parser.add_argument("--immersive", action="store_true")
parser.add_argument("--video", action="store_true")
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
stages = ["01-main-controls", "02-activation-baseline", "03-layer-checkpoint-token", "04-activation-3d-scene"]
if args.focused:
    stages = ["05-synthetic-focused-baseline", "06-synthetic-focused-comparison", "07-public-tiny-gpt-focused"]
if args.immersive:
    stages = ["08-room-scale-animation", "09-room-scale-locked-comparison"]
seen = set()
started = time.monotonic()
while len(seen) < len(stages) and time.monotonic() - started < 240:
    text = args.log.read_text(errors="replace") if args.log.exists() else ""
    for stage in stages:
        if stage not in seen and "GRANDTOUR_CAPTURE:" + stage in text:
            time.sleep(2)
            subprocess.run(["xcrun", "simctl", "io", args.udid, "screenshot", str(args.output / (stage + ".png"))], check=True)
            seen.add(stage)
            print("Captured " + stage, flush=True)
            if args.video and stage == "08-room-scale-animation":
                recording = subprocess.Popen(["xcrun", "simctl", "io", args.udid, "recordVideo", "--codec=h264", "--force", str(args.output / "08-room-scale-animation.mov")])
                time.sleep(10)
                recording.send_signal(signal.SIGINT)
                try:
                    recording.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    recording.terminate()
                    recording.wait(timeout=5)
                print("Simulator video exit: " + str(recording.returncode), flush=True)
    time.sleep(0.25)
if len(seen) != len(stages):
    raise SystemExit("Incomplete screenshot capture; inspect the UI test log")
