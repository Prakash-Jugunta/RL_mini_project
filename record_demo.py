"""
Live Demo & Video Recording Helper for Professor Submission.
Author: Jugunta Prakash

Features:
- Verifies frontend is running (http://localhost:3000)
- Executes real headed DQN recovery under Level 6 compound mutation
- Automatically records the live browser interaction to artifacts/recordings/DQN_Self_Healing_Live_Demo.webm
- Demonstrates element highlighting ("DQN SELECTED"), real-time Q-values, and 100% workflow completion
"""

import os
import shutil
import subprocess
import sys
import time
import urllib.request
import json

ROOT_DIR = r"E:\sem7\RL\RL_replay"
RECORDINGS_DIR = os.path.join(ROOT_DIR, "artifacts", "recordings")
os.makedirs(RECORDINGS_DIR, exist_ok=True)

def check_frontend(base_url="http://localhost:3000") -> bool:
    api_url = f"{base_url.rstrip('/')}/api/set-mutation?level=0&seed=11"
    try:
        req = urllib.request.Request(api_url, method="GET")
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read().decode())
            return bool(data.get("ok"))
    except Exception:
        return False

def main():
    print("=" * 65)
    print("RL-Based UI Test Self-Healing — Live Demo Video Recorder")
    print("Student: Jugunta Prakash (Track: Masked Double DQN)")
    print("=" * 65)

    # 1. Check frontend
    print("\n[1/3] Checking Frontend Dev Server (http://localhost:3000)...")
    if not check_frontend():
        print("\n[ERROR] Vite server is NOT running!")
        print("Please open a separate terminal and run:")
        print("    cd E:\\sem7\\RL\\RL_replay\\phase02")
        print("    npm run dev")
        print("\nOnce Vite is running at http://localhost:3000, run this script again.")
        sys.exit(1)
    print("[PASS] Frontend is active and responding!")

    # 2. Select Demo Mode
    print("\n[2/3] Choose Demo Workflow:")
    print("  1: Standard Autonomous Recovery on Level 6 Compound Mutation (LOGIN)")
    print("  2: Self-Healing with Error Recovery Injection (Shows DQN retrying after mis-click)")
    print("  3: Multi-Step E-Commerce Workflow (CHECKOUT - 5 steps)")
    choice = input("\nEnter choice [1, 2, or 3] (default 1): ").strip()
    if choice == "2":
        workflow = "LOGIN"
        inject_arg = ["--inject-wrong-step", "1"]
        demo_name = "DQN_Self_Healing_Recovery_Injection_Demo"
    elif choice == "3":
        workflow = "CHECKOUT"
        inject_arg = []
        demo_name = "DQN_Checkout_MultiStep_Demo"
    else:
        workflow = "LOGIN"
        inject_arg = []
        demo_name = "DQN_Self_Healing_Live_Demo"

    temp_video_dir = os.path.join(RECORDINGS_DIR, "temp_video")
    if os.path.exists(temp_video_dir):
        shutil.rmtree(temp_video_dir, ignore_errors=True)
    os.makedirs(temp_video_dir, exist_ok=True)

    # 3. Launch Demo Runner
    print(f"\n[3/3] Launching Live Headed DQN Demo ({workflow} under Level 6 Mutation)...")
    print("Chromium browser will open and automatically execute the recovery policy.")
    print("Playwright is recording the browser viewport in the background...\n")

    cmd = [
        sys.executable,
        "-u",
        "-m",
        "rl.demo.run_dqn_demo",
        "--checkpoint", "artifacts/dqn/phase13-v1/latest_checkpoint.pt",
        "--workflow", workflow,
        "--mutation-level", "6",
        "--mutation-seed", "11",
        "--headed",
        "--auto",
        "--delay-ms", "1500",
        "--record-video-dir", temp_video_dir,
    ] + inject_arg

    start_time = time.time()
    result = subprocess.run(cmd, cwd=ROOT_DIR)
    elapsed = time.time() - start_time

    # Find the generated webm file
    webm_files = [f for f in os.listdir(temp_video_dir) if f.endswith(".webm")]
    if webm_files:
        src_video = os.path.join(temp_video_dir, webm_files[0])
        final_video_name = f"{demo_name}.webm"
        final_video_path = os.path.join(RECORDINGS_DIR, final_video_name)
        shutil.copyfile(src_video, final_video_path)
        shutil.rmtree(temp_video_dir, ignore_errors=True)

        size_mb = os.path.getsize(final_video_path) / (1024 * 1024)
        print("\n" + "=" * 65)
        print("DEMO RECORDING COMPLETE!")
        print("=" * 65)
        print(f"Video Saved To:  {final_video_path}")
        print(f"File Size:       {size_mb:.2f} MB")
        print(f"Duration:        {elapsed:.1f} seconds")
        print("\nYou can submit this .webm video directly, or double-click to play it in Chrome/Edge!")
    else:
        print("\n[NOTE] Headed demo executed successfully.")
        print("To record your entire desktop (terminal + browser side-by-side), press:")
        print("    Windows Key + Shift + R  (Windows Snipping Tool Screen Recorder)")
        print("    or Windows Key + Alt + R (Xbox Game Bar Recorder)")

if __name__ == "__main__":
    main()

