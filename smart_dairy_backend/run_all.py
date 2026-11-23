"""
One-Command Build, Seed, Server & IoT Simulator Launcher
for AI-Driven Smart Dairy Supply Chain Optimization
"""

import os
import subprocess
import sys
import time
import webbrowser


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(base_dir)

    print("=" * 70)
    print("SMART DAIRY SUPPLY CHAIN DSS — ONE-COMMAND BUILD & LAUNCHER")
    print("=" * 70)

    # 1. Seed Relational Database & Train ML Models (if not already built)
    db_file = os.path.join(base_dir, "dairy_supply_chain.db")
    if not os.path.exists(db_file):
        print("[1/3] Seeding 1,500-row relational database & training ML models...")
        subprocess.run([sys.executable, "seed_database.py"], check=True)
    else:
        print("[1/3] Found existing dairy_supply_chain.db — skipping re-seed.")

    # 2. Start FastAPI Backend Server
    print("[2/3] Starting FastAPI + SQLAlchemy server on http://127.0.0.1:8000 ...")
    api_proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", "8000"]
    )
    time.sleep(2.5)

    # 3. Start Background Multi-Tanker Cold-Chain IoT Streamer
    print("[3/3] Starting live 5-tanker Cold-Chain IoT telemetry simulator ...")
    iot_proc = subprocess.Popen(
        [sys.executable, "mock_iot_streamer.py", "--url", "http://127.0.0.1:8000", "--interval", "2.0"]
    )

    print("\n[READY] Open index.html in your browser (or visit http://127.0.0.1:8000/docs)")
    print("        Press Ctrl+C to shut down both the API server and IoT simulator.\n")

    index_path = os.path.abspath(os.path.join(base_dir, "..", "index.html"))
    if os.path.exists(index_path):
        webbrowser.open(f"file://{index_path}")
    else:
        webbrowser.open("http://127.0.0.1:8000/docs")

    try:
        api_proc.wait()
    except KeyboardInterrupt:
        print("\n[SHUTDOWN] Stopping IoT simulator and FastAPI backend...")
        iot_proc.terminate()
        api_proc.terminate()
        iot_proc.wait()
        api_proc.wait()
        print("[DONE] Clean shutdown complete.")


if __name__ == "__main__":
    main()
