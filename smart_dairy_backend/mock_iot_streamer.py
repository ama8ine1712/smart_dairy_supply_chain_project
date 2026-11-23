"""
AI-Driven Smart Dairy Supply Chain Optimization
Option 2: Multi-Tanker Cold-Chain IoT Edge Gateway Simulator

Simulates 5 refrigerated milk tankers transporting raw milk from Gujarat BMC hubs
(Anand, Mehsana, Banaskantha, Sabarkantha, Kheda) to Mother Dairy Gandhinagar,
streaming real-time temperature, door status, compressor RPM, and GPS telemetry
into POST /api/iot/telemetry.
"""

import argparse
import json
import random
import time
import urllib.error
import urllib.request
from datetime import datetime

try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False

DEPOT = {"name": "Mother Dairy Processing Plant - Gandhinagar", "lat": 23.2156, "lon": 72.6369}

FLEET_STATE = [
    {
        "tanker_id": "TKR-01",
        "batch_id": "BATCH-11496",
        "corridor": "CORR-A1-Express (Anand -> Gandhinagar)",
        "start_lat": 22.5645,
        "start_lon": 72.9289,
        "progress": 0.15,
        "reefer_temp_c": 3.25,
        "ambient_temp_c": 32.4,
        "door_open_ticks": 0,
        "fault_mode": False,
    },
    {
        "tanker_id": "TKR-03",
        "batch_id": "BATCH-11497",
        "corridor": "CORR-B2-Rural (Mehsana -> Gandhinagar)",
        "start_lat": 23.5880,
        "start_lon": 72.3693,
        "progress": 0.40,
        "reefer_temp_c": 3.55,
        "ambient_temp_c": 37.8,
        "door_open_ticks": 0,
        "fault_mode": False,
    },
    {
        "tanker_id": "TKR-07",
        "batch_id": "BATCH-11498",
        "corridor": "CORR-C3-Highway (Banaskantha -> Gandhinagar)",
        "start_lat": 24.1711,
        "start_lon": 72.4386,
        "progress": 0.25,
        "reefer_temp_c": 3.55,
        "ambient_temp_c": 38.5,
        "door_open_ticks": 0,
        "fault_mode": False,
    },
    {
        "tanker_id": "TKR-12",
        "batch_id": "BATCH-11499",
        "corridor": "CORR-D4-Feeder (Sabarkantha -> Gandhinagar)",
        "start_lat": 23.8477,
        "start_lon": 72.9933,
        "progress": 0.60,
        "reefer_temp_c": 3.80,
        "ambient_temp_c": 35.1,
        "door_open_ticks": 0,
        "fault_mode": False,
    },
    {
        "tanker_id": "TKR-15",
        "batch_id": "BATCH-11500",
        "corridor": "CORR-A1-Express (Kheda -> Gandhinagar)",
        "start_lat": 22.7507,
        "start_lon": 72.6847,
        "progress": 0.50,
        "reefer_temp_c": 3.10,
        "ambient_temp_c": 31.2,
        "door_open_ticks": 0,
        "fault_mode": False,
    },
]


def step_tanker_physics(tanker: dict, forced_fault_tanker: str | None = None) -> dict:
    """Advances GPS coordinates and computes chamber thermal physics for one tick."""
    tanker["progress"] += random.uniform(0.015, 0.035)
    if tanker["progress"] >= 1.0:
        tanker["progress"] = 0.0

    lat = (
        tanker["start_lat"]
        + (DEPOT["lat"] - tanker["start_lat"]) * tanker["progress"]
        + random.uniform(-0.0015, 0.0015)
    )
    lon = (
        tanker["start_lon"]
        + (DEPOT["lon"] - tanker["start_lon"]) * tanker["progress"]
        + random.uniform(-0.0015, 0.0015)
    )

    tanker["ambient_temp_c"] = round(
        max(24.0, min(42.5, tanker["ambient_temp_c"] + random.uniform(-0.35, 0.35))), 1
    )

    is_fault = tanker["fault_mode"] or (forced_fault_tanker == tanker["tanker_id"])
    if tanker["door_open_ticks"] > 0:
        tanker["door_open_ticks"] -= 1
    elif is_fault and random.random() < 0.25:
        tanker["door_open_ticks"] = random.randint(2, 3)
    door_status = "OPEN" if (tanker["door_open_ticks"] > 0 or is_fault) else "CLOSED"

    if is_fault:
        tanker["reefer_temp_c"] += random.uniform(0.22, 0.45)
        if tanker["reefer_temp_c"] > 6.4:
            tanker["reefer_temp_c"] = round(random.uniform(5.6, 6.3), 2)
    elif door_status == "OPEN":
        heat_gain = (tanker["ambient_temp_c"] - tanker["reefer_temp_c"]) * 0.014
        tanker["reefer_temp_c"] += heat_gain
    else:
        cooling = (3.25 - tanker["reefer_temp_c"]) * 0.28 + random.uniform(-0.08, 0.08)
        tanker["reefer_temp_c"] += cooling

    tanker["reefer_temp_c"] = round(max(2.2, min(7.5, tanker["reefer_temp_c"])), 2)

    return {
        "tanker_id": tanker["tanker_id"],
        "batch_id": tanker["batch_id"],
        "latitude": round(lat, 4),
        "longitude": round(lon, 4),
        "reefer_temp_c": tanker["reefer_temp_c"],
        "ambient_temp_c": tanker["ambient_temp_c"],
        "compressor_rpm": 3180 if tanker["reefer_temp_c"] > 4.3 else (0 if door_status == "OPEN" else 2380),
        "door_status": door_status,
        "battery_voltage": round(random.uniform(23.8, 24.5), 1),
    }


def check_backend_health(base_url: str) -> bool:
    health_url = f"{base_url.rstrip('/')}/api/health"
    try:
        if HAS_HTTPX:
            res = httpx.get(health_url, timeout=3.0)
            if res.status_code == 200:
                info = res.json()
                print(f"[CONNECTED] FastAPI Backend Online | DB Batches: {info['counts']['milk_batch_records']}")
                return True
        else:
            with urllib.request.urlopen(health_url, timeout=3.0) as resp:
                if resp.status == 200:
                    info = json.loads(resp.read().decode("utf-8"))
                    print(f"[CONNECTED] FastAPI Backend Online | DB Batches: {info['counts']['milk_batch_records']}")
                    return True
    except Exception as exc:
        print(f"[WARNING] Cannot reach FastAPI backend at {health_url} ({exc}).")
        print("          Start server first: uvicorn app.main:app --host 0.0.0.0 --port 8000")
    return False


def post_packet_urllib(endpoint: str, payload: dict) -> dict:
    req = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=5.0) as resp:
        return json.loads(resp.read().decode("utf-8"))


def run_simulator(
    base_url: str,
    interval: float,
    forced_fault_tanker: str | None,
    max_packets: int | None,
    dry_run: bool,
):
    endpoint = f"{base_url.rstrip('/')}/api/iot/telemetry"
    print("=" * 78)
    print("SMART DAIRY COLD-CHAIN IoT TELEMETRY SIMULATOR")
    print(f"Target Endpoint : {endpoint if not dry_run else 'DRY-RUN (Local Console Only)'}")
    print(f"Stream Interval : {interval}s | Forced Fault Tanker: {forced_fault_tanker or 'None (Normal Mode)'}")
    print("=" * 78)

    if not dry_run:
        check_backend_health(base_url)

    packet_num = 0
    client = httpx.Client(timeout=5.0) if (HAS_HTTPX and not dry_run) else None

    try:
        while max_packets is None or packet_num < max_packets:
            tanker = FLEET_STATE[packet_num % len(FLEET_STATE)]
            payload = step_tanker_physics(tanker, forced_fault_tanker)
            packet_num += 1
            now_str = datetime.now().strftime("%H:%M:%S")

            if dry_run:
                breach = payload["reefer_temp_c"] > 4.5 or (
                    payload["door_status"] == "OPEN" and payload["ambient_temp_c"] > 32.0
                )
                status_tag = "BREACH (>4.5C / OPEN)" if breach else "OK"
                print(
                    f"[{now_str}] #{packet_num:04d} | {payload['tanker_id']} | "
                    f"Reefer: {payload['reefer_temp_c']:4.2f}°C | Ambient: {payload['ambient_temp_c']:4.1f}°C | "
                    f"Door: {payload['door_status']:6s} | {status_tag}"
                )
            else:
                try:
                    if client is not None:
                        response = client.post(endpoint, json=payload)
                        response.raise_for_status()
                        data = response.json()
                    else:
                        data = post_packet_urllib(endpoint, payload)

                    if data.get("alert_triggered"):
                        status_tag = f"ALERT SAVED -> {data['alert_message']}"
                    else:
                        status_tag = f"DB Row #{data['id']} OK"

                    print(
                        f"[{now_str}] #{packet_num:04d} | {payload['tanker_id']} | "
                        f"Reefer: {payload['reefer_temp_c']:4.2f}°C | Ambient: {payload['ambient_temp_c']:4.1f}°C | "
                        f"Door: {payload['door_status']:6s} | {status_tag}"
                    )
                except Exception as exc:
                    print(f"[{now_str}] [RETRYING] Backend unreachable ({exc})...")

            time.sleep(interval)
    except KeyboardInterrupt:
        print(f"\n[STOPPED] Simulator halted after {packet_num} telemetry packets.")
    finally:
        if client is not None:
            client.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mock IoT Cold-Chain Telemetry Streamer")
    parser.add_argument("--url", default="http://127.0.0.1:8000", help="FastAPI backend base URL")
    parser.add_argument("--interval", type=float, default=2.0, help="Seconds between telemetry packets")
    parser.add_argument("--fault-tanker", default=None, help="Force continuous thermal breach on a tanker (e.g. TKR-07)")
    parser.add_argument("--count", type=int, default=None, help="Stop after N packets (default: infinite)")
    parser.add_argument("--dry-run", action="store_true", help="Simulate locally without calling FastAPI")
    args = parser.parse_args()

    run_simulator(
        base_url=args.url,
        interval=args.interval,
        forced_fault_tanker=args.fault_tanker,
        max_packets=args.count,
        dry_run=args.dry_run,
    )
