import os
import time

from pymavlink import mavutil
from timesync import TimeSync
from gate_verifier import GateVerifier
from mavlink_rx import MAVLinkRX
from controller import Controller

def setup_components(shared_data, system_boot_ms, server_ip, server_udp_port):
    # Create a timestamped run directory for all log files this session
    ts = time.strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs', f'run_{ts}')
    os.makedirs(run_dir, exist_ok=True)
    print(f"[setup] run directory: {run_dir}", flush=True)

    # MAVLink connection
    sim_conn = mavutil.mavlink_connection('udpin:%s:%s' % (server_ip, server_udp_port))
    print("Waiting for heartbeat...", flush=True)
    sim_conn.wait_heartbeat()
    print(f"Connected to system: {sim_conn.target_system}", flush=True)

    # MAVLink receiver thread
    print("Setting up MAVLink rx...", flush=True)
    mavlink_rx = MAVLinkRX.create_mavlink_rx(sim_conn, shared_data)

    # TimeSync thread — Step 1.3: must use create_timesync(), NOT the plain constructor
    print("Setting up Timesync loop...", flush=True)
    ts_loop = TimeSync.create_timesync(sim_conn, shared_data)

    # Vision + CV pipeline — GateVerifier replaces VisionRX stub
    vision_rx = GateVerifier(shared_data, run_dir=run_dir)

    # Controller
    controller = Controller(sim_conn, shared_data, system_boot_ms, run_dir=run_dir)

    return {
        'vision_rx': vision_rx,
        'mavlink_rx': mavlink_rx,
        'ts_loop': ts_loop,
        'sim_conn': sim_conn,
        'controller': controller,
        'run_dir': run_dir,
    }
