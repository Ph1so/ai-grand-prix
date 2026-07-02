#
# VQ2 Python client for the AI GP controller
#

import ctypes
import time

import numpy as np

from setup import setup_components

# Improve Windows timer resolution for accurate CONTROL_HZ sleep timing
ctypes.windll.winmm.timeBeginPeriod(1)

SIM_SERVER_UDP_IP   = "127.0.0.1"
SIM_SERVER_UDP_PORT = 14550

system_boot_ms = int(time.time() * 1000)

# VQ2: ATTITUDE and LOCAL_POSITION_NED are blocked.
# Pre-populate pos at takeoff origin and zero vel so the controller and
# gate_verifier have valid data from the first update cycle.
# integrated_yaw is written by the controller each cycle from IMU integration.
shared_data = {
    'pos':               np.array([0.0, 0.0, -0.5]),  # fixed — no NED available
    'vel':               np.zeros(3),
    'integrated_yaw':    0.0,
    'gyro_yaw_rate':     0.0,
    'highres_imu_time_us': 0,
    'race_status': {
        'active_gate':   0,
        'race_started':  False,
        'race_finished': False,
    },
    'time_boot_ms': 0,
}

components  = setup_components(shared_data, system_boot_ms, SIM_SERVER_UDP_IP, SIM_SERVER_UDP_PORT)
controller  = components['controller']
ts_loop     = components['ts_loop']
mavlink_rx  = components['mavlink_rx']
vision_rx   = components['vision_rx']

print("Arming drone...", flush=True)
controller.arm()
print("Starting control loop...", flush=True)
try:
    while True:
        controller.update()
except KeyboardInterrupt:
    print("\nInterrupted — shutting down.", flush=True)
finally:
    controller.close_log()

# Clean shutdown
ts_loop.get_thread_for_join().join(timeout=1.0)
mavlink_rx.get_thread_for_join().join(timeout=1.0)
vision_rx.get_thread_for_join().join(timeout=1.0)

ctypes.windll.winmm.timeEndPeriod(1)
print("Client exited!", flush=True)
