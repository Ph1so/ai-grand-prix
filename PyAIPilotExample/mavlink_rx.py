import csv
import json
import os
import struct
import time
import threading

_LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
os.makedirs(_LOG_DIR, exist_ok=True)

import numpy as np
from pymavlink import mavutil

ENCAPSULATED_RACE_STATUS_MSG_ID = 1
ENCAPSULATED_TRACK_INFO_MSG_ID  = 2

class MAVLinkRX:

    def __init__(self, mavlink_connection, data):
        self.mavlink_conn = mavlink_connection
        self.data = data
        self.thread = None
        self.is_running = False

        self.track_chunks = {}
        self.expected_num_track_chunks = {}

        ts = time.strftime('%Y%m%d_%H%M%S')
        self.run_dir = os.path.join(_LOG_DIR, f"run_{ts}")
        os.makedirs(self.run_dir, exist_ok=True)

        log_stem = f"flight_log_{ts}"
        log_filename = os.path.join(self.run_dir, f"{log_stem}.csv")
        self._gates_filename = os.path.join(self.run_dir, f"{log_stem}_gates.json")
        self._log_file = open(log_filename, 'w', newline='', buffering=1)
        self._csv = csv.writer(self._log_file)
        self._csv.writerow(['wall_time_s', 'time_boot_ms',
                            'pos_x', 'pos_y', 'pos_z',
                            'vel_x', 'vel_y', 'vel_z', 'speed',
                            'roll', 'pitch', 'yaw',
                            'active_gate'])
        print(f"[log] writing to {log_filename}", flush=True)

    @classmethod
    def create_mavlink_rx(cls, mavlink_connection, data):
        rx = cls(mavlink_connection, data)
        rx.thread = threading.Thread(
            target=rx.mavlink_receive_loop,
            daemon = False
        )
        rx.is_running = True
        rx.thread.start()
        return rx

    def get_thread_for_join(self):
        self.is_running = False
        self._log_file.close()
        return self.thread

    def mavlink_receive_loop(self):
        """
        Continuously receive MAVLink messages without blocking.
        """
        while self.is_running:

            try:
                msg = self.mavlink_conn.recv_match(blocking=False)
            except ConnectionResetError:
                print('WARNING: ConnectionResetError was thrown. No longer listening to MAVLink port.')
                self._log_file.close()
                return

            if msg is None:
                time.sleep(0.001)
                continue

            msg_type = msg.get_type()

            if msg_type == "BAD_DATA":
                continue

            # --------------------------------------------------------------------------------------
            # HEARTBEAT
            # --------------------------------------------------------------------------------------
            if msg_type == "HEARTBEAT":
                self.on_heartbeat(msg)

            # --------------------------------------------------------------------------------------
            # TIMESYNC
            # --------------------------------------------------------------------------------------
            elif msg_type == "TIMESYNC":
                self.on_timesync(msg)

            # --------------------------------------------------------------------------------------
            # ATTITUDE
            # --------------------------------------------------------------------------------------
            elif msg_type == "ATTITUDE":
                self.on_attitude(msg)

            # --------------------------------------------------------------------------------------
            # LOCAL_POSITION_NED
            # --------------------------------------------------------------------------------------
            elif msg_type == "LOCAL_POSITION_NED":
                self.on_local_position_ned(msg)

            # --------------------------------------------------------------------------------------
            # ODOMETRY
            # --------------------------------------------------------------------------------------
            elif msg_type == "ODOMETRY":
                self.on_odometry(msg)

            # --------------------------------------------------------------------------------------
            # HIGHRES_IMU
            # --------------------------------------------------------------------------------------
            elif msg_type == "HIGHRES_IMU":
                self.on_highres_imu(msg)

            # --------------------------------------------------------------------------------------
            # ENCAPSULATED_DATA
            # --------------------------------------------------------------------------------------
            elif msg_type == "ENCAPSULATED_DATA":
                self.on_encapsulated_data(msg)

            # --------------------------------------------------------------------------------------
            # ACTUATOR_OUTPUT_STATUS
            # --------------------------------------------------------------------------------------
            elif msg_type == "ACTUATOR_OUTPUT_STATUS":
                self.on_actuator_output_status(msg)

            # --------------------------------------------------------------------------------------
            # COLLISION
            # --------------------------------------------------------------------------------------
            elif msg_type == "COLLISION":
                self.on_collision(msg)

            # --------------------------------------------------------------------------------------
            # DATA_TRANSMISSION_HANDSHAKE - Repurposed and used for upcoming 'Track Data' packets
            # --------------------------------------------------------------------------------------
            elif msg.get_type() == "DATA_TRANSMISSION_HANDSHAKE":
                track_data_transfer_id = msg.width
                self.track_chunks[track_data_transfer_id] = {}
                self.expected_num_track_chunks[track_data_transfer_id] = msg.packets

    def on_heartbeat(self, msg):
        armed = msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED

    def on_timesync(self, msg):
        request_time = msg.ts1
        response_time = msg.tc1

    def on_attitude(self, msg):
        self.data['attitude'] = (msg.roll, msg.pitch, msg.yaw)

    def on_local_position_ned(self, msg):
        self.data['pos'] = np.array([msg.x, msg.y, msg.z])
        self.data['vel'] = np.array([msg.vx, msg.vy, msg.vz])
        self.data['time_boot_ms'] = msg.time_boot_ms
        speed = float(np.linalg.norm(self.data['vel']))
        roll, pitch, yaw = self.data.get('attitude', (0.0, 0.0, 0.0))
        active_gate = self.data.get('race_status', {}).get('active_gate', -1)
        self._csv.writerow([
            time.time(), msg.time_boot_ms,
            msg.x, msg.y, msg.z,
            msg.vx, msg.vy, msg.vz, speed,
            roll, pitch, yaw,
            active_gate,
        ])

    def on_odometry(self, msg):
        pass

    def on_highres_imu(self, msg):
        pass

    def on_encapsulated_data(self, msg):
        if msg:
            raw_payload = bytes(msg.data)
            data_type = raw_payload[0]

            if int(data_type) == ENCAPSULATED_RACE_STATUS_MSG_ID:
                self.on_race_status(msg)
            elif int(data_type) == ENCAPSULATED_TRACK_INFO_MSG_ID:
                self.on_track_data_packet(msg)

    def on_race_status(self, msg):
        raw_payload = bytes(msg.data)
        data_type, sim_boot_time_ms, race_start_boot_time_ms, race_finish_time_ns, active_gate_index, last_gate_race_time = struct.unpack_from(
            "<BQqqIq", raw_payload)
        new_status = {
            'active_gate': active_gate_index,
            'race_started': race_start_boot_time_ms >= 0,
            'race_finished': race_finish_time_ns >= 0,
        }
        old_status = self.data.get('race_status', {})
        if new_status != old_status:
            print(f"[race] gate={active_gate_index} started={new_status['race_started']} finished={new_status['race_finished']}", flush=True)
        self.data['race_status'] = new_status

    def on_track_data_packet(self, msg):
        raw_payload = bytes(msg.data)
        # header:
        #   data_type - ID of this message
        #   transfer_id - ID of the group of packets this chunk belongs to
        data_type, transfer_id = struct.unpack_from("<BH", raw_payload)
        if transfer_id not in self.expected_num_track_chunks:
            return
        raw_payload = raw_payload[3:]
        self.track_chunks[transfer_id][msg.seqnr] = raw_payload
        if len(self.track_chunks[transfer_id]) == self.expected_num_track_chunks[transfer_id]:
            full_payload = bytes()
            for i in range(len(self.track_chunks[transfer_id])):
                full_payload = full_payload + self.track_chunks[transfer_id][i]
            del self.track_chunks[transfer_id]
            del self.expected_num_track_chunks[transfer_id]
            self.on_track_data(full_payload)

    def on_track_data(self, payload):
        num_gates, = struct.unpack_from("<H", payload)
        payload = payload[2:]
        gates = []
        for i in range(num_gates):
            gate_id, x, y, z, qw, qx, qy, qz, width, height = struct.unpack_from(
                "<Hfffffffff", payload)
            # Sim sends gate z as altitude-above-ground (z-up), drone telemetry is NED (z-down).
            # Negate z to convert to NED so both share the same frame.
            gates.append({'id': gate_id, 'pos': np.array([x, y, -z]), 'width': width, 'height': height,
                          'quat': [qw, qx, qy, qz]})
            payload = payload[38:]
        self.data['gates'] = sorted(gates, key=lambda g: g['id'])
        print(f"[track] received {num_gates} gates", flush=True)

        serialisable = [
            {'id': g['id'], 'pos': g['pos'].tolist(), 'width': g['width'], 'height': g['height'],
             'quat': g['quat']}
            for g in self.data['gates']
        ]
        with open(self._gates_filename, 'w') as f:
            json.dump(serialisable, f, indent=2)
        print(f"[log] gate map saved to {self._gates_filename}", flush=True)

    def on_actuator_output_status(self, msg):
        pass

    def on_collision(self, msg):
        # collision_id: 1001 = gate, 1002 = environment
        self.data['collision'] = {
            'id': msg.id,
            'threat_level': msg.threat_level,
            'impulse': msg.horizontal_minimum_delta,
        }
        print(f"[collision] id={msg.id} impulse={msg.horizontal_minimum_delta:.2f}", flush=True)