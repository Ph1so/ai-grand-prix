## AI Grand Prix Virtual Qualifier Technical Specification 

Document ID: VADR-TS-003 Issue: 00.03 Date: 2026-06-24 

**==> picture [267 x 89] intentionally omitted <==**

DOCUMENT ID: VADR-TS-003 

AI GRAND PRIX — TECHNICAL SPECIFICATION 

ISSUE: 00.03 

## **Table Of Contents** 

|1.|Document Control|3|
|---|---|---|
|_1.1_|_Revision History_|_3_|
|_1.2_|_Audience_|_3_|
|2.|Purpose and Scope|4|
|_2.1_|_Purpose_|_4_|
|_2.2_|_Scope_|_4_|
|_2.3_|_Out of Scope_|_4_|
|3.|Simulation Environment|5|
|_3.1_|_General Environment_|_5_|
|_3.2_|_Physical Simulation Model_|_5_|
|_3.3_|_Spatial Reference Model_|_5_|
|_3.4_|_Visual Environment_|_5_|
|_3.5_|_Environmental Determinism_|_5_|
|_3.6_|_Drone chassis_|_6_|
|_3.7_|_Gate dimension_|_6_|
|_3.8_|_Coordinate Frames_|_7_|
|4.|Communication Protocol — MAVLink Interface|8|
|_4.1_|_Overview_|_8_|
|_4.2_|_Transport_|_8_|
|_4.3_|_Supported MAVLink Messages_|_8_|
|_4.4_|_Timing_|_8_|
|_4.5_|_Telemetry (deprecated)_|_8_|
|_4.6_|_Vision Stream_|_9_|
|_4.7_|_Software_-_in_-_the_-_Loop Bridge_|_9_|
|5.|Contestant Software Environment|10|
|_5.1_|_Runtime Environment_|_10_|
|_5.2_|_Client Responsibilities_|_10_|
|_5.3_|_Intended Control Architecture_|_10_|
|6.|Example Control Session|11|
|7.|Compliance|11|
|8.|Round One —Qualification Phase|11|
|_8.1_|_Objective_|_11_|
|_8.2_|_Course Structure_|_11_|
|_8.3_|_Maximum Run Duration_|_11_|
|9.|Round Two –Qualification Phase|11|
|_9.1_|_Objective_|_11_|
|_9.2_|_User Flow & Simulator Interface_|_12_|
|_9.3_|_Telemetry API Restrictions_|_12_|
|_9.4._|_Leaderboard, Scoring & Ranking Logic_|_12_|



DOCUMENT ID: VADR-TS-003 ISSUE: 00.03 

AI GRAND PRIX — TECHNICAL SPECIFICATION 

## **1. Document Control** 

## 1.1 Revision History 

|Issue|Date|Author|
|---|---|---|
|00.01|2026-03-09|KH|
|00.02|2026-05-04|NT|
|00.03|2026-06-24|NT|



## 1.2 Audience 

This specification is addressed to competition participants (“Teams”). It defines the technical interface and operational requirements required to develop autonomous control software for the Virtual AI Drone Race. 

DOCUMENT ID: VADR-TS-003 ISSUE: 00.03 

AI GRAND PRIX — TECHNICAL SPECIFICATION 

## **2. Purpose and Scope** 

## 2.1 Purpose 

This document defines the interface between contestant-developed control software and race simulator. The specification ensures that all teams interact with the simulator through a consistent communication and control interface. 

## 2.2 Scope 

- 

   - communication interfaces between contestant software and the simulator 

- control input requirements 

- telemetry interfaces 

- vision data interfaces 

- simulation timing and performance constraints 

- virtual race environment definition 

- qualification run requirements 

## 2.3 Out of Scope 

- internal simulator architecture 

- event operations 

- 

- contractual or commercial matters 

DOCUMENT ID: VADR-TS-003 ISSUE: 00.03 

AI GRAND PRIX — TECHNICAL SPECIFICATION 

## **3. Simulation Environment** 

## 3.1 General Environment 

- - The race takes place within a high fidelity real time physics simulator. 

- start gate 

- sequential race gates 

- finish gate 

- vertical and horizontal obstacles 

- boundary elements 

- terrain and environmental structures 

Gates will be visually distinctive to the environment, but consistent throughout the Virtual Qualifier 1 track. 

## 3.2 Physical Simulation Model 

- The simulator implements a rigid body drone flight model including thrust generation, aerodynamic drag, gravity, and collision physics. 

Physics update frequency: 120 Hz. 

## 3.3 Spatial Reference Model 

The simulator uses a local Cartesian coordinate system internally. No geographic coordinates are provided. GPS simulation is not available and absolute global position is not exposed. 

## 3.4 Visual Environment 

- - The simulator provides a forward facing first person camera and visual environment including gates, course guidance structures, static scene objects, and dynamic lighting. 

## 3.5 Environmental Determinism 

- course geometry is identical for all participants 

- • physics parameters are identical 

- environmental conditions are deterministic 

DOCUMENT ID: VADR-TS-003 

AI GRAND PRIX — TECHNICAL SPECIFICATION 

ISSUE: 00.03 

## 3.6 Drone chassis 

- Width: 280mm 

- Length: 280mm 

- Height: 160mm 

## 3.7 Gate dimension 

## **Gate boundaries** 

- Width: 2700mm 

- Height: 2700mm 

- Depth:  260mm 

## **Gate inner square boundaries:** 

- Width: 1500mm 

- Height: 1500mm 

- Depth:  260mm 

DOCUMENT ID: VADR-TS-003 ISSUE: 00.03 

AI GRAND PRIX — TECHNICAL SPECIFICATION 

## 3.8 Coordinate Frames 

The Mavlink2 coordinate convention is NED 

|||
|---|---|
|**Frame Constant**|**Description**|
|||
|MAV_FRAME_LOCAL_NED|The origin (0,0,0) is a fixed physical point on the ground (usually where<br>the drone armed).|
|MAV_FRAME_BODY_NED|The origin is the vehicle itself. X points forward, Y points right, Z points<br>down.|



## **Body to Camera** 

The camera and the body frame the same origin. The camera is tilted upwards by 20° upwards. Be aware that all coordinates are NED and you might need to rotate the camera frame into the camera coordinate convention of your specific image processing library. 

## **Body to IMU** 

The body to imu transformation is the identity map. 

## **Camera intrinsics** 

We use a standard pinhole camera model without lens distortion. 

- Image resolution: 640px x 360 px 

- [cx,cy] = [320px,180px] 

- [fx,fy] =[320,320] 

- VFoV= 90° 

AI GRAND PRIX — TECHNICAL SPECIFICATION 

DOCUMENT ID: VADR-TS-003 ISSUE: 00.03 

## **4. Communication Protocol — MAVLink Interface** 

## 4.1 Overview 

simulator communicates with contestant control software using  https://github.com/mavlink/c_library_v2 through MAVSDK-compatible interfaces. 

## 4.2 Transport 

Supported transport: UDP 

## 4.3 Supported MAVLink Messages 

Please refer to the MAVLINK 2 documentation for the definition of  messages https://mavlink.io/en/guide/mavlink_2.html 

|Message|Direction|Purpose|
|---|---|---|
|HEARTBEAT|Simulator → Client|Connection status|
|ATTITUDE|Simulator → Client|Vehicle attitude|
|HIGHRES_IMU|Simulator → Client|Vehicle status|
|SET_POSITION_TARGET_LOCAL_NED|Client → Simulator|Control interface|
|SET_ATTITUDE_TARGET|Client → Simulator|Control interface|
|TIMESYNC|Simulator → Client|Timing|
|HIGHRES_IMU|Simulator → Client|Measurements|



## 4.4 Timing 

Physics simulation rate: 120 Hz Command rate <100Hz Minimum heartbeat rate: 2 Hz 

## 4.5 Telemetry (deprecated) 

- ~~vehicle attitude~~ 

- ~~orientation~~ 

- ~~linear velocities~~ 

- 

- ~~system status flags~~ 

DOCUMENT ID: VADR-TS-003 ISSUE: 00.03 

AI GRAND PRIX — TECHNICAL SPECIFICATION 

## 4.6 Vision Stream 

Frequency of the camera stream is 30 HZ with a resolution of 640 by 360 pixels. 

## **Transport Summary** 

- **Protocol:** UDP 

- **Port:** 5600 (Default) 

- **Byte Order:** Little-Endian (<) 

- **Header Size:** 24 Bytes 

## **Packet Structure** 

Each packet consists of a fixed-length **Metadata Header** immediately followed by a variable-length **Binary Payload** . 

|||||
|---|---|---|---|
|**Field**|**Type**|**Size**|**Description**|
|||||
|**frame_id**|uint32|4B|Unique sequence ID for the image frame.|
|**chunk_id**|uint16|2B|The index of this packet within the frame (0<br>tototal_chunks - 1).|
|**total_chunks**|uint16|2B|Total number of packets required to<br>complete this frame.|
|**jpeg_size**|uint32|4B|Total size of the final reconstructed JPEG file<br>in bytes.|
|**payload_size**|uint32|4B|Size of the JPEG data slice contained in_this_<br>packet.|
|**sim_time_ns**|uint64|8B|Simulation epoch timestamp in nanoseconds.|



## - - - 4.7 Software in the Loop Bridge 

- The simulator provides a low latency UDP SITL bridge enabling external AI controllers to exchange telemetry and control commands. 

DOCUMENT ID: VADR-TS-003 ISSUE: 00.03 

AI GRAND PRIX — TECHNICAL SPECIFICATION 

## **5. Contestant Software Environment** 

## 5.1 Runtime Environment 

- Participants may assume a Python based runtime environment. Python 3.14.2 is known to operate correctly. Participants are allowed to choose other environments 

The DCL Simulator software runs on windows 11 with a standard PC and a descent GPU with 8GB VRAM 

Currently we do not support Linux OS. 

## 5.2 Client Responsibilities 

- establish MAVLink communication 

- maintain heartbeat messages 

- send control commands 

- process telemetry data 

- process vision stream data 

## 5.3 Intended Control Architecture 

Typical conceptual control pipeline: 

Vision + Telemetry → Perception → Planning → Control → Pilot Commands → Stabilized Controller 

DOCUMENT ID: VADR-TS-003 ISSUE: 00.03 

AI GRAND PRIX — TECHNICAL SPECIFICATION 

## **6. Example Control Session** 

- Client initializes MAVSDK 

- Client connects to simulator endpoint 

- Simulator transmits HEARTBEAT 

- Client streams control commands 

- Simulator applies commands 

- Telemetry and vision streams returned 

## **7. Compliance** 

Participants must ensure their implementation conforms to this specification. Specifically, human interaction during the flight which the participants submit as a timed run is grounds for immediate disqualification. 

## **8. Round One — Qualification Phase** 

## 8.1 Objective 

Round One verifies that contestant software can successfully navigate the racecourse. 

## 8.2 Course Structure 

- start gate 

- intermediate gates 

- finish gate 

## 8.3 Maximum Run Duration 

Maximum run duration: 8 minutes. 

## **9. Round Two – Qualification Phase** 

## 9.1 Objective 

Objective: The core objective of Phase 2 is to qualify for the Physical Qualifier Event in September. 

DOCUMENT ID: VADR-TS-003 ISSUE: 00.03 

AI GRAND PRIX — TECHNICAL SPECIFICATION 

## 9.2 User Flow & Simulator Interface 

To help you adapt to these changes, teams can now choose between two distinct flight modes: 

   - Training Flights: These flights allow you to test your algorithms and state estimation pipelines freely. They are not counted toward the official leaderboard. 

   - Competitive Flights: These are official, time-tracked runs that count directly toward your qualification standing. 

- **Event Selection** : The simulator interface will present two distinct, manually selectable event blocks to the user: 

      - Training 

      - Qualification (VQ2) - runs count for qualification for Physical Qualifier 

- **Execution:** The contestant must select the desired event block to launch the respective simulator mode. 

⚠ Important Note on Code Integrity: Once you submit a time-tracked race for qualification, DCL reserves the right to review your codebase. If the Race Director suspects any form of cheating or manipulation of the simulator constraints, a formal code audit will be triggered. 

## 9.3 Telemetry API Restrictions 

To ensure competitive integrity during the qualification phase, certain direct data streams from the simulator API are restricted, and strict track validation is enforced: 

- **Blocked Telemetry Messages:** Starting in Phase 2, the following interfaces/messages will no longer be available directly from the simulator API: 

   - ATTITUDE 

   - LOCAL_POSITION_NED 

   - ODOMETRY 

   - GATE_INFO 

## 9.4. Leaderboard, Scoring & Ranking Logic 

Ranking Criteria: Leaderboard (Qualification Event) ranking is strictly determined based on the timing results of valid, completed runs. 

- Faster times rank higher. 

- Attempts: Contestants are permitted an unlimited number of attempts to set their fastest time. 

- Team Scoring: Leaderboard scoring is team-centric. The best single timing result from a team dictates that team's overall rank, regardless of which individual team member achieved the time. 

