"""
Block-body pickup + block flatbed trailer sway experiment.
Geometry knobs (trailer subtree is REGENERATED from these before compile,
because MuJoCo bakes inertia at compile time):
  TONGUE_LEN   - hitch-bar length, ball to front edge of the deck [m].
                 Axle sits at deck center: x_axle = -(TONGUE_LEN + 1.22).
  CARGO_OFFSET - payload position RELATIVE TO THE AXLE [m].
                   > 0 : ahead of axle, toward truck -> tongue weight, stable
                   < 0 : behind axle -> negative tongue weight, sway-prone
  CARGO_MASS   - payload mass [kg].
The script solves the static moment balance about the hitch ball to set the
trailer spring preload and prints the resulting tongue load. A negative
tongue load is the classic real-world predictor of divergent sway.
Phases: settle -> spinup -> disturb (swerve or gust) -> record (CSV).
Body/joint/actuator names match the original scripts (car, trailer, hitch,
drive_left, drive_right), so existing tracking code only needs xml_path.
"""

import csv
import math
import os
import time

import mujoco
import mujoco.viewer
import numpy as np

XML_PATH = "block_pickup_trailer.xml"

# ---------------- geometry knobs ----------------
TONGUE_LEN = 1.50        # m, ball -> deck front edge
CARGO_OFFSET = 0.10      # m relative to axle (+ = ahead/stable, - = behind/sway)
CARGO_MASS = 350.0       # kg
CARGO_HALF = (0.50, 0.60, 0.25)   # payload half-sizes (x, y, z)

TRAILER_TIRE_MU = 0.55   # sliding friction of trailer tires
HITCH_DAMPING = 0.0      # N*m*s/rad on the ball coupler

# ---------------- experiment knobs ----------------
SPEED_CTRL = 58.0        # rad/s wheel target (58 -> ~23.5 m/s / ~52 mph)
DISTURBANCE = "swerve"   # "swerve" or "gust"
STEER_FLICK = 0.15       # rad, swerve steering amplitude (each phase 0.4 s)
GUST_N = 8000.0          # N, only if DISTURBANCE == "gust"
GUST_TIME = 0.5          # s

COAST = True             # release drive after disturbance (lift off throttle)
SETTLE_TIME = 3.0        # s
SPINUP_TIME = 8.0        # s
MAX_RECORD = 40.0        # s
REALTIME = True
# ---------------------------------------------------

# fixed trailer constants (match the XML comments)
TONGUE_MASS = 45.0
BED_MASS = 250.0
BED_HALF_X, BED_HALF_Y, BED_HALF_Z = 1.22, 0.76, 0.06
BED_Z = 0.10                     # deck center height in trailer frame
TRAILER_K = 35000.0              # spring stiffness per wheel [N/m]
G = 9.81

# ---------------- pivot (towed oscillation) mode ----------------
PIVOT_MODE = True       # True: controller OFF, car towed by a pivot point
                         # fixed to the lane centerline (x-motion only).
                         # Car may yaw and roll about the pivot -> undamped
                         # oscillation to measure raw trailer sway response.
PIVOT_AHEAD = 1.5        # m in front of the front bumper
CAR_HALF_LEN = 2.95      # truck block half length (front bumper x)
PIVOT_SPEED = SPEED_CTRL * WHEEL_RADIUS   # m/s tow speed


def trailer_xml():
    """Generate the trailer subtree for the current knobs."""
    L = TONGUE_LEN
    ax = -(L + BED_HALF_X)                      # axle x = deck center
    cx = ax + CARGO_OFFSET                      # cargo x
    cz = BED_Z + BED_HALF_Z + CARGO_HALF[2]     # cargo sits on the deck

    # keep the payload fully on the deck
    max_off = BED_HALF_X - CARGO_HALF[0]
    if abs(CARGO_OFFSET) > max_off:
        raise ValueError(f"CARGO_OFFSET must be within +-{max_off:.2f} m "
                         f"to keep the payload on the deck")

    # static moment balance about the ball: axle load and tongue load
    total = TONGUE_MASS + BED_MASS + CARGO_MASS
    moment = TONGUE_MASS * (-L / 2) + BED_MASS * ax + CARGO_MASS * cx
    axle_load = G * moment / ax                 # N, total vertical on axle
    tongue_load = G * total - axle_load         # N, vertical on the ball
    springref = -(axle_load / 2) / TRAILER_K    # preload per wheel
    springref = max(min(springref, 0.0), -0.095)

    print(f"Axle at x = {ax:.2f} m | cargo at x = {cx:.2f} m "
          f"({CARGO_OFFSET:+.2f} m from axle)")
    print(f"Static tongue load = {tongue_load / G:.0f} kg "
          f"({100 * tongue_load / (G * total):.0f}% of trailer weight)"
          + ("  << NEGATIVE: sway-prone!" if tongue_load < 0 else ""))

    return f"""<!-- TRAILER_START -->
      <body name="trailer" pos="-3.10 0 -0.15">
        <joint name="hitch" type="ball" limited="true" range="0 45" damping="{HITCH_DAMPING}"/>
        <geom name="trailer_tongue" type="box" size="{L/2:.3f} 0.05 0.04" pos="{-L/2:.3f} 0 0" mass="{TONGUE_MASS}" material="trailer_mat"/>
        <geom name="trailer_bed" type="box" size="{BED_HALF_X} {BED_HALF_Y} {BED_HALF_Z}" pos="{ax:.3f} 0 {BED_Z}" mass="{BED_MASS}" material="trailer_mat"/>
        <geom name="cargo" type="box" size="{CARGO_HALF[0]} {CARGO_HALF[1]} {CARGO_HALF[2]}" pos="{cx:.3f} 0 {cz:.3f}" mass="{CARGO_MASS}" material="cargo_mat"/>
        <body name="tl_susp" pos="{ax:.3f} 0.88 -0.105">
          <joint name="tl_susp" type="slide" axis="0 0 1" range="-0.10 0.10" stiffness="{TRAILER_K:.0f}" damping="900" springref="{springref:.4f}"/>
          <geom type="sphere" size="0.05" mass="12" material="trailer_mat" contype="0" conaffinity="0"/>
          <body name="tl_wheel">
            <joint name="tl_hinge" class="spin"/>
            <geom class="tire" size="0.345 0.1025" zaxis="0 1 0" mass="25" friction="{TRAILER_TIRE_MU} 0.005 0.0001"/>
          </body>
        </body>
        <body name="tr_susp" pos="{ax:.3f} -0.88 -0.105">
          <joint name="tr_susp" type="slide" axis="0 0 1" range="-0.10 0.10" stiffness="{TRAILER_K:.0f}" damping="900" springref="{springref:.4f}"/>
          <geom type="sphere" size="0.05" mass="12" material="trailer_mat" contype="0" conaffinity="0"/>
          <body name="tr_wheel">
            <joint name="tr_hinge" class="spin"/>
            <geom class="tire" size="0.345 0.1025" zaxis="0 1 0" mass="25" friction="{TRAILER_TIRE_MU} 0.005 0.0001"/>
          </body>
        </body>
      </body>
      <!-- TRAILER_END -->"""


def build_model():
    xml = open(XML_PATH).read()
    start = xml.index("<!-- TRAILER_START -->")
    end = xml.index("<!-- TRAILER_END -->") + len("<!-- TRAILER_END -->")
    xml = xml[:start] + trailer_xml() + xml[end:]

    if PIVOT_MODE:
        px = CAR_HALF_LEN + PIVOT_AHEAD   # pivot x in car frame
        # wrap the car in a pivot body: slide along lane centerline only,
        # car free to yaw and roll about the pivot point (both undamped)
        rs = xml.index("<!-- CAR_ROOT_START -->")
        re = xml.index("<!-- CAR_ROOT_END -->") + len("<!-- CAR_ROOT_END -->")
        pivot_root = f"""<body name="pivot" pos="{px} 0 0.60">
      <joint name="pivot_x" type="slide" axis="1 0 0"/>
      <geom name="pivot_marker" type="sphere" size="0.08" mass="1"
            contype="0" conaffinity="0" rgba="0 1 0 0.8"/>
      <body name="car" pos="{-px} 0 0">
        <joint name="car_yaw"  type="hinge" axis="0 0 1" pos="{px} 0 0" damping="0"/>
        <joint name="car_roll" type="hinge" axis="1 0 0" pos="{px} 0 0" damping="0"/>"""
        xml = xml[:rs] + pivot_root + xml[re:]
        # close the extra pivot body
        xml = xml.replace("<!-- CAR_END -->", "</body>\n    <!-- CAR_END -->")
        # actuator that pulls the pivot (this is what tows the car)
        xml = xml.replace("</actuator>",
            '  <velocity name="pivot_drive" joint="pivot_x" kv="50000" '
            'ctrlrange="0 30"/>\n  </actuator>')

    return mujoco.MjModel.from_xml_string(xml)


def quat_to_euler(w, x, y, z):
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    sinp = 2 * (w * y - z * x)
    pitch = math.copysign(math.pi / 2, sinp) if abs(sinp) >= 1 else math.asin(sinp)
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return math.degrees(roll), math.degrees(pitch), math.degrees(yaw)


def run_simulation():
    model = build_model()
    data = mujoco.MjData(model)

    dl = model.actuator("drive_left").id
    dr = model.actuator("drive_right").id
    sl = model.actuator("steer_left").id
    sr = model.actuator("steer_right").id
    trailer_id = model.body("trailer").id
    hitch_adr = model.joint("hitch").qposadr[0]
    dt = model.opt.timestep
    trailer_rear_x = -(TONGUE_LEN + 2 * BED_HALF_X)   # gust application point
    pv = None
    if PIVOT_MODE:
        pv = model.actuator("pivot_drive").id
        # freewheel the drive wheels: the pivot pulls the car, not the wheels
        for a in (dl, dr):
            model.actuator_gainprm[a, 0] = 0
            model.actuator_biasprm[a, 2] = 0
    os.makedirs("csvs", exist_ok=True)
    csv_filename = (
        f"csvs/Block_Sway_{DISTURBANCE}_v{SPEED_CTRL:.0f}"
        f"_tongue{TONGUE_LEN:.2f}_cargo{CARGO_MASS:.0f}kg"
        f"_off{CARGO_OFFSET:+.2f}_mu{TRAILER_TIRE_MU}.csv"
    )
    headers = ["time", "rel_trailer_x", "rel_trailer_y", "rel_trailer_z",
               "hitch_roll", "hitch_pitch", "hitch_yaw", "car_speed"]

    swerve_steps = int(0.4 / dt)
    if DISTURBANCE == "swerve":
        disturb = [("swerve_r", swerve_steps), ("swerve_l", swerve_steps),
                   ("swerve_c", swerve_steps)]
    else:
        disturb = [("gust", int(GUST_TIME / dt))]
    schedule = ([("settle", int(SETTLE_TIME / dt)),
                 ("spinup", int(SPINUP_TIME / dt))]
                + disturb
                + [("record", int(MAX_RECORD / dt))])

    with open(csv_filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)

        with mujoco.viewer.launch_passive(model, data) as viewer:
            phase_idx, step_in_phase = 0, 0
            coasting = False
            peak_yaw = 0.0
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            viewer.cam.trackbodyid = model.body("car").id

            while viewer.is_running() and phase_idx < len(schedule):
                step_start = time.time()
                phase = schedule[phase_idx][0]

                if PIVOT_MODE:
                    # controller OFF; pivot tows the car along the centerline
                    if phase != "settle":
                        data.ctrl[pv] = PIVOT_SPEED
                    steer = {"swerve_r": magnitude,
                             "swerve_l": -magnitude}.get(phase, 0.0)
                    data.ctrl[sl] = steer
                    data.ctrl[sr] = steer
                elif phase == "record":
                    steer, speed = control_function(
                        read_sensors(model, data), dt, ctrl_state)
                    steer = max(-MAX_STEER, min(MAX_STEER, steer))
                    speed = max(0.0, min(60.0, speed))
                    data.ctrl[sl] = steer
                    data.ctrl[sr] = steer
                    data.ctrl[dl] = speed
                    data.ctrl[dr] = speed
                else:
                    if phase != "settle":
                        data.ctrl[dl] = SPEED_CTRL
                        data.ctrl[dr] = SPEED_CTRL
                    steer = {"swerve_r": magnitude,
                             "swerve_l": -magnitude}.get(phase, 0.0)
                    data.ctrl[sl] = steer
                    data.ctrl[sr] = steer

                if phase == "gust":
                    rear = data.body("trailer").xpos + data.body(
                        "trailer").xmat.reshape(3, 3) @ np.array(
                        [trailer_rear_x, 0, 0])
                    qfrc = np.zeros(model.nv)
                    mujoco.mj_applyFT(model, data, np.array([0, GUST_N, 0]),
                                      np.zeros(3), rear, trailer_id, qfrc)
                    data.qfrc_applied[:] = qfrc
                else:
                    data.qfrc_applied[:] = 0

                mujoco.mj_step(model, data)

                if phase == "record":
                    car_pos = data.body("car").xipos
                    trailer_pos = data.body("trailer").xipos
                    car_R = data.body("car").xmat.reshape(3, 3)
                    rel = car_R.T @ (trailer_pos - car_pos)
                    q = data.qpos[hitch_adr:hitch_adr + 4]
                    roll, pitch, yaw = quat_to_euler(*q)
                    peak_yaw = max(peak_yaw, abs(yaw))
                    speed = float(np.linalg.norm(data.body("car").cvel[3:6]))
                    writer.writerow([data.time, rel[0], rel[1], rel[2],
                                     roll, pitch, yaw, speed])

                step_in_phase += 1
                if step_in_phase >= schedule[phase_idx][1]:
                    phase_idx += 1
                    step_in_phase = 0
                    if phase_idx < len(schedule):
                        if schedule[phase_idx][0] == "record":
                            print(f"Disturbance done at t={data.time:.2f}s. "
                                  f"Recording...")
                            if COAST:
                                coasting = True
                                data.ctrl[dl] = 0
                                data.ctrl[dr] = 0
                                # zero the servo gain: true freewheel coast
                                model.actuator_gainprm[dl, 0] = 0
                                model.actuator_biasprm[dl, 2] = 0
                                model.actuator_gainprm[dr, 0] = 0
                                model.actuator_biasprm[dr, 2] = 0

                viewer.sync()
                if REALTIME:
                    leftover = dt - (time.time() - step_start)
                    if leftover > 0:
                        time.sleep(leftover)

    final_speed = float(np.linalg.norm(data.body("car").cvel[3:6]))
    outcome = ("LOST CONTROL (sway divergence / jackknife)"
               if final_speed < 5 else "recovered")
    print(f"Outcome: {outcome}. Peak |hitch yaw| = {peak_yaw:.1f} deg, "
          f"final speed = {final_speed:.1f} m/s")
    print(f"CSV saved at {csv_filename}")


if __name__ == "__main__":
    run_simulation()