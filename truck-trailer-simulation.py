"""
Block-body pickup + flatbed trailer sway experiment (no suspension).

- Lane: 3.66 m wide, centered on y=0, road runs along +x. Lane center = y=0.
- IMUs on truck and trailer (accel, gyro, orientation quat) + hitch sensor
  (trailer angle relative to truck).
- control_function() below is YOUR hook: it receives only the sensor data
  and returns (steer, speed). Preloaded with a human-driver model
  (preview steering, reaction delay, rate limit) that always corrects
  toward the middle of the lane but is blind to the trailer.

MODES
- PIVOT_MODE: controller OFF. A single massless "tow point" is attached to
  the center of the truck's front axle by a ball-type connect constraint:
  the truck is completely free in pitch, yaw, and roll about the point.
  The point moves freely along z, is driven forward along x at PIVOT_SPEED,
  and moves side to side against a proportional spring (PIVOT_SPRING_K)
  plus damper (PIVOT_SPRING_C) that always pulls it back to lane center.
- PLANAR_MODE: kills ALL vertical-axis dynamics. Truck root becomes
  x/y/yaw only, the hitch becomes a yaw-only hinge, so there is no pitch,
  roll, or heave anywhere: zero weight transfer, constant normal forces
  (truck COM is auto-centered so all 4 truck wheels carry equal load),
  and one common friction coefficient on all six tires (PLANAR_TIRE_MU).
  Can be combined with PIVOT_MODE.

- N_RUNS simulations. Each run i uses:
    disturbance magnitude = DISTURB_START + i * DISTURB_STEP
    cargo offset          = CARGO_OFFSET  + i * CARGO_OFFSET_STEP
    cargo mass            = CARGO_MASS    + i * CARGO_MASS_STEP
  Swerve units: rad. Gust units: N.
- At the end, one figure per run: hitch angle, lateral hitch force on the
  truck, truck yaw vs road, truck lane offset, trailer tire grip (L/R),
  truck tire grip (all 4), front vs rear truck axle weight, and truck
  speed. Blue bands = trailer at max swing (hitting the hitch stop /
  truck); red bands = truck flipped sideways.
"""

import csv
import math
import os
import time

import matplotlib.pyplot as plt
import mujoco
import mujoco.viewer
import numpy as np

XML_PATH = "truck-trailer.xml"

# ---------------- geometry knobs ----------------
TONGUE_LEN = 1.0        # m, ball -> deck front edge
CARGO_OFFSET = 0.72      # m relative to axle (+ = ahead/stable, - = behind/sway)
CARGO_MASS = 850.0       # kg
CARGO_HALF = (0.50, 0.60, 0.25)

TRAILER_TIRE_MU = 0.55
HITCH_DAMPING = 0.0

# ---------------- experiment knobs ----------------
SPEED_CTRL = 58.0        # rad/s wheel target (~23.5 m/s)
DISTURBANCE = "swerve"   # "swerve" or "gust"

N_RUNS = 3               # how many simulations to run
DISTURB_START = 0.15     # first-run magnitude: rad (swerve) or N (gust)
DISTURB_STEP = 0.00      # added to the magnitude after every run
CARGO_OFFSET_STEP = -0.72  # m added to CARGO_OFFSET after every run
CARGO_MASS_STEP = 0.0    # kg added to CARGO_MASS after every run

GUST_TIME = 0.5          # s
SETTLE_TIME = 1.0        # s
SPINUP_TIME = 5.0        # s
MAX_RECORD = 15.0        # s
REALTIME = True
# ---------------------------------------------------

# fixed trailer constants (match the XML comments)
TONGUE_MASS = 45.0
BED_MASS = 250.0
BED_HALF_X, BED_HALF_Y, BED_HALF_Z = 1.22, 0.76, 0.06
BED_Z = 0.10
G = 9.81
WHEEL_RADIUS = 0.41      # truck tire radius, for speed dead reckoning
MAX_STEER = 0.61         # rad

HITCH_LIMIT_DEG = 45.0   # hitch joint range from the XML
HITCH_HIT_MARGIN = 1.0   # deg, within this of the limit counts as contact
FLIP_ROLL_DEG = 60.0     # deg of truck roll that counts as flipped sideways

# ---------------- pivot (towed oscillation) mode ----------------
PIVOT_MODE = True        # True: controller OFF, truck towed by a tow point
TOW_EYE = (1.85, 0.0, -0.19)   # car frame: center of the front axle
PIVOT_SPEED = SPEED_CTRL * WHEEL_RADIUS   # m/s tow speed (customizable)
PIVOT_SPRING_K = 11000.0  # N/m, lateral spring pulling the tow point to y=0
PIVOT_SPRING_C = 3500.0   # N*s/m, lateral damping on the tow point

# ---------------- planar / no-weight-shift mode ----------------
PLANAR_MODE = False      # True: no vertical-axis motion at all. No fore/aft
                         # or side-to-side weight shift, constant normal
                         # forces, same friction coefficient on every tire.
PLANAR_TIRE_MU = 1.0     # friction coefficient applied to ALL six tires

TRUCK_TIRES = ["fl_tire", "fr_tire", "rl_tire", "rr_tire"]
TRAILER_TIRES = ["tl_tire", "tr_tire"]

SENSOR_NAMES = {
    "car_accel": "car_imu_accel",
    "car_gyro": "car_imu_gyro",
    "car_quat": "car_imu_quat",
    "trailer_accel": "trailer_imu_accel",
    "trailer_gyro": "trailer_imu_gyro",
    "trailer_quat": "trailer_imu_quat",
    "hitch_quat": "hitch_angle",
}

# ---------------- human driver parameters ----------------
WHEELBASE = 3.70         # m, truck wheelbase (pure-pursuit geometry)
LOOKAHEAD_TIME = 1.5     # s, how far down the road the driver looks
LOOKAHEAD_MIN = 8.0      # m, minimum preview distance at low speed
REACTION_DELAY = 0.25    # s, perception + neuromuscular delay
STEER_RATE_MAX = 0.7     # rad/s max front-wheel rate (hand-over-hand limit)
Y_DEADBAND = 0.05        # m, offsets smaller than this are ignored


# =====================================================================
# >>> CONTROLLER — EDIT THIS FUNCTION <<<
# (unchanged — never called in PIVOT_MODE)
# =====================================================================
def control_function(sensors, dt, state):
    # heading from the truck IMU
    w, x, y, z = sensors["car_quat"]
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

    if "v_est" not in state:
        state["v_est"] = SPEED_CTRL * WHEEL_RADIUS   # speed at handoff
        state["y_est"] = 0.0
        state["steer"] = 0.0
        state["cmd_buf"] = []                        # reaction-delay queue

    # dead-reckoned speed and lateral offset from lane center
    state["v_est"] += float(sensors["car_accel"][0]) * dt
    state["y_est"] += state["v_est"] * math.sin(yaw) * dt

    # driver ignores tiny offsets
    y_err = state["y_est"] if abs(state["y_est"]) > Y_DEADBAND else 0.0

    # aim at a point on the centerline, lookahead distance ahead
    ld = max(LOOKAHEAD_MIN, state["v_est"] * LOOKAHEAD_TIME)
    alpha = math.atan2(-y_err, ld) - yaw             # angle to preview point
    target = math.atan2(2 * WHEELBASE * math.sin(alpha), ld)  # pure pursuit
    target = max(-MAX_STEER, min(MAX_STEER, target))

    # reaction delay: act on the command from REACTION_DELAY seconds ago
    state["cmd_buf"].append(target)
    n_delay = max(1, int(REACTION_DELAY / dt))
    delayed = (state["cmd_buf"].pop(0)
               if len(state["cmd_buf"]) > n_delay else 0.0)

    # steering-rate limit: hands can only turn the wheel so fast
    max_step = STEER_RATE_MAX * dt
    delta = max(-max_step, min(max_step, delayed - state["steer"]))
    state["steer"] += delta

    return state["steer"], SPEED_CTRL
# =====================================================================
# >>> END CONTROLLER <<<
# =====================================================================


def trailer_xml(cargo_offset, cargo_mass):
    """Generate the (suspension-free) trailer subtree for the given cargo."""
    L = TONGUE_LEN
    ax = -(L + BED_HALF_X)
    cx = ax + cargo_offset
    cz = BED_Z + BED_HALF_Z + CARGO_HALF[2]

    max_off = BED_HALF_X - CARGO_HALF[0]
    if abs(cargo_offset) > max_off:
        raise ValueError(f"cargo offset must be within +-{max_off:.2f} m")

    total = TONGUE_MASS + BED_MASS + cargo_mass
    moment = TONGUE_MASS * (-L / 2) + BED_MASS * ax + cargo_mass * cx
    axle_load = G * moment / ax
    tongue_load = G * total - axle_load
    print(f"Axle at x = {ax:.2f} m | cargo at x = {cx:.2f} m "
          f"({cargo_offset:+.2f} m from axle), {cargo_mass:.0f} kg")
    print(f"Static tongue load = {tongue_load / G:.0f} kg "
          f"({100 * tongue_load / (G * total):.0f}% of trailer weight)"
          + ("  << NEGATIVE: sway-prone!" if tongue_load < 0 else ""))

    if PLANAR_MODE:
        hitch = (f'<joint name="hitch" type="hinge" axis="0 0 1" '
                 f'limited="true" range="-{HITCH_LIMIT_DEG:.0f} '
                 f'{HITCH_LIMIT_DEG:.0f}" damping="{HITCH_DAMPING}"/>')
        mu = PLANAR_TIRE_MU
    else:
        hitch = (f'<joint name="hitch" type="ball" limited="true" '
                 f'range="0 {HITCH_LIMIT_DEG:.0f}" '
                 f'damping="{HITCH_DAMPING}"/>')
        mu = TRAILER_TIRE_MU

    return f"""<!-- TRAILER_START -->
      <body name="trailer" pos="-3.10 0 -0.15">
        {hitch}
        <site name="hitch_force_site" pos="0 0 0" size="0.02" rgba="0 0 1 0.5"/>
        <site name="imu_trailer" pos="{ax:.3f} 0 0.10" size="0.03" rgba="1 0 0 0.5"/>
        <geom name="trailer_tongue" type="box" size="{L/2:.3f} 0.05 0.04" pos="{-L/2:.3f} 0 0" mass="{TONGUE_MASS}" material="trailer_mat"/>
        <geom name="trailer_bed" type="box" size="{BED_HALF_X} {BED_HALF_Y} {BED_HALF_Z}" pos="{ax:.3f} 0 {BED_Z}" mass="{BED_MASS}" material="trailer_mat"/>
        <geom name="cargo" type="box" size="{CARGO_HALF[0]} {CARGO_HALF[1]} {CARGO_HALF[2]}" pos="{cx:.3f} 0 {cz:.3f}" mass="{cargo_mass}" material="cargo_mat"/>
        <body name="tl_wheel" pos="{ax:.3f} 0.88 -0.105">
          <joint name="tl_hinge" class="spin"/>
          <geom name="tl_tire" class="tire" size="0.345 0.1025" zaxis="0 1 0" mass="25" friction="{mu} 0.005 0.0001"/>
        </body>
        <body name="tr_wheel" pos="{ax:.3f} -0.88 -0.105">
          <joint name="tr_hinge" class="spin"/>
          <geom name="tr_tire" class="tire" size="0.345 0.1025" zaxis="0 1 0" mass="25" friction="{mu} 0.005 0.0001"/>
        </body>
      </body>
      <!-- TRAILER_END -->"""


def build_model(cargo_offset, cargo_mass):
    xml = open(XML_PATH).read()
    start = xml.index("<!-- TRAILER_START -->")
    end = xml.index("<!-- TRAILER_END -->") + len("<!-- TRAILER_END -->")
    xml = xml[:start] + trailer_xml(cargo_offset, cargo_mass) + xml[end:]

    # force sensor at the hitch: lateral force the trailer puts on the truck
    xml = xml.replace(
        "</sensor>",
        '  <force name="hitch_force" site="hitch_force_site"/>\n  </sensor>')

    if PIVOT_MODE:
        # single tow point, starting exactly at the front-axle center
        # (tow eye). Slides: x = driven forward, y = spring+damper to lane
        # center, z = free. MuJoCo needs positive mass on jointed bodies;
        # 1 kg vs a 2.5 t truck is effectively massless.
        eye = (TOW_EYE[0], TOW_EYE[1], 0.60 + TOW_EYE[2])   # world at qpos0
        leader = f"""<body name="leader" pos="{eye[0]:.3f} {eye[1]:.3f} {eye[2]:.3f}">
      <joint name="leader_x" type="slide" axis="1 0 0"/>
      <joint name="leader_y" type="slide" axis="0 1 0" stiffness="{PIVOT_SPRING_K}" damping="{PIVOT_SPRING_C}"/>
      <joint name="leader_z" type="slide" axis="0 0 1"/>
      <geom name="leader_marker" type="sphere" size="0.06" mass="1"
            contype="0" conaffinity="0" rgba="0 1 0 0.8"/>
    </body>
    <!-- CAR_ROOT_START -->"""
        xml = xml.replace("<!-- CAR_ROOT_START -->", leader, 1)

        # ball-type attachment: tow point and front-axle center coincide,
        # truck rotation about the point is completely free (pitch/yaw/roll)
        eq = """<equality>
    <connect name="tow_ball" body1="leader" body2="car" anchor="0 0 0"/>
  </equality>
  <actuator>"""
        xml = xml.replace("<actuator>", eq, 1)
        xml = xml.replace("</actuator>",
            '  <velocity name="leader_drive" joint="leader_x" kv="50000" '
            'ctrlrange="0 40"/>\n  </actuator>')

    if PLANAR_MODE:
        # truck root: x / y / yaw only -> no heave, pitch, or roll anywhere
        rs = xml.index("<!-- CAR_ROOT_START -->")
        re_ = xml.index("<!-- CAR_ROOT_END -->") + len("<!-- CAR_ROOT_END -->")
        planar_root = """<body name="car" pos="0 0 0.60">
      <joint name="car_x" type="slide" axis="1 0 0"/>
      <joint name="car_y" type="slide" axis="0 1 0"/>
      <joint name="car_yaw" type="hinge" axis="0 0 1"/>"""
        xml = xml[:rs] + planar_root + xml[re_:]
        # hitch is a hinge in planar mode: swap the ball sensor for jointpos
        xml = xml.replace('<ballquat name="hitch_angle" joint="hitch"/>',
                          '<jointpos name="hitch_angle" joint="hitch"/>')
        # one common friction coefficient on the truck tires (the trailer
        # tires already got PLANAR_TIRE_MU in trailer_xml)
        xml = xml.replace('friction="1.3 0.005 0.0001"',
                          f'friction="{PLANAR_TIRE_MU} 0.005 0.0001"')
        # center the truck COM so all four truck wheels carry equal load
        xml = xml.replace('<inertial pos="0.26 0 0.10"',
                          '<inertial pos="0 0 0.10"')

    return mujoco.MjModel.from_xml_string(xml)


def quat_to_euler(w, x, y, z):
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    sinp = 2 * (w * y - z * x)
    pitch = math.copysign(math.pi / 2, sinp) if abs(sinp) >= 1 else math.asin(sinp)
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return math.degrees(roll), math.degrees(pitch), math.degrees(yaw)


def read_sensors(model, data):
    out = {}
    for key, name in SENSOR_NAMES.items():
        s = model.sensor(name)
        adr, dim = s.adr[0], s.dim[0]
        out[key] = data.sensordata[adr:adr + dim].copy()
    return out


def tire_normal_loads(model, data, tire_ids, floor_id):
    """Total ground contact normal force per tire geom [N].
    Grip capacity is proportional to mu * normal load."""
    loads = {gid: 0.0 for gid in tire_ids.values()}
    f = np.zeros(6)
    for i in range(data.ncon):
        c = data.contact[i]
        other = None
        if c.geom1 == floor_id:
            other = c.geom2
        elif c.geom2 == floor_id:
            other = c.geom1
        if other in loads:
            mujoco.mj_contactForce(model, data, i, f)
            loads[other] += f[0]
    return {name: loads[gid] for name, gid in tire_ids.items()}


def mode_tag():
    tag = "pivot" if PIVOT_MODE else "drive"
    if PLANAR_MODE:
        tag += "_planar"
    return tag


def run_simulation(magnitude, cargo_offset, cargo_mass, run_idx):
    model = build_model(cargo_offset, cargo_mass)
    data = mujoco.MjData(model)

    dl = model.actuator("drive_left").id
    dr = model.actuator("drive_right").id
    sl = model.actuator("steer_left").id
    sr = model.actuator("steer_right").id
    trailer_id = model.body("trailer").id
    hitch_joint = model.joint("hitch")
    hitch_adr = hitch_joint.qposadr[0]
    hitch_is_ball = int(hitch_joint.type[0]) == mujoco.mjtJoint.mjJNT_BALL
    hf_adr = model.sensor("hitch_force").adr[0]
    dt = model.opt.timestep
    trailer_rear_x = -(TONGUE_LEN + 2 * BED_HALF_X)

    pv = None
    if PIVOT_MODE:
        pv = model.actuator("leader_drive").id
        # freewheel the drive wheels: the tow point pulls, not the wheels
        for a in (dl, dr):
            model.actuator_gainprm[a, 0] = 0
            model.actuator_biasprm[a, 2] = 0

    tire_ids = {n: model.geom(n).id for n in TRUCK_TIRES + TRAILER_TIRES}
    floor_id = model.geom("floor").id

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

    os.makedirs("csvs", exist_ok=True)
    csv_filename = (
        f"csvs/Block_Sway_{mode_tag()}_{DISTURBANCE}_run{run_idx + 1}"
        f"_mag{magnitude:g}_v{SPEED_CTRL:.0f}_tongue{TONGUE_LEN:.2f}"
        f"_cargo{cargo_mass:.0f}kg_off{cargo_offset:+.2f}"
        f"_mu{TRAILER_TIRE_MU}.csv"
    )
    headers = ["time", "hitch_yaw_deg", "hitch_lat_force", "trailer_y",
               "car_yaw_deg", "car_roll_deg", "car_y", "tl_grip", "tr_grip",
               "fl_grip", "fr_grip", "rl_grip", "rr_grip", "front_weight",
               "rear_weight", "car_speed"]
    log = {h: [] for h in headers}

    ctrl_state = {}          # persists across steps, passed to control_function
    peak_yaw = 0.0

    with open(csv_filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)

        with mujoco.viewer.launch_passive(model, data) as viewer:
            phase_idx, step_in_phase = 0, 0
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            viewer.cam.trackbodyid = model.body("car").id

            while viewer.is_running() and phase_idx < len(schedule):
                step_start = time.time()
                phase = schedule[phase_idx][0]

                if PIVOT_MODE:
                    # controller OFF; the tow point pulls the truck forward,
                    # its lane-center spring handles everything lateral
                    if phase != "settle":
                        data.ctrl[pv] = PIVOT_SPEED
                    steer = {"swerve_r": magnitude,
                             "swerve_l": -magnitude}.get(phase, 0.0)
                    data.ctrl[sl] = steer
                    data.ctrl[sr] = steer
                elif phase == "record":
                    # ---- your controller drives steering AND speed ----
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
                    mujoco.mj_applyFT(model, data, np.array([0, magnitude, 0]),
                                      np.zeros(3), rear, trailer_id, qfrc)
                    data.qfrc_applied[:] = qfrc
                else:
                    data.qfrc_applied[:] = 0

                mujoco.mj_step(model, data)

                if phase not in ("settle", "spinup"):
                    if hitch_is_ball:
                        q = data.qpos[hitch_adr:hitch_adr + 4]
                        _, _, hitch_yaw = quat_to_euler(*q)
                    else:
                        hitch_yaw = math.degrees(data.qpos[hitch_adr])
                    peak_yaw = max(peak_yaw, abs(hitch_yaw))
                    cw, cx_, cy_, cz_ = data.body("car").xquat
                    car_roll, _, car_yaw = quat_to_euler(cw, cx_, cy_, cz_)
                    grips = tire_normal_loads(model, data, tire_ids, floor_id)

                    # hitch force: sensor reports the force from the truck on
                    # the trailer, in the sensor-site (trailer) frame. Rotate
                    # to world, take the lane-perpendicular (y) component,
                    # and negate -> force the trailer pushes on the truck.
                    # Positive = trailer shoving the truck toward +y (left).
                    f_local = data.sensordata[hf_adr:hf_adr + 3]
                    f_world = data.site(
                        "hitch_force_site").xmat.reshape(3, 3) @ f_local
                    hitch_lat = -float(f_world[1])

                    row = {
                        "time": data.time,
                        "hitch_yaw_deg": hitch_yaw,
                        "hitch_lat_force": hitch_lat,
                        "trailer_y": data.body("trailer").xipos[1],
                        "car_yaw_deg": car_yaw,
                        "car_roll_deg": car_roll,
                        "car_y": data.body("car").xipos[1],
                        "tl_grip": grips["tl_tire"],
                        "tr_grip": grips["tr_tire"],
                        "fl_grip": grips["fl_tire"],
                        "fr_grip": grips["fr_tire"],
                        "rl_grip": grips["rl_tire"],
                        "rr_grip": grips["rr_tire"],
                        "front_weight": grips["fl_tire"] + grips["fr_tire"],
                        "rear_weight": grips["rl_tire"] + grips["rr_tire"],
                        "car_speed": float(np.linalg.norm(
                            data.body("car").cvel[3:6])),
                    }
                    writer.writerow([row[h] for h in headers])
                    for h in headers:
                        log[h].append(row[h])

                step_in_phase += 1
                if step_in_phase >= schedule[phase_idx][1]:
                    phase_idx += 1
                    step_in_phase = 0
                    if (phase_idx < len(schedule)
                            and schedule[phase_idx][0] == "record"):
                        print(f"Disturbance done at t={data.time:.2f}s. "
                              + ("Towed, no controller. Recording..."
                                 if PIVOT_MODE else
                                 "Controller engaged, recording..."))

                viewer.sync()
                if REALTIME:
                    leftover = dt - (time.time() - step_start)
                    if leftover > 0:
                        time.sleep(leftover)

    final_speed = float(np.linalg.norm(data.body("car").cvel[3:6]))
    outcome = ("LOST CONTROL (sway divergence / jackknife)"
               if final_speed < 5 else "recovered")
    print(f"Run {run_idx + 1} ({DISTURBANCE} mag {magnitude:g}, "
          f"cargo {cargo_mass:.0f} kg at {cargo_offset:+.2f} m): {outcome}. "
          f"Peak |hitch yaw| = {peak_yaw:.1f} deg, "
          f"final speed = {final_speed:.1f} m/s")
    print(f"CSV saved at {csv_filename}")
    log = {h: np.array(v) for h, v in log.items()}
    log["magnitude"] = magnitude
    log["cargo_offset"] = cargo_offset
    log["cargo_mass"] = cargo_mass
    return log


def mask_spans(t, mask):
    """Convert a boolean mask over time samples into (start, end) spans."""
    spans, start = [], None
    for i, m in enumerate(mask):
        if m and start is None:
            start = t[i]
        elif not m and start is not None:
            spans.append((start, t[i]))
            start = None
    if start is not None:
        spans.append((start, t[-1]))
    return spans


def plot_results(runs):
    os.makedirs("plots", exist_ok=True)
    tag = mode_tag()
    for i, r in enumerate(runs):
        t = r["time"]
        # blue: trailer at max swing, hitting the hitch stop (= the truck)
        hit_mask = np.abs(r["hitch_yaw_deg"]) >= (HITCH_LIMIT_DEG
                                                  - HITCH_HIT_MARGIN)
        # red: truck flipped sideways
        flip_mask = np.abs(r["car_roll_deg"]) >= FLIP_ROLL_DEG
        hit_spans = mask_spans(t, hit_mask)
        flip_spans = mask_spans(t, flip_mask)

        fig, ax = plt.subplots(4, 2, figsize=(14, 16), sharex=True)
        fig.suptitle(
            f"Run {i + 1} ({tag}) — {DISTURBANCE} "
            f"magnitude {r['magnitude']:g}, "
            f"cargo {r['cargo_mass']:.0f} kg @ {r['cargo_offset']:+.2f} m\n"
            "blue = trailer at max swing / contacting truck, "
            "red = truck flipped sideways")

        ax[0, 0].plot(t, r["hitch_yaw_deg"])
        ax[0, 0].set_ylabel("deg")
        ax[0, 0].set_title("Trailer angle relative to truck (hitch yaw)")

        ax[0, 1].plot(t, r["hitch_lat_force"], color="tab:red")
        ax[0, 1].axhline(0, color="gray", lw=0.5)
        ax[0, 1].set_ylabel("N")
        ax[0, 1].set_title("Lateral force trailer pushes on truck "
                           "(+ = toward +y / left)")

        ax[1, 0].plot(t, r["car_yaw_deg"])
        ax[1, 0].set_ylabel("deg")
        ax[1, 0].set_title("Truck angle relative to road")

        ax[1, 1].plot(t, r["car_y"], color="tab:red")
        ax[1, 1].set_ylabel("m")
        ax[1, 1].set_title("Truck displacement from lane center")

        ax[2, 0].plot(t, r["tl_grip"], label="trailer L")
        ax[2, 0].plot(t, r["tr_grip"], label="trailer R")
        ax[2, 0].set_ylabel("N (normal load)")
        ax[2, 0].set_title("Trailer tire grip")
        ax[2, 0].legend()

        for k, lbl in [("fl_grip", "FL"), ("fr_grip", "FR"),
                       ("rl_grip", "RL"), ("rr_grip", "RR")]:
            ax[2, 1].plot(t, r[k], label=lbl)
        ax[2, 1].set_ylabel("N (normal load)")
        ax[2, 1].set_title("Truck tire grip")
        ax[2, 1].legend()

        ax[3, 0].plot(t, r["front_weight"], label="front axle")
        ax[3, 0].plot(t, r["rear_weight"], label="rear axle")
        ax[3, 0].set_ylabel("N")
        ax[3, 0].set_xlabel("time [s]")
        ax[3, 0].set_title("Truck weight: front vs rear")
        ax[3, 0].legend()

        ax[3, 1].plot(t, r["car_speed"], color="tab:green")
        ax[3, 1].set_ylabel("m/s")
        ax[3, 1].set_xlabel("time [s]")
        ax[3, 1].set_title("Truck speed")

        # shade highlight bands across every subplot
        for a in ax.flat:
            for s, e in hit_spans:
                a.axvspan(s, e, color="tab:blue", alpha=0.3, lw=0)
            for s, e in flip_spans:
                a.axvspan(s, e, color="red", alpha=0.3, lw=0)

        fig.tight_layout()
        out = (f"plots/run{i + 1}_{tag}_{DISTURBANCE}"
               f"_mag{r['magnitude']:g}.png")
        fig.savefig(out, dpi=120)
        print(f"Plot saved at {out}")
    plt.show()


if __name__ == "__main__":
    results = []
    for i in range(N_RUNS):
        mag = DISTURB_START + i * DISTURB_STEP
        off = CARGO_OFFSET + i * CARGO_OFFSET_STEP
        mass = CARGO_MASS + i * CARGO_MASS_STEP
        print(f"\n=== Run {i + 1}/{N_RUNS} — {DISTURBANCE} mag {mag:g}, "
              f"cargo {mass:g} kg @ {off:+.2f} m ===")
        results.append(run_simulation(mag, off, mass, i))
    plot_results(results)
