import time
import csv
import math
import numpy as np
import mujoco
import mujoco.viewer

def quat_to_euler(w, x, y, z):
    """Converts a quaternion (w, x, y, z) to Euler angles (roll, pitch, yaw) in degrees."""
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = math.copysign(math.pi / 2, sinp)
    else:
        pitch = math.asin(sinp)

    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return math.degrees(roll), math.degrees(pitch), math.degrees(yaw)

def run_simulation(IMPULSE):
    xml_path = "trailer_sway.xml"
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    drive_left = model.actuator("drive_left").id
    drive_right = model.actuator("drive_right").id

    trailer_id = model.body("trailer").id

    rear_point = data.body("trailer").xpos + \
                data.body("trailer").xmat.reshape(3,3) @ np.array([-1.0, 0.0, 0.0])


    hitch_qpos_adr = model.joint("hitch").qposadr[0]

    csv_filename = f"csvs/Trailer_Position_and_Orientation_Over_Time_For_Impulse_{IMPULSE}.csv"

    headers = [
        "time", 
        "rel_trailer_x", "rel_trailer_y", "rel_trailer_z",  
        "hitch_roll", "hitch_pitch", "hitch_yaw"
    ]

    prev_yaw = None
    stable_time = 0.0

    RATE_TOL = 0.1
    HOLD_TIME = .5

    with open(csv_filename, mode="w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(headers)

        with mujoco.viewer.launch_passive(model, data) as viewer:
            force = np.array([0.0, IMPULSE, 0.0])   # +Y
            torque = np.zeros(3)

            qfrc = np.zeros(model.nv)

            mujoco.mj_applyFT(
                model,
                data,
                force,
                torque,
                rear_point,
                trailer_id,
                qfrc,
            )

            data.qfrc_applied[:] = qfrc

            mujoco.mj_step(model, data)

            data.qfrc_applied[:] = 0


            while viewer.is_running():
                step_start = time.time()

                data.ctrl[drive_left] = 50.0
                data.ctrl[drive_right] = 50.0

                mujoco.mj_step(model, data)

                sim_time = data.time
                car_pos = data.body("car").xipos
                trailer_pos = data.body("trailer").xipos
                
                car_rot_matrix = data.body("car").xmat.reshape(3, 3)

                translation = trailer_pos - car_pos
                relative_trailer_pos = np.dot(car_rot_matrix.T, translation)

                q = data.qpos[hitch_qpos_adr : hitch_qpos_adr + 4]
                hitch_roll, hitch_pitch, hitch_yaw = quat_to_euler(q[0], q[1], q[2], q[3])

                writer.writerow([
                    sim_time,
                    relative_trailer_pos[0], relative_trailer_pos[1], relative_trailer_pos[2],
                    hitch_roll, hitch_pitch, hitch_yaw
                ])

                viewer.sync()

                time_until_next_step = model.opt.timestep - (time.time() - step_start)

                if prev_yaw is not None:
                    yaw_rate = (hitch_yaw - prev_yaw) / model.opt.timestep

                    if abs(yaw_rate) < RATE_TOL:
                        stable_time += model.opt.timestep
                    else:
                        stable_time = 0.0

                    if stable_time >= HOLD_TIME:
                        print(f"Stabalized at {data.time}s")
                        break

                prev_yaw = hitch_yaw
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)

    print("Simulation finished. CSV file successfully saved at {csv_filename}")




for i in range(20):
    IMPULSE = 100000.0 + 100.0 * i
    run_simulation(IMPULSE)