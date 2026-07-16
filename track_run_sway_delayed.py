import time
import csv
import math
import numpy as np
import mujoco
import mujoco.viewer

STEP_WAIT = 1000
SPEED = 100
PULL_FORCE = 15000
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
    IMPULSE_FORCE = IMPULSE / 0.005
    print(IMPULSE_FORCE)
    xml_path = "trailer_sway.xml"
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    # drive_r_left = model.actuator("drive_r_left").id
    # drive_r_right = model.actuator("drive_r_right").id
    # drive_f_left = model.actuator("drive_f_left").id
    # drive_f_right = model.actuator("drive_f_right").id
    front_pull = model.actuator("front_pull").id

    trailer_id = model.body("trailer").id


    hitch_qpos_adr = model.joint("hitch").qposadr[0]

    csv_filename = f"csvs/Trailer_Position_and_Orientation_Over_Time_For_Impulse_{IMPULSE}.csv"

    headers = [
        "time", 
        "rel_trailer_x", "rel_trailer_y", "rel_trailer_z",  
        "hitch_roll", "hitch_pitch", "hitch_yaw"
    ]

    prev_yaw = None
    stable_time = 0.0

    RATE_TOL = 1
    HOLD_TIME = .5

    with open(csv_filename, mode="w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(headers)
        with mujoco.viewer.launch_passive(model, data) as viewer:
            step = 0
            #camera settings
            car_id = model.body('car').id
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            viewer.cam.trackbodyid = car_id

            impulse_force_mag = IMPULSE / model.opt.timestep
            
            while viewer.is_running():
                step += 1
                
                data.qfrc_applied[:] = 0.0

                if step == STEP_WAIT and False:
                    R = data.body("trailer").xmat.reshape(3, 3)
                    rear_point = (
                        data.body("trailer").xpos +
                        R @ np.array([-1.0, 0.0, 0.0])
                    )

                    force = np.array([0.0, impulse_force_mag, 0.0])
                    qfrc = np.zeros(model.nv)
                    
                    mujoco.mj_applyFT(
                        model,
                        data,
                        force,
                        np.zeros(3),
                        rear_point,
                        trailer_id,
                        qfrc,
                    )

                    data.qfrc_applied[:] = qfrc
                    
                    print(f"Applied impulse force of {impulse_force_mag:.2f} at {data.time:.2f} seconds")

                step_start = time.time()

                # data.ctrl[drive_r_left] = SPEED
                # data.ctrl[drive_r_right] = SPEED
                # data.ctrl[drive_f_left] = SPEED
                # data.ctrl[drive_f_right] = SPEED
                data.ctrl[front_pull] = PULL_FORCE

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
                    print(yaw_rate)

                    if abs(yaw_rate) < RATE_TOL:
                        stable_time += model.opt.timestep
                    else:
                        stable_time = 0.0

                    if stable_time >= HOLD_TIME and step > STEP_WAIT and False:
                        print(f"Stabilized at {data.time:.2f}s")
                        break

                prev_yaw = hitch_yaw
                
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)
    print(f"Simulation finished. CSV file successfully saved at {csv_filename}")
    



if __name__ == "__main__":
    for i in range(10):
        IMPULSE = 30000 + 250 * i
        run_simulation(IMPULSE)
        time.sleep(.5)