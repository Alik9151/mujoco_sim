import pandas as pd
import matplotlib.pyplot as plt

PULL = float(input("Enter the pull force for the simulation you want to graph: "))



csv_filename = f"csvs/Trailer_Position_and_Orientation_Over_Time_For_Pull_{PULL}.csv"

df = pd.read_csv(csv_filename)

df.columns = df.columns.str.strip()

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

ax1.axvline(x=1.5, color='red', linestyle=':', linewidth=2, label='Pull Applied')
ax2.axvline(x=1.5, color='red', linestyle=':', linewidth=2, label='Pull Applied')

ax1.plot(df['time'], df['rel_trailer_x'], label='Hitch X', color='b')
ax1.plot(df['time'], df['rel_trailer_y'], label='Hitch Y', color='g')
ax1.plot(df['time'], df['rel_trailer_z'], label='Hitch Z', color='r')
ax1.set_ylabel('Position (units)')
ax1.set_title(csv_filename)
ax1.legend()
ax1.grid(True)

ax2.plot(df['time'], df['hitch_roll'], label='Roll', color='c', linestyle='--')
ax2.plot(df['time'], df['hitch_pitch'], label='Pitch', color='m', linestyle='--')
ax2.plot(df['time'], df['hitch_yaw'], label='Yaw', color='y', linestyle='--')
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Orientation (deg/rad)')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()