import mujoco
import mujoco.viewer
#this file is old
# 1. Load the model from the XML file
# Make sure "trailer_sway.xml" is in the same directory as this script
xml_path = "trailer_sway.xml"
model = mujoco.MjModel.from_xml_path(xml_path)

# 2. Create the data structure to hold the simulation state
data = mujoco.MjData(model)

print("Starting simulation... Close the window to stop.")

# 3. Launch the interactive viewer
# This function handles the physics stepping and rendering loop for you
mujoco.viewer.launch(model, data)