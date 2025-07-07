"""Note that additional trials exist"""
import os

import numpy as np
from nptdms import TdmsFile
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

FILES = [
    "OpenKneeData/joint_mechanics-oks001/PatellofemoralJoint/KinematicsKinetics/005_Passive Flexion 0-60/Data/005_Passive Flexion 0-60_main_processed.tdms",
    "OpenKneeData/joint_mechanics-oks001/PatellofemoralJoint/KinematicsKinetics/006_Passive Flexion 0-60, optimized/Data/006_Passive Flexion 0-60, optimized_main_processed.tdms",
]

def read_openknee_file(filepath):
    tdms_file = TdmsFile.read(filepath)

    # List all groups and channels
    for group in tdms_file.groups():
        print(f"Group: {group.name}")
        for channel in group.channels():
            print(f"  Channel: {channel.name}")

    # Access a specific channel
    # Example: assuming group is 'Group1' and channel is 'Channel1'
    try:
        channel = tdms_file["Group1"]["Channel1"]
        data = channel[:]  # Extract the data as a NumPy array
        print(data)
    except KeyError:
        print("Specified group/channel not found in the file.")

    # Extract torque channels
    external_torque = tdms_file["State.JCS Load"]["JCS Load External Rotation Torque"][:]  # this is the external torque causing flexion
    flexion_angle = tdms_file["Kinematics.JCS.Actual"]["Flexion Angle"][:]
    extension_torque = tdms_file["State.JCS Load"]["JCS Load Extension Torque"][:]  # the muscle resists with extension torque
    dt = tdms_file["Timing.Control Loop Actual dt"]["Actual dt"][:]  #time

    time = np.cumsum(dt)

    # Create side-by-side subplots
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Flexion Angle vs Extension Torque
    axs[0].plot(flexion_angle, extension_torque, color='tab:purple')
    axs[0].set_xlabel("Flexion Angle (degrees)")
    axs[0].set_ylabel("Extension Torque (Nm)")
    axs[0].set_title("Extension Torque vs Flexion Angle")
    axs[0].grid(True)

    # Plot 2: Extension Torque and Flexion Angle over Time with twinx
    ax2 = axs[1]
    ax2.plot(time, extension_torque, color='tab:red', label="Extension Torque")
    ax2.set_ylabel("Extension Torque (Nm)", color='tab:red')
    ax2.tick_params(axis='y', labelcolor='tab:red')

    # Twin y-axis for flexion angle
    ax2b = ax2.twinx()
    ax2b.plot(time, flexion_angle, color='tab:blue', label="Flexion Angle")
    ax2b.set_ylabel("Flexion Angle (deg)", color='tab:blue')
    ax2b.tick_params(axis='y', labelcolor='tab:blue')

    # Common x-axis settings
    ax2.set_xlabel("Time (s)")
    ax2.set_title("Torque and Angle Over Time")
    ax2.grid(True)

    # Remove scientific notation from time axis
    ax2.xaxis.set_major_formatter(ScalarFormatter(useOffset=False))
    ax2.ticklabel_format(style='plain', axis='x')

    plt.tight_layout()
    plt.show()

def read_openknee_file_main():
    for filepath in FILES:
        read_openknee_file(filepath)

if __name__ == "__main__":
    read_openknee_file_main()