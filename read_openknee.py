import os
import numpy as np
from nptdms import TdmsFile
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

# TODO validate the units
# TODO validate what is "optimized" meaning
# TODO validate sampling rate and other meta data

FILES = [
    "OpenKneeData/joint_mechanics-oks001/PatellofemoralJoint/KinematicsKinetics/005_Passive Flexion 0-60/Data/005_Passive Flexion 0-60_main_processed.tdms",
    "OpenKneeData/joint_mechanics-oks001/PatellofemoralJoint/KinematicsKinetics/006_Passive Flexion 0-60, optimized/Data/006_Passive Flexion 0-60, optimized_main_processed.tdms",
    "OpenKneeData/joint_mechanics-oks001/TibiofemoralJoint/KinematicsKinetics/004_passive flexion/Data/004_passive flexion_main_processed.tdms",
    "OpenKneeData/joint_mechanics-oks001/TibiofemoralJoint/KinematicsKinetics/005_passive flexion_optimized/Data/005_passive flexion_optimized_main_processed.tdms"
]

def extract_openknee_data(filepath):
    tdms_file = TdmsFile.read(filepath)

    # Extract relevant signals
    flexion_angle = tdms_file["Kinematics.JCS.Actual"]["Flexion Angle"][:]
    extension_torque = tdms_file["State.JCS Load"]["JCS Load Extension Torque"][:]
    dt = tdms_file["Timing.Control Loop Actual dt"]["Actual dt"][:]

    # Time reconstruction
    # TODO refer to source code for this
    time = np.cumsum(dt) if len(dt) > 1 else np.arange(len(flexion_angle)) * dt[0]

    return time, flexion_angle, extension_torque

def read_and_plot_all_trials():
    all_trials = []

    for filepath in FILES:
        time, flexion_angle, extension_torque = extract_openknee_data(filepath)
        trial_label = os.path.basename(filepath).split("_")[0]
        all_trials.append({
            "label": trial_label,
            "time": time,
            "angle": flexion_angle,
            "torque": extension_torque
        })

    # Create figure with two side-by-side subplots
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))

    # Subplot 1: Extension Torque vs Flexion Angle
    for trial in all_trials:
        axs[0].plot(trial["angle"], trial["torque"], label=trial["label"])
    axs[0].set_xlabel("Flexion Angle (degrees)")
    axs[0].set_ylabel("Extension Torque (Nm)")
    axs[0].set_title("Extension Torque vs Flexion Angle")
    axs[0].grid(True)
    axs[0].legend()

    # Subplot 2: Torque and Angle Over Time (twinx)
    ax2 = axs[1]
    for trial in all_trials:
        ax2.plot(trial["time"], trial["torque"], label=f"{trial['label']} - Torque")
    ax2.set_ylabel("Extension Torque (Nm)", color='tab:red')
    ax2.tick_params(axis='y', labelcolor='tab:red')
    ax2.set_xlabel("Time (s)")
    ax2.set_title("Torque and Angle Over Time")
    ax2.grid(True)

    # Add second y-axis for flexion angle
    ax2b = ax2.twinx()
    for trial in all_trials:
        ax2b.plot(trial["time"], trial["angle"], linestyle='--', label=f"{trial['label']} - Angle")
    ax2b.set_ylabel("Flexion Angle (deg)", color='tab:blue')
    ax2b.tick_params(axis='y', labelcolor='tab:blue')

    # Clean up time axis formatting
    ax2.xaxis.set_major_formatter(ScalarFormatter(useOffset=False))
    ax2.ticklabel_format(style='plain', axis='x')

    # Combine legends from both y-axes
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2b.get_legend_handles_labels()
    axs[1].legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    read_and_plot_all_trials()