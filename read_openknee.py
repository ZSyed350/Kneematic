"""
http://archive.simtk.org/oks/MRI_JointMech/doc/2013RB-001-002.A simVITRO Data File Structure Quick Reference Guide.pdf --> CONTAINS IMPORTANT INFO ON THE PROCESSED FILES
including that the data is TIME ALIGNED and been resampled. However, the sampling rate IS AT LARGE

THE CONFIG FILES ARE VERY IMPORTANT, THEY CONTAIN THE UNITS

explanation on exactly what the data is in mentioned in the appendix A2 at https://www.sciencedirect.com/science/article/pii/S2352340921001086?via%3Dihub#sec0014

TODO read the experiment steps to see what the timeline for each tril is supposed to be
"""


import os
import numpy as np
from nptdms import TdmsFile
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

PRINT_TDMS = False

# TODO validate sampling rate and other meta data
# TODO confirm the units of dt
DATA_ROOT = "OpenKneeData"
PATELLOFEMORAL = "PatellofemoralJoint/KinematicsKinetics"
TIBIOFEMORAL = "TibiofemoralJoint/KinematicsKinetics"
DATA_MAP = "datamap.json"

def get_avg_sampling_rate(actual_dt):
    actual_dt_s = actual_dt / 1000.0

    # Focus on the region where it stabilizes (~after first 15–20 entries)
    stable_dt = actual_dt_s[20:] # skip initialization

    # Estimate sampling rate
    mean_dt = np.mean(stable_dt)
    sampling_rate_hz = 1.0 / mean_dt
    print(f"Sampling rate: {sampling_rate_hz}")

    return sampling_rate_hz

def extract_openknee_data(filepath):
    tdms_file = TdmsFile.read(filepath)

    if PRINT_TDMS:
        # List all groups and channels
        for group in tdms_file.groups():
            print(f"Group: {group.name}")
            for channel in group.channels():
                print(f" Channel: {channel.name}")

    # Extract relevant signals
    # NOTE the first 20 samples have odd dt, not sure how that effects the analysis
    flexion_angle = tdms_file["Kinematics.JCS.Actual"]["Flexion Angle"][:]
    extension_torque = tdms_file["State.JCS Load"]["JCS Load Extension Torque"][:]
    actual_dt = tdms_file["Timing.Control Loop Actual dt"]["Actual dt"][:] # Control loop Δt values in ms
    setpoint_time = tdms_file["Timing.Sync Trigger"]["Setpoint Time"][:] # Not sure exactly what this is

    # Time reconstruction
    actual_dt = tdms_file["Timing.Control Loop Actual dt"]["Actual dt"][:] # Control loop intervals in ms
    actual_dt_s = actual_dt / 1000.0 # Convert to seconds
    time = np.cumsum(actual_dt_s) # Time signal in seconds

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