"""
http://archive.simtk.org/oks/MRI_JointMech/doc/2013RB-001-002.A simVITRO Data File Structure Quick Reference Guide.pdf --> CONTAINS IMPORTANT INFO ON THE PROCESSED FILES
including that the data is TIME ALIGNED and been resampled. However, the sampling rate IS AT LARGE

THE CONFIG FILES ARE VERY IMPORTANT, THEY CONTAIN THE UNITS

explanation on exactly what the data is in mentioned in the appendix A2 at https://www.sciencedirect.com/science/article/pii/S2352340921001086?via%3Dihub#sec0014

TODO read the experiment steps to see what the timeline for each tril is supposed to be
"""


import os
import json
import numpy as np
from nptdms import TdmsFile
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import matplotlib.patches as mpatches

from extract_data import extract_openknee_data

PRINT_TDMS = False

# TODO confirm the units of dt
DATA_ROOT = "OpenKneeData"
DATA_MAP = "datamap.json"

# Color map by joint type and optimization status
COLOR_SCHEME = {
    "PatellofemoralJoint/KinematicsKinetics": {
        "optimized": "darkblue",
        "unoptimized": "lightblue"
    },
    "TibiofemoralJoint/KinematicsKinetics": {
        "optimized": "deeppink",
        "unoptimized": "lightpink"
    }
}

def get_avg_sampling_rate(actual_dt):
    actual_dt_s = actual_dt / 1000.0

    # Focus on the region where it stabilizes (~after first 15–20 entries)
    stable_dt = actual_dt_s[20:] # skip initialization

    # Estimate sampling rate
    mean_dt = np.mean(stable_dt)
    sampling_rate_hz = 1.0 / mean_dt
    print(f"Sampling rate: {sampling_rate_hz}")

    return sampling_rate_hz

def read_all_trials():
    with open(DATA_MAP, "r") as f:
        data_map = json.load(f)

    all_trials = []

    for joint_type, conditions in data_map.items():
        for opt_status, trials in conditions.items():
            for trial in trials:
                rel_path = trial.get("path", "").strip()
                if not rel_path:
                    print(f"[WARNING] Skipping trial with empty path: {trial}")
                    continue

                full_path = os.path.join(DATA_ROOT, trial["subject"], joint_type, rel_path)
                if not os.path.isfile(full_path):
                    print(f"[WARNING] File not found: {full_path}. Skipping.")
                    continue

                try:
                    time, angle, torque = extract_openknee_data(full_path, verbose=PRINT_TDMS)

                    if np.min(torque) < -8:
                        print(f"[OUTLIER] Torque below -10 Nm in file: {full_path} (min: {np.min(torque):.2f})")

                    all_trials.append({
                        "filepath": full_path,
                        "joint": joint_type,
                        "optimized": opt_status,
                        "time": time,
                        "angle": angle,
                        "torque": torque
                    })
                except Exception as e:
                    print(f"[ERROR] Failed to load {full_path}: {e}")

    return all_trials

def plot_all_data(all_trials):
    # Create figure with two side-by-side subplots
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    # Subplot 1: Extension Torque vs Flexion Angle
    for trial in all_trials:
        color = COLOR_SCHEME[trial["joint"]][trial["optimized"]]
        try:
            axs[0].plot(trial["angle"], trial["torque"], color=color)
        except ValueError:
            print(f"[ValueError] Array length mismath for torque vs angle {trial["filepath"]}")
    axs[0].set_xlabel("Flexion Angle (degrees)")
    axs[0].set_ylabel("Extension Torque (Nm)")
    axs[0].set_title("Extension Torque vs Flexion Angle")
    axs[0].grid(True)

    # Subplot 2: Torque and Angle Over Time
    ax2 = axs[1]
    ax2b = ax2.twinx()
    for trial in all_trials:
        color = COLOR_SCHEME[trial["joint"]][trial["optimized"]]
        try:
            ax2.plot(trial["time"], trial["torque"], color=color)
            ax2b.plot(trial["time"], trial["angle"], linestyle='--', color=color)
        except ValueError:
            print(f"[ValueError] Array length mismatch for time-series data {trial["filepath"]}")

    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Extension Torque (Nm)", color='tab:red')
    ax2.tick_params(axis='y', labelcolor='tab:red')
    ax2.set_title("Torque and Angle Over Time")
    ax2.grid(True)

    ax2b.set_ylabel("Flexion Angle (deg)", color='tab:blue')
    ax2b.tick_params(axis='y', labelcolor='tab:blue')

    # Format x-axis to avoid scientific notation
    ax2.xaxis.set_major_formatter(ScalarFormatter(useOffset=False))
    ax2.ticklabel_format(style='plain', axis='x')

        # Add a legend for color coding
    legend_patches = [
        mpatches.Patch(color='lightblue', label='Patellofemoral – Unoptimized'),
        mpatches.Patch(color='darkblue', label='Patellofemoral – Optimized'),
        mpatches.Patch(color='lightpink', label='Tibiofemoral – Unoptimized'),
        mpatches.Patch(color='deeppink', label='Tibiofemoral – Optimized')
    ]
    axs[0].legend(handles=legend_patches, loc="upper right", title="Trial Categories")

    plt.tight_layout()
    plt.savefig("all_data.png")
    plt.show()

def main():
    all_trials = read_all_trials()
    plot_all_data(all_trials)
    
if __name__ == "__main__":
    main()