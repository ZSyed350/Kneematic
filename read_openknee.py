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
from scipy.interpolate import interp1d

from extract_data import extract_openknee_data

PRINT_TDMS = False

# TODO confirm the units of dt
DATA_ROOT = "OpenKneeData"
DATA_MAP = "datamap.json"
VERBOSE = True

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
    plt.savefig("pictures/all_data.png")
    plt.show()

def plot_tibiofemoral_data(all_trials):
    tibiofemoral_trials = [trial for trial in all_trials if trial["joint"] == "TibiofemoralJoint/KinematicsKinetics"]

    if not tibiofemoral_trials:
        print("No tibiofemoral trials found.")
        return

    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    # Subplot 1: Extension Torque vs Flexion Angle
    for trial in tibiofemoral_trials:
        color = COLOR_SCHEME[trial["joint"]][trial["optimized"]]
        try:
            axs[0].plot(trial["angle"], trial["torque"], color=color)
        except ValueError:
            print(f"[ValueError] Array length mismatch for angle/torque in: {trial['filepath']}")

    axs[0].set_xlabel("Flexion Angle (degrees)")
    axs[0].set_ylabel("Extension Torque (Nm)")
    axs[0].set_title("Tibiofemoral: Extension Torque vs Flexion Angle")
    axs[0].grid(True)

    # Subplot 2: Torque and Angle Over Time
    ax2 = axs[1]
    ax2b = ax2.twinx()

    for trial in tibiofemoral_trials:
        color = COLOR_SCHEME[trial["joint"]][trial["optimized"]]
        try:
            ax2.plot(trial["time"], trial["torque"], color=color)
            ax2b.plot(trial["time"], trial["angle"], linestyle='--', color=color)
        except ValueError:
            print(f"[ValueError] Array length mismatch for time-series in: {trial['filepath']}")

    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Extension Torque (Nm)", color='tab:red')
    ax2.tick_params(axis='y', labelcolor='tab:red')
    ax2.set_title("Tibiofemoral: Torque and Angle Over Time")
    ax2.grid(True)

    ax2b.set_ylabel("Flexion Angle (deg)", color='tab:blue')
    ax2b.tick_params(axis='y', labelcolor='tab:blue')
    ax2.xaxis.set_major_formatter(ScalarFormatter(useOffset=False))
    ax2.ticklabel_format(style='plain', axis='x')

    # Legend specific to Tibiofemoral
    legend_patches = [
        mpatches.Patch(color='lightpink', label='Tibiofemoral – Unoptimized'),
        mpatches.Patch(color='deeppink', label='Tibiofemoral – Optimized')
    ]
    axs[0].legend(handles=legend_patches, loc="upper right", title="Trial Categories")

    plt.tight_layout()
    plt.savefig("pictures/tibiofemoral_data.png")
    plt.show()

def split_data_by_flexion(all_trials):
    tibiofemoral_trials = [t for t in all_trials if t["joint"] == "TibiofemoralJoint/KinematicsKinetics"]

    if not tibiofemoral_trials:
        print("No tibiofemoral trials found.")
        return
    
    # Flatten all angle arrays and compute global min/max
    all_angles = np.concatenate([np.array(trial["angle"]) for trial in tibiofemoral_trials])
    global_min_angle = np.min(all_angles)
    global_max_angle = np.max(all_angles)

    processed_trials = []
    
    for trial in all_trials:
        angle = np.array(trial["angle"])
        torque = np.array(trial["torque"])
        time = np.array(trial["time"])

        slope = np.gradient(angle)  # approximate derivative of angle over time
        slope_threshold = 0.002  # degrees per sample

        # Directly get index arrays
        rising_indices = np.where(slope > slope_threshold)[0]
        falling_indices = np.where(slope < -slope_threshold)[0]
        
        if len(rising_indices) == 0 or len(falling_indices) == 0:
            continue  # skip trial if we can't find valid rising/falling

        # Extract segments
        last_rising_idx = rising_indices[-1]
        first_falling_idx = falling_indices[0]

        if last_rising_idx >= first_falling_idx:
            continue  # skip if invalid index range

        rising_idx = np.arange(rising_indices[0], last_rising_idx + 1)
        flat_idx = np.arange(last_rising_idx + 1, first_falling_idx)
        falling_idx = np.arange(first_falling_idx, falling_indices[-1] + 1)

        processed_trials.append({
            "rising": {
                "angle": angle[rising_idx],
                "torque": torque[rising_idx],
                "time": time[rising_idx],
            },
            "flat": {
                "angle": angle[flat_idx],
                "torque": torque[flat_idx],
                "time": time[flat_idx],
            },
            "falling": {
                "angle": angle[falling_idx],
                "torque": torque[falling_idx],
                "time": time[falling_idx],
            }
        })

    if VERBOSE:
        # Create 2 subplots: one for rising (flexion), one for falling (extension)
        fig, axs = plt.subplots(1, 2, figsize=(18, 6), sharey=True)
        axs[0].set_title("Flexion Phase (Rising)")
        axs[1].set_title("Extension Phase (Falling)")

        # Define common visual styles
        torque_color = "tab:red"
        angle_color = "tab:blue"
        torque_alpha = 0.6
        angle_alpha = 0.5
        angle_style = "--"

        for i, phase in enumerate(["rising", "falling"]):
            ax_torque = axs[i]
            ax_angle = ax_torque.twinx()

            for trial in processed_trials:
                ax_torque.plot(
                    trial[phase]["time"],
                    trial[phase]["torque"],
                    color=torque_color,
                    alpha=torque_alpha,
                )
                ax_angle.plot(
                    trial[phase]["time"],
                    trial[phase]["angle"],
                    linestyle=angle_style,
                    color=angle_color,
                    alpha=angle_alpha,
                )

            ax_torque.set_xlabel("Time (s)")
            ax_torque.set_ylabel("Torque (Nm)", color=torque_color)
            ax_torque.tick_params(axis="y", labelcolor=torque_color)
            ax_torque.grid(True)
            ax_torque.xaxis.set_major_formatter(ScalarFormatter(useOffset=False))
            ax_torque.ticklabel_format(style="plain", axis="x")

            ax_angle.set_ylabel("Angle (deg)", color=angle_color)
            ax_angle.tick_params(axis="y", labelcolor=angle_color)

        plt.tight_layout()
        plt.savefig("pictures/flexion_extension_split.png")
        plt.show()

        return processed_trials
    
def process_split_trials(split_trials):
    if not split_trials:
        print("No trials to process.")
        return

    # Filter: Keep only trials where the maximum angle (from all phases) is at least 80 degrees
    filtered_trials = []
    for trial in split_trials:
        all_angles = np.concatenate([
            trial["rising"]["angle"],
            trial["flat"]["angle"],
            trial["falling"]["angle"]
        ])
        if np.max(all_angles) >= 80:
            filtered_trials.append(trial)

    if not filtered_trials:
        print("No trials passed the 80-degree flexion threshold.")
        return

    # Plotting
    fig, axs = plt.subplots(1, 2, figsize=(18, 6), sharey=True)
    axs[0].set_title("Flexion Phase (Rising) - Filtered")
    axs[1].set_title("Extension Phase (Falling) - Filtered")

    # Define common visual styles
    torque_color = "tab:red"
    angle_color = "tab:blue"
    torque_alpha = 0.6
    angle_alpha = 0.5
    angle_style = "--"

    for i, phase in enumerate(["rising", "falling"]):
        ax_torque = axs[i]
        ax_angle = ax_torque.twinx()

        for trial in filtered_trials:
            ax_torque.plot(
                trial[phase]["time"],
                trial[phase]["torque"],
                color=torque_color,
                alpha=torque_alpha,
            )
            ax_angle.plot(
                trial[phase]["time"],
                trial[phase]["angle"],
                linestyle=angle_style,
                color=angle_color,
                alpha=angle_alpha,
            )

        ax_torque.set_xlabel("Time (s)")
        ax_torque.set_ylabel("Torque (Nm)", color=torque_color)
        ax_torque.tick_params(axis="y", labelcolor=torque_color)
        ax_torque.grid(True)
        ax_torque.xaxis.set_major_formatter(ScalarFormatter(useOffset=False))
        ax_torque.ticklabel_format(style="plain", axis="x")

        ax_angle.set_ylabel("Angle (deg)", color=angle_color)
        ax_angle.tick_params(axis="y", labelcolor=angle_color)

    plt.tight_layout()
    plt.savefig("pictures/flexion_extension_filtered.png")
    plt.show()

    return filtered_trials


def main():
    all_trials = read_all_trials()
    # plot_all_data(all_trials)
    # plot_tibiofemoral_data(all_trials)
    flexion_split_trials = split_data_by_flexion(all_trials)
    process_split_trials(flexion_split_trials)
    
if __name__ == "__main__":
    main()