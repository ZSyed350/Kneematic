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
VERBOSE = False

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
    
    if VERBOSE:

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

import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

def plot_torque_vs_angle(filtered_trials, savepath):
    if not filtered_trials:
        print("No filtered trials to plot.")
        return

    fig, axs = plt.subplots(1, 2, figsize=(18, 6), sharey=True)
    axs[0].set_title("Flexion Phase (Rising) – Torque vs Angle")
    axs[1].set_title("Extension Phase (Falling) – Torque vs Angle")

    # Visual styles
    color = "tab:red"
    alpha = 0.6

    # Plot Flexion (Rising)
    for trial in filtered_trials:
        axs[0].plot(
            trial["rising"]["angle"],
            trial["rising"]["torque"],
            color=color,
            alpha=alpha
        )

    # Plot Extension (Falling) — flip x-axis
    for trial in filtered_trials:
        axs[1].plot(
            trial["falling"]["angle"],
            trial["falling"]["torque"],
            color=color,
            alpha=alpha
        )
    axs[1].invert_xaxis()

    # Axis labels and formatting
    for ax in axs:
        ax.set_xlabel("Angle (degrees)")
        ax.set_ylabel("Torque (Nm)")
        ax.grid(True)
        ax.xaxis.set_major_formatter(ScalarFormatter(useOffset=False))
        ax.ticklabel_format(style="plain", axis="x")

    plt.tight_layout()
    plt.savefig(savepath)
    plt.show()

def smooth_signals(filtered_trials, window_size=50):
    """
    Applies simple moving average to torque signals in each trial's rising and falling phases.
    Returns a new list of trials with smoothed 'torque' arrays.
    """
    if not filtered_trials:
        print("No trials to smooth.")
        return []

    def moving_average(x, w):
        return np.convolve(x, np.ones(w) / w, mode='same')

    smoothed_trials = []

    for trial in filtered_trials:
        new_trial = {}

        for phase in ["rising", "flat", "falling"]:
            angle = trial[phase]["angle"]
            torque = trial[phase]["torque"]
            time = trial[phase]["time"]

            if phase in ["rising", "falling"] and len(torque) >= window_size:
                smoothed_torque = moving_average(torque, window_size)
            else:
                smoothed_torque = torque  # unchanged for 'flat' or too-short segments

            new_trial[phase] = {
                "angle": angle,
                "torque": smoothed_torque,
                "time": time
            }

        smoothed_trials.append(new_trial)

    return smoothed_trials

def average_trials(smoothed_trials, num_points=200):
    """
    Interpolates and averages multiple smoothed trials over a common angle grid.
    Plots all individual trials in red and the average in solid black.
    Returns a dict with 'rising' and 'falling' averaged angle and torque arrays.
    """
    if not smoothed_trials:
        print("No trials to average.")
        return None

    def average_phase(phase):
        angle_sets = [trial[phase]["angle"] for trial in smoothed_trials]
        torque_sets = [trial[phase]["torque"] for trial in smoothed_trials]

        # Compute common overlapping range
        min_angle = max(np.min(a) for a in angle_sets)
        max_angle = min(np.max(a) for a in angle_sets)
        if min_angle >= max_angle:
            raise ValueError(f"Insufficient angle overlap in {phase} phase for averaging.")
        
        common_grid = np.linspace(min_angle, max_angle, num_points)
        interpolated_torques = []

        for angle, torque in zip(angle_sets, torque_sets):
            if len(angle) < 2:
                continue
            try:
                f_interp = interp1d(angle, torque, kind='linear', bounds_error=False, fill_value="extrapolate")
                interpolated = f_interp(common_grid)
                interpolated_torques.append(interpolated)
            except Exception as e:
                print(f"Skipping trial due to interpolation error: {e}")
                continue

        if not interpolated_torques:
            return {"angle": common_grid, "torque": np.zeros_like(common_grid)}

        mean_torque = np.mean(interpolated_torques, axis=0)
        return {
            "angle": common_grid,
            "torque": mean_torque,
            "all_interpolated": interpolated_torques  # for plotting
        }

    # Average both phases
    avg_rising = average_phase("rising")
    avg_falling = average_phase("falling")

    # === Plotting ===
    fig, axs = plt.subplots(1, 2, figsize=(18, 6), sharey=True)
    axs[0].set_title("Flexion Phase (Rising)")
    axs[1].set_title("Extension Phase (Falling)")

    for interp in avg_rising["all_interpolated"]:
        axs[0].plot(avg_rising["angle"], interp, color="tab:red", alpha=0.5)
    axs[0].plot(avg_rising["angle"], avg_rising["torque"], color="black", linewidth=2, label="Average")

    for interp in avg_falling["all_interpolated"]:
        axs[1].plot(avg_falling["angle"], interp, color="tab:red", alpha=0.5)
    axs[1].plot(avg_falling["angle"], avg_falling["torque"], color="black", linewidth=2, label="Average")
    axs[1].invert_xaxis()

    for ax in axs:
        ax.set_xlabel("Angle (degrees)")
        ax.set_ylabel("Torque (Nm)")
        ax.grid(True)
        ax.legend()
        ax.xaxis.set_major_formatter(ScalarFormatter(useOffset=False))
        ax.ticklabel_format(style="plain", axis="x")

    plt.tight_layout()
    plt.savefig("pictures/averaged_torque_vs_angle.png")
    plt.show()

    return {
        "rising": {
            "angle": avg_rising["angle"],
            "torque": avg_rising["torque"]
        },
        "falling": {
            "angle": avg_falling["angle"],
            "torque": avg_falling["torque"]
        }
    }

def fit_linear_region(averaged_trial, smoothed_trials):
    """
    Fits a linear equation to the averaged rising and falling phases between 10 and 80 degrees.
    Overlays the fit and displays the equation on the plot.
    Returns a dict with slope and intercept for each phase.
    """
    results = {}

    fig, axs = plt.subplots(1, 2, figsize=(18, 6), sharey=True)
    axs[0].set_title("Flexion Phase (Rising)")
    axs[1].set_title("Extension Phase (Falling)")

    color_smoothed = "tab:red"
    color_avg = "black"
    color_fit = "limegreen"

    for i, phase in enumerate(["rising", "falling"]):
        avg_angle = averaged_trial[phase]["angle"]
        avg_torque = averaged_trial[phase]["torque"]

        # Select data in [10, 80] degrees
        mask = (avg_angle >= 10) & (avg_angle <= 80)
        x = avg_angle[mask]
        y = avg_torque[mask]

        # Fit line: torque = m * angle + b
        if len(x) < 2:
            print(f"Not enough data points for linear fit in {phase} phase.")
            m, b = np.nan, np.nan
        else:
            m, b = np.polyfit(x, y, deg=1)
            y_fit = m * x + b

        results[phase] = {"slope": m, "intercept": b}

        ax = axs[i]

        # Plot all smoothed trials in light red
        for trial in smoothed_trials:
            ax.plot(trial[phase]["angle"], trial[phase]["torque"], color=color_smoothed, alpha=0.4)

        # Plot averaged trial in black
        ax.plot(avg_angle, avg_torque, color=color_avg, linewidth=2, label="Average")

        # Plot linear fit in blue
        if len(x) >= 2:
            ax.plot(x, y_fit, color=color_fit, linewidth=2, linestyle='--', label="Linear Fit")
            # Display equation on plot
            equation_text = f"y = {m:.7f}x + {b:.7f}"
            x_pos = x[int(len(x) * 0.05)]  # near start of fit
            y_pos = y_fit[int(len(y_fit) * 0.05)]
            ax.text(x_pos, y_pos, equation_text, color=color_fit, fontsize=10, fontweight="bold")

        # Flip x-axis for falling phase
        if phase == "falling":
            ax.invert_xaxis()

        ax.set_xlabel("Angle (degrees)")
        ax.set_ylabel("Torque (Nm)")
        ax.grid(True)
        ax.legend()

    plt.tight_layout()
    plt.savefig("pictures/linear_fit_torque_vs_angle_labeled.png")
    plt.show()

    return results

def fit_piecewise_flexion(averaged_trial, smoothed_trials):
    """
    Fits two separate linear equations to the rising (flexion) phase:
    - One from 10 to 35 degrees
    - One from 35 to 80 degrees
    Returns both equations and plots them with the average and raw smoothed data.
    """
    phase = "rising"
    angle = averaged_trial[phase]["angle"]
    torque = averaged_trial[phase]["torque"]

    # Masks for each segment
    mask_10_35 = (angle >= 10) & (angle <= 35)
    mask_35_80 = (angle >= 35) & (angle <= 80)

    x1, y1 = angle[mask_10_35], torque[mask_10_35]
    x2, y2 = angle[mask_35_80], torque[mask_35_80]

    # Fit lines
    m1, b1 = np.polyfit(x1, y1, deg=1) if len(x1) >= 2 else (np.nan, np.nan)
    m2, b2 = np.polyfit(x2, y2, deg=1) if len(x2) >= 2 else (np.nan, np.nan)

    y1_fit = m1 * x1 + b1 if not np.isnan(m1) else None
    y2_fit = m2 * x2 + b2 if not np.isnan(m2) else None

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.title("Piecewise Linear Fit – Flexion Phase (Rising)")

    # Plot smoothed raw data
    for trial in smoothed_trials:
        plt.plot(trial["rising"]["angle"], trial["rising"]["torque"], color="tab:red", alpha=0.3)

    # Plot average
    plt.plot(angle, torque, color="black", linewidth=2, label="Average")

    # Plot fits
    if y1_fit is not None:
        plt.plot(x1, y1_fit, color="limegreen", linestyle='--', linewidth=2,
                 label=f"Fit 10–35°: y = {m1:.7f}x + {b1:.7f}")
        plt.text(x1[1], y1_fit[1], f"{m1:.7f}x + {b1:.7f}", color="limegreen", fontsize=10, fontweight="bold")
    if y2_fit is not None:
        plt.plot(x2, y2_fit, color="cyan", linestyle='--', linewidth=2,
                 label=f"Fit 35–80°: y = {m2:.7f}x + {b2:.7f}")
        plt.text(x2[1], y2_fit[1], f"{m2:.7f}x + {b2:.7f}", color="cyan", fontsize=10, fontweight="bold")

    plt.xlabel("Angle (degrees)")
    plt.ylabel("Torque (Nm)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("pictures/piecewise_flexion_fit.png")
    plt.show()

    return {
        "segment_10_35": {"slope": m1, "intercept": b1},
        "segment_35_80": {"slope": m2, "intercept": b2}
    }

def fit_piecewise_flexion_and_extension(averaged_trial, smoothed_trials):
    """
    Fits two linear equations to the flexion (rising) phase:
    - One from 10 to 35 degrees
    - One from 35 to 80 degrees
    Also fits one linear equation to the extension (falling) phase from 10 to 80 degrees.
    Plots all smoothed trials, the average, and the fitted lines.
    """
    results = {}

    fig, axs = plt.subplots(1, 2, figsize=(18, 6), sharey=True)
    axs[0].set_title("Piecewise Linear Fit – Flexion Phase (Rising)")
    axs[1].set_title("Linear Fit – Extension Phase (Falling)")

    # === Flexion (Rising) ===
    phase = "rising"
    angle = averaged_trial[phase]["angle"]
    torque = averaged_trial[phase]["torque"]

    # Segment masks
    mask_10_35 = (angle >= 10) & (angle <= 35)
    mask_35_80 = (angle >= 35) & (angle <= 80)

    x1, y1 = angle[mask_10_35], torque[mask_10_35]
    x2, y2 = angle[mask_35_80], torque[mask_35_80]

    m1, b1 = np.polyfit(x1, y1, deg=1) if len(x1) >= 2 else (np.nan, np.nan)
    m2, b2 = np.polyfit(x2, y2, deg=1) if len(x2) >= 2 else (np.nan, np.nan)

    y1_fit = m1 * x1 + b1 if not np.isnan(m1) else None
    y2_fit = m2 * x2 + b2 if not np.isnan(m2) else None

    # Plot all smoothed flexion trials
    for trial in smoothed_trials:
        axs[0].plot(trial["rising"]["angle"], trial["rising"]["torque"], color="tab:red", alpha=0.3)

    # Plot average
    axs[0].plot(angle, torque, color="black", linewidth=2, label="Average")

    # Plot linear fits
    if y1_fit is not None:
        axs[0].plot(x1, y1_fit, color="limegreen", linestyle='--', linewidth=2,
                    label=f"Fit 10–35°: y = {m1:.7f}x + {b1:.7f}")
        axs[0].text(x1[1], y1_fit[1], f"{m1:.7f}x + {b1:.7f}", color="limegreen", fontsize=10, fontweight="bold")
    if y2_fit is not None:
        axs[0].plot(x2, y2_fit, color="cyan", linestyle='--', linewidth=2,
                    label=f"Fit 35–80°: y = {m2:.7f}x + {b2:.7f}")
        axs[0].text(x2[1], y2_fit[1], f"{m2:.7f}x + {b2:.7f}", color="cyan", fontsize=10, fontweight="bold")

    axs[0].set_xlabel("Angle (degrees)")
    axs[0].set_ylabel("Torque (Nm)")
    axs[0].grid(True)
    axs[0].legend()

    results["segment_10_35"] = {"slope": m1, "intercept": b1}
    results["segment_35_80"] = {"slope": m2, "intercept": b2}

    # === Extension (Falling) ===
    phase = "falling"
    angle_ext = averaged_trial[phase]["angle"]
    torque_ext = averaged_trial[phase]["torque"]

    mask_ext = (angle_ext >= 10) & (angle_ext <= 80)
    x_ext = angle_ext[mask_ext]
    y_ext = torque_ext[mask_ext]

    m_ext, b_ext = np.polyfit(x_ext, y_ext, deg=1) if len(x_ext) >= 2 else (np.nan, np.nan)
    y_ext_fit = m_ext * x_ext + b_ext if not np.isnan(m_ext) else None

    # Plot all smoothed extension trials
    for trial in smoothed_trials:
        axs[1].plot(trial["falling"]["angle"], trial["falling"]["torque"], color="tab:red", alpha=0.3)

    # Plot average
    axs[1].plot(angle_ext, torque_ext, color="black", linewidth=2, label="Average")

    # Plot fit
    if y_ext_fit is not None:
        axs[1].plot(x_ext, y_ext_fit, color="limegreen", linestyle='--', linewidth=2,
                    label=f"Fit 10–80°: y = {m_ext:.7f}x + {b_ext:.7f}")
        axs[1].text(x_ext[1], y_ext_fit[1], f"{m_ext:.7f}x + {b_ext:.7f}", color="limegreen", fontsize=10, fontweight="bold")

    axs[1].invert_xaxis()
    axs[1].set_xlabel("Angle (degrees)")
    axs[1].set_ylabel("Torque (Nm)")
    axs[1].grid(True)
    axs[1].legend()

    results["falling_10_80"] = {"slope": m_ext, "intercept": b_ext}

    plt.tight_layout()
    plt.savefig("pictures/piecewise_flexion_and_extension_fit.png")
    plt.show()

    return results

def main():
    all_trials = read_all_trials()
    # plot_all_data(all_trials)
    # plot_tibiofemoral_data(all_trials)
    flexion_split_trials = split_data_by_flexion(all_trials)
    filtered_trials = process_split_trials(flexion_split_trials)
    plot_torque_vs_angle(filtered_trials, "pictures/torque_vs_angle_normalized.png")
    smooth_trials = smooth_signals(filtered_trials, window_size=250)
    plot_torque_vs_angle(smooth_trials, "pictures/torque_vs_angle_smooth.png")
    average_trial = average_trials(smooth_trials)
    equations = fit_linear_region(average_trial, smooth_trials)
    piecewise_results = fit_piecewise_flexion_and_extension(average_trial, smooth_trials)
    
if __name__ == "__main__":
    main()