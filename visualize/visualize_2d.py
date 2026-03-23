import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import matplotlib.cm as cm

import process_data

NUM_CHIPS = 6
SAMPLE = "visualize/generated_data.txt"

# Layout settings
CHIP_WIDTH = 2  # how many columns in each PCAP
CHIP_HEIGHT = 3  # how many rows in each PCAP
GAP = 0.4

def initialize_PCAP_visualization(num_chips):
    fig_width = num_chips * (CHIP_WIDTH + GAP)
    # fig, ax = plt.subplots(figsize=(fig_width * 1.5, 6))
    fig, (ax, angle_ax) = plt.subplots(
        1, 2,
        figsize=(fig_width * 1.8, 6),
        gridspec_kw={'width_ratios': [4, 1]}
    )

    norm = Normalize(vmin=0, vmax=100)
    cmap = cm.viridis

    cell_rects = {}
    cell_texts = {}

    for chip_num in range(num_chips):
        x0 = chip_num * (CHIP_WIDTH + GAP)
        y0 = 0

        # Storage for this chip
        cell_rects[chip_num] = []
        cell_texts[chip_num] = []

        # Chip label above rectangle
        ax.text(
            x0 + CHIP_WIDTH / 2,
            y0 + CHIP_HEIGHT + 0.2,
            f"Chip {chip_num}",
            ha='center',
            va='bottom',
            fontsize=12,
            fontweight='bold'
        )

        # Fill each cell with initial value 0.0
        for r in range(3):
            for c in range(2):
                value = 0.0
                cell_x0 = x0 + c
                cell_y0 = y0 + (2 - r)

                rect = Rectangle(
                    (cell_x0, cell_y0),
                    1,
                    1,
                    facecolor=cmap(norm(value)),
                    edgecolor='black',
                    linewidth=1
                )
                ax.add_patch(rect)
                cell_rects[chip_num].append(rect)

                txt = ax.text(
                    cell_x0 + 0.5,
                    cell_y0 + 0.5,
                    f"{value:.1f}",
                    ha='center',
                    va='center',
                    fontsize=11,
                    color='white'
                )
                cell_texts[chip_num].append(txt)

        # Outer chip rectangle
        outer = Rectangle(
            (x0, y0),
            CHIP_WIDTH,
            CHIP_HEIGHT,
            fill=False,
            edgecolor='black',
            linewidth=2
        )
        ax.add_patch(outer)

    # One shared legend/colorbar
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.035, pad=0.04)
    cbar.set_label("Raw Pressure Value")
    cbar.set_ticks([0, 20, 40, 60, 80, 100])

    # Clean up axes
    ax.set_xlim(-0.2, num_chips * (CHIP_WIDTH + GAP))
    ax.set_ylim(-0.2, CHIP_HEIGHT + 0.6)
    ax.set_aspect('equal')
    ax.axis('off')

    angle_ax.set_title("Knee Position")
    angle_ax.set_xlim(-1.5, 1.5)
    angle_ax.set_ylim(0, 10)
    angle_ax.set_aspect('equal')
    angle_ax.grid(True)
    angle_ax.axis('off')

    # Fixed joint locations for initial straight leg
    hip = (0.0, 9.0)
    knee = (0.0, 5.0)
    ankle = (0.0, 1.0)

    # Upper leg: hip -> knee
    thigh_line, = angle_ax.plot(
        [hip[0], knee[0]],
        [hip[1], knee[1]],
        linewidth=4,
        color='blue'
    )

    # Lower leg: knee -> ankle
    shin_line, = angle_ax.plot(
        [knee[0], ankle[0]],
        [knee[1], ankle[1]],
        linewidth=4,
        color='red'
    )

    # Optional: show the knee joint
    knee_point, = angle_ax.plot(knee[0], knee[1], marker='o', markersize=6, color='black')
    return fig, ax, angle_ax, thigh_line, shin_line, knee_point, cell_rects, cell_texts, cmap, norm

def update_chip_visualization(chip, s0, s1, s2, s3, s4, s5, cell_rects, cell_texts, cmap, norm):
    values = [s0, s1, s2, s3, s4, s5]

    for i, value in enumerate(values):
        cell_rects[chip][i].set_facecolor(cmap(norm(value)))
        cell_texts[chip][i].set_text(f"{value:.1f}")

if __name__ == "__main__":
    fig, ax, angle_ax, thigh_line, shin_line, knee_point, cell_rects, cell_texts, cmap, norm = initialize_PCAP_visualization(NUM_CHIPS)

    # READ DATA - TO SIMULATE READING FROM A SERIAL MONITOR
    with open(SAMPLE, 'r', encoding='utf-8') as file:
        content = file.read()
    lines = content.split('\n')

    for line in lines:
        try:
            chip, s0, s1, s2, s3, s4, s5, pos = process_data.process_line(line)
        except:
            print("[ERROR] Failed to read line.")
            continue
        update_chip_visualization(
            chip, s0, s1, s2, s3, s4, s5,
            cell_rects, cell_texts, cmap, norm
        )
        plt.pause(0.001)  # NOTE DELETE ME WHEN ARDUINO

    plt.tight_layout()
    plt.show()