import open3d as o3d
import numpy as np
import os

import compute
import camera

# PATHS
STL_FILE = "models/Leg_Reduced_mesh.STL"
RAW_PCD = "models/leg_pcd.ply"
PROCESSED_PCD = "models/leg_processed.ply"
AXIS = "models/leg_axis.ply"

# PCD
NUM_POINTS = 5000
KNEE_CENTER = [464.18004978, 116.53999796, 640.04481245]

# CAMERA
ZOOM = 0.7
WIDTH = 1000
HEIGHT = 800

# POINT REMOVAL PARAMETERS
KNEE_RADIUS = 100.0
KNEE_DEPTH = 56.0
HIP_RADIUS = 500.0
HIP_DEPTH = 360.0
FOOT_RADIUS = 600.0
FOOT_DEPTH = 150.0


if __name__ == "__main__":

    if not os.path.exists(RAW_PCD):
        mesh = o3d.io.read_triangle_mesh(STL_FILE)
        pcd = mesh.sample_points_uniformly(number_of_points=NUM_POINTS)
        o3d.io.write_point_cloud(RAW_PCD, pcd, format='auto', write_ascii=False, compressed=False, print_progress=False)
    else:
        pcd = o3d.io.read_point_cloud(RAW_PCD)

    pcd.paint_uniform_color(color=(0.0, 1.0, 0.0))
    points = np.asarray(pcd.points)

    _, long_axis, start_point, end_point = compute.get_long_axis(points)
    axis_line = compute.create_line_set(start_point, end_point, color=(0.0, 0.0, 0.0))

    pcd, _, _, _ = compute.remove_points_in_cylinder_volume(
        axis_line, KNEE_CENTER, KNEE_DEPTH, KNEE_RADIUS, pcd
    )

    pcd, _, _, _ = compute.remove_points_in_cylinder_volume(
        axis_line, end_point, HIP_DEPTH, HIP_RADIUS, pcd
    )

    # Get foot
    footless_pcd, foot_mask, _, _ = compute.remove_points_in_cylinder_volume(
        axis_line, start_point, FOOT_DEPTH, FOOT_RADIUS, pcd
    )
    foot_pcd = o3d.geometry.PointCloud()
    points = np.asarray(pcd.points)
    foot_pcd.points = o3d.utility.Vector3dVector(points[foot_mask])
    foot_pcd.paint_uniform_color(color=(0.0, 1.0, 1.0))
    foot_points = np.asarray(foot_pcd.points)

    # Get foot axis
    _, foot_axis, foot_start, foot_end = compute.get_long_axis(foot_points)
    foot_line = compute.create_line_set(foot_start, foot_end, color=(0.0, 0.0, 0.0))

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="JBOSINO", width=WIDTH, height=HEIGHT)

    geometries = [footless_pcd, axis_line, foot_pcd, foot_line]
    for g in geometries:
        vis.add_geometry(g)

    vc = vis.get_view_control()
    front, up = camera.compute_camera_vectors(long_axis)

    vc.set_lookat(KNEE_CENTER)
    vc.set_up(up)
    vc.set_front(front)
    vc.set_zoom(ZOOM)

    vis.run()
    vis.destroy_window()

    if not os.path.exists(PROCESSED_PCD):
        o3d.io.write_point_cloud(PROCESSED_PCD, pcd)
    if not os.path.exists(AXIS):
        o3d.io.write_line_set(AXIS, axis_line)
    