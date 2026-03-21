import numpy as np
import open3d as o3d


def compute_pca(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute PCA.

    Returns:
        center:      (3,) centroid
        eigenvalues: (3,) sorted descending
        eigenvectors:(3,3) columns are principal axes, sorted descending
    """
    center = points.mean(axis=0)
    centered = points - center

    cov = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    return center, eigenvalues, eigenvectors


def get_long_axis(vertices: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the principal long axis.

    Returns:
        center:      mesh centroid
        long_axis:   unit vector of principal axis
        start_point: start of axis line across mesh extent
        end_point:   end of axis line across mesh extent
    """
    center, _, eigenvectors = compute_pca(vertices)
    long_axis = eigenvectors[:, 0]
    long_axis = long_axis / np.linalg.norm(long_axis)

    centered = vertices - center
    projections = centered @ long_axis

    min_proj = projections.min()
    max_proj = projections.max()

    start_point = center + min_proj * long_axis
    end_point = center + max_proj * long_axis

    return center, long_axis, start_point, end_point


def create_line_set(
    start_point: np.ndarray,
    end_point: np.ndarray,
    color=(1.0, 0.0, 0.0)
) -> o3d.geometry.LineSet:
    """
    Create a colored line between two 3D points.
    """
    points = np.array([start_point, end_point])
    lines = np.array([[0, 1]])

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector([color])

    return line_set


def remove_points_in_cylinder_volume(
    line_set: o3d.geometry.LineSet,
    center_point: np.ndarray,
    depth: float,
    radius: float,
    pcd: o3d.geometry.PointCloud,
):
    """
    Remove all points from a point cloud that fall inside a finite cylinder.

    The cylinder is defined by:
    - axis direction from the first line in line_set
    - center at center_point
    - total length = depth
    - radius = radius

    Args:
        line_set: Open3D LineSet containing at least one line.
        center_point: (3,) center of the cylinder.
        depth: Total cylinder length.
        radius: Cylinder radius.
        pcd: Open3D point cloud.

    Returns:
        filtered_pcd: point cloud with inside points removed
        inside_mask: boolean mask for points inside the cylinder
        segment_start: start point of cylinder axis
        segment_end: end point of cylinder axis
    """
    if depth <= 0:
        raise ValueError("depth must be positive")
    if radius < 0:
        raise ValueError("radius must be non-negative")

    line_points = np.asarray(line_set.points)
    line_indices = np.asarray(line_set.lines)
    cloud_points = np.asarray(pcd.points)
    center_point = np.asarray(center_point, dtype=float).reshape(3)

    if len(line_points) == 0:
        raise ValueError("line_set has no points")
    if len(line_indices) == 0:
        raise ValueError("line_set has no lines")

    i0, i1 = line_indices[0]
    p0 = line_points[i0]
    p1 = line_points[i1]

    axis_vec = p1 - p0
    axis_len = np.linalg.norm(axis_vec)
    if axis_len == 0:
        raise ValueError("line in line_set has zero length")

    axis_dir = axis_vec / axis_len
    half_depth = depth / 2.0

    # Axis segment endpoints, for reference/visualization
    segment_start = center_point - half_depth * axis_dir
    segment_end = center_point + half_depth * axis_dir

    # Vector from cylinder center to each point
    rel = cloud_points - center_point

    # Signed distance along cylinder axis
    axial = rel @ axis_dir

    # Perpendicular component to axis
    radial_vec = rel - np.outer(axial, axis_dir)
    radial_dist_sq = np.sum(radial_vec ** 2, axis=1)

    # Inside finite cylinder volume
    inside_mask = (np.abs(axial) <= half_depth) & (radial_dist_sq <= radius ** 2)

    filtered_pcd = o3d.geometry.PointCloud()
    filtered_pcd.points = o3d.utility.Vector3dVector(cloud_points[~inside_mask])

    if pcd.has_colors():
        colors = np.asarray(pcd.colors)
        filtered_pcd.colors = o3d.utility.Vector3dVector(colors[~inside_mask])

    if pcd.has_normals():
        normals = np.asarray(pcd.normals)
        filtered_pcd.normals = o3d.utility.Vector3dVector(normals[~inside_mask])

    return filtered_pcd, inside_mask, segment_start, segment_end