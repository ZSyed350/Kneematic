import numpy as np
import open3d as o3d
from pathlib import Path

STL_FILE = "Leg_Reduced_mesh.stl"
PLANE_SIZE = 250.0


def load_mesh(mesh_path: str) -> o3d.geometry.TriangleMesh:
    """
    Load a triangle mesh from file and compute normals.
    """
    mesh = o3d.io.read_triangle_mesh(mesh_path)

    if mesh.is_empty():
        raise ValueError(f"Could not load mesh from: {mesh_path}")

    mesh.compute_vertex_normals()
    return mesh


def color_mesh(mesh: o3d.geometry.TriangleMesh, color=(0.7, 0.7, 0.7)) -> o3d.geometry.TriangleMesh:
    """
    Paint the mesh a uniform color.
    """
    mesh.paint_uniform_color(color)
    return mesh


def compute_pca(vertices: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute PCA of 3D vertices.

    Returns:
        center:      (3,) centroid of vertices
        eigenvalues: (3,) sorted descending
        eigenvectors:(3,3) columns are principal axes, sorted descending
    """
    center = vertices.mean(axis=0)
    centered = vertices - center

    cov = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    return center, eigenvalues, eigenvectors


def get_long_axis(vertices: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the principal long axis of the mesh.

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


def create_center_plane(
    center: np.ndarray,
    normal: np.ndarray,
    size: float = 40.0,
    color=(1.0, 0.0, 0.0)
) -> o3d.geometry.TriangleMesh:
    """
    Create a square plane centered at `center`, with its surface normal
    aligned to `normal`.

    Args:
        center: 3D center point of the plane
        normal: desired plane normal vector
        size:   side length of the square plane
        color:  RGB tuple in [0, 1]

    Returns:
        Open3D TriangleMesh representing the plane
    """
    normal = np.asarray(normal, dtype=float)
    normal = normal / np.linalg.norm(normal)

    # Pick a helper vector that is not parallel to the normal
    helper = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(helper, normal)) > 0.9:
        helper = np.array([0.0, 1.0, 0.0])

    # Build two orthonormal in-plane directions
    u = np.cross(normal, helper)
    u = u / np.linalg.norm(u)

    v = np.cross(normal, u)
    v = v / np.linalg.norm(v)

    half = size / 2.0

    # Square corners
    p0 = center - half * u - half * v
    p1 = center + half * u - half * v
    p2 = center + half * u + half * v
    p3 = center - half * u + half * v

    vertices = np.array([p0, p1, p2, p3], dtype=float)
    triangles = np.array([
        [0, 1, 2],
        [0, 2, 3]
    ], dtype=np.int32)

    plane = o3d.geometry.TriangleMesh()
    plane.vertices = o3d.utility.Vector3dVector(vertices)
    plane.triangles = o3d.utility.Vector3iVector(triangles)
    plane.paint_uniform_color(color)
    plane.compute_vertex_normals()

    return plane

def adjust_center_plane(
    plane: o3d.geometry.TriangleMesh,
    long_axis: np.ndarray,
    offset: float
) -> o3d.geometry.TriangleMesh:
    """
    Return a translated copy of the plane, moved along the long axis.

    Args:
        plane:      Open3D TriangleMesh plane
        long_axis:  Direction vector to translate along
        offset:     Translation amount along that axis

    Returns:
        New translated plane mesh
    """
    axis = np.asarray(long_axis, dtype=float)
    axis = axis / np.linalg.norm(axis)

    translation = axis * offset

    new_plane = o3d.geometry.TriangleMesh(plane)
    new_plane.translate(translation)
    return new_plane


def visualize_geometries(
    geometries: list,
    window_name: str = "Open3D Viewer",
    width: int = 1000,
    height: int = 800
) -> None:
    """
    Display the provided geometries in an Open3D window.
    """
    o3d.visualization.draw_geometries(
        geometries,
        window_name=window_name,
        width=width,
        height=height,
        mesh_show_back_face=True
    )

def compute_camera_vectors(long_axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute camera front and up vectors so the long axis appears vertical on screen.
    
    Returns:
        front: viewing direction
        up:    screen-up direction
    """
    up = np.asarray(long_axis, dtype=float)
    up = up / np.linalg.norm(up)

    # Pick a helper vector not parallel to up
    helper = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(helper, up)) > 0.9:
        helper = np.array([0.0, 1.0, 0.0])

    # Make front perpendicular to up
    front = np.cross(up, helper)
    front = front / np.linalg.norm(front)

    return front, up


def visualize_with_long_axis_vertical(
    geometries: list,
    center: np.ndarray,
    long_axis: np.ndarray,
    window_name: str = "Open3D Viewer",
    width: int = 1000,
    height: int = 800,
    zoom: float = 0.7
) -> None:
    """
    Visualize geometry so the long axis appears vertical in the view plane.
    """
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name, width=width, height=height)

    for g in geometries:
        vis.add_geometry(g)

    vc = vis.get_view_control()
    front, up = compute_camera_vectors(long_axis)

    vc.set_lookat(center)
    vc.set_up(up)
    vc.set_front(front)
    vc.set_zoom(zoom)

    vis.run()
    vis.destroy_window()


def get_plane_frame(
    plane: o3d.geometry.TriangleMesh
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    """
    Extract plane center, local in-plane axes, normal, and side lengths.

    Returns:
        center: plane center
        u: first in-plane unit vector
        v: second in-plane unit vector
        n: plane normal unit vector
        side_u: plane side length along u
        side_v: plane side length along v
    """
    vertices = np.asarray(plane.vertices)

    if vertices.shape[0] < 4:
        raise ValueError("Plane must have at least 4 vertices.")

    p0, p1, p2, p3 = vertices[:4]

    center = vertices.mean(axis=0)

    u = p1 - p0
    side_u = np.linalg.norm(u)
    u = u / side_u

    v = p3 - p0
    side_v = np.linalg.norm(v)
    v = v / side_v

    n = np.cross(u, v)
    n = n / np.linalg.norm(n)

    return center, u, v, n, side_u, side_v

def create_cut_box_from_plane(
    plane: o3d.geometry.TriangleMesh,
    normal_depth: float,
    in_plane_scale: float = 1.0,
    color=(1.0, 0.0, 0.0)
) -> o3d.geometry.OrientedBoundingBox:
    """
    Create an oriented box centered on the plane.

    Args:
        plane: adjusted center plane
        normal_depth: thickness of box along plane normal
        in_plane_scale: scale factor for plane width/height
        color: debug visualization color

    Returns:
        OrientedBoundingBox
    """
    center, u, v, n, side_u, side_v = get_plane_frame(plane)

    extent = np.array([
        side_u * in_plane_scale,
        side_v * in_plane_scale,
        normal_depth
    ], dtype=float)

    R = np.column_stack((u, v, n))

    box = o3d.geometry.OrientedBoundingBox(center=center, R=R, extent=extent)
    box.color = color
    return box

def remove_vertices_inside_box(
    mesh: o3d.geometry.TriangleMesh,
    box: o3d.geometry.OrientedBoundingBox
) -> o3d.geometry.TriangleMesh:
    """
    Remove triangles connected to vertices inside the oriented box.
    """
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)

    inside_indices = box.get_point_indices_within_bounding_box(
        o3d.utility.Vector3dVector(vertices)
    )
    inside_indices = set(inside_indices)

    if not inside_indices:
        print("No mesh vertices found inside the box.")
        cleaned = o3d.geometry.TriangleMesh(mesh)
        cleaned.compute_vertex_normals()
        return cleaned

    keep_mask = np.array([
        not (tri[0] in inside_indices or tri[1] in inside_indices or tri[2] in inside_indices)
        for tri in triangles
    ], dtype=bool)

    new_mesh = o3d.geometry.TriangleMesh()
    new_mesh.vertices = o3d.utility.Vector3dVector(vertices.copy())
    new_mesh.triangles = o3d.utility.Vector3iVector(triangles[keep_mask].copy())

    new_mesh.remove_unreferenced_vertices()
    new_mesh.remove_degenerate_triangles()
    new_mesh.remove_duplicated_triangles()
    new_mesh.remove_duplicated_vertices()
    new_mesh.compute_vertex_normals()

    return new_mesh

def cut_mesh_with_plane_box(
    mesh: o3d.geometry.TriangleMesh,
    plane: o3d.geometry.TriangleMesh,
    normal_depth: float,
    in_plane_scale: float = 1.0
) -> tuple[o3d.geometry.TriangleMesh, o3d.geometry.OrientedBoundingBox]:
    """
    Build a box from the plane and remove mesh geometry inside it.

    Returns:
        cleaned_mesh, cut_box
    """
    cut_box = create_cut_box_from_plane(
        plane=plane,
        normal_depth=normal_depth,
        in_plane_scale=in_plane_scale
    )

    cleaned_mesh = remove_vertices_inside_box(mesh, cut_box)
    return cleaned_mesh, cut_box

def main() -> None:
    mesh_path = STL_FILE

    if not Path(mesh_path).exists():
        raise FileNotFoundError(f"File not found: {mesh_path}")

    # -----------------------------
    # Load and color mesh
    # -----------------------------
    mesh = load_mesh(mesh_path)
    mesh = color_mesh(mesh, color=(0.7, 0.7, 0.7))

    vertices = np.asarray(mesh.vertices)

    # -----------------------------
    # PCA / long axis
    # -----------------------------
    center, eigenvalues, eigenvectors = compute_pca(vertices)
    _, long_axis, axis_start, axis_end = get_long_axis(vertices)

    print("Mesh center:", center)
    print("Eigenvalues:", eigenvalues)
    print("Principal axes (columns):\n", eigenvectors)
    print("Long axis:", long_axis)

    # -----------------------------
    # Visualization helpers
    # -----------------------------
    axis_line = create_line_set(axis_start, axis_end, color=(1.0, 0.0, 0.0))

    center_plane = create_center_plane(
        center=center,
        normal=long_axis,
        size=PLANE_SIZE,
        color=(0.0, 1.0, 0.0)
    )

    adjusted_plane = adjust_center_plane(
        plane=center_plane,
        long_axis=long_axis,
        offset=100.0
    )

    # -----------------------------
    # Knee removal parameters
    # -----------------------------
    normal_depth = 40.0     # thickness of removed region through the knee
    in_plane_scale = 1.2    # enlarge plane footprint slightly

    # Build cut box from plane
    cut_box = create_cut_box_from_plane(
        plane=adjusted_plane,
        normal_depth=normal_depth,
        in_plane_scale=in_plane_scale,
        color=(0.0, 0.0, 1.0)   # blue debug box
    )

    # -----------------------------
    # Debug visualization
    # -----------------------------
    visualize_with_long_axis_vertical(
        geometries=[mesh, axis_line, adjusted_plane, cut_box],
        center=center,
        long_axis=long_axis,
        window_name="Debug Knee Removal"
    )

    # -----------------------------
    # Remove mesh inside box
    # -----------------------------
    cleaned_mesh = remove_vertices_inside_box(mesh, cut_box)
    cleaned_mesh = color_mesh(cleaned_mesh, color=(0.7, 0.7, 0.7))

    # -----------------------------
    # Final visualization
    # -----------------------------
    visualize_with_long_axis_vertical(
        geometries=[cleaned_mesh, axis_line],
        center=center,
        long_axis=long_axis,
        window_name="Mesh After Knee Removal"
    )

# def main() -> None:
#     mesh_path = "Leg_Reduced_mesh.stl"

#     if not Path(mesh_path).exists():
#         raise FileNotFoundError(f"File not found: {mesh_path}")

#     mesh = load_mesh(mesh_path)
#     mesh = color_mesh(mesh, color=(0.7, 0.7, 0.7))

#     vertices = np.asarray(mesh.vertices)

#     center, eigenvalues, eigenvectors = compute_pca(vertices)
#     _, long_axis, axis_start, axis_end = get_long_axis(vertices)

#     print("Mesh center:", center)
#     print("Eigenvalues:", eigenvalues)
#     print("Principal axes (columns):\n", eigenvectors)
#     print("Long axis:", long_axis)

#     axis_line = create_line_set(axis_start, axis_end, color=(1.0, 0.0, 0.0))
#     center_plane = create_center_plane(
#         center=center,
#         normal=long_axis,
#         size=250.0,
#         color=(0.0, 1.0, 0.0)
#     )

#     adjusted_plane = adjust_center_plane(center_plane, long_axis, offset=100.0)

#     visualize_with_long_axis_vertical(
#         geometries=[mesh, axis_line, adjusted_plane],
#         center=center,
#         long_axis=long_axis,
#         window_name="STL Viewer - Long Axis Vertical"
#     )


if __name__ == "__main__":
    main()