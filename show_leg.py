import open3d as o3d

# Change this to your STL file path
stl_path = "Leg_Reduced_mesh.stl"

# Load the mesh
mesh = o3d.io.read_triangle_mesh(stl_path)

# Check that it loaded properly
if mesh.is_empty():
    raise ValueError(f"Could not load mesh from: {stl_path}")

# Compute normals so lighting looks correct
mesh.compute_vertex_normals()

# Show the mesh
o3d.visualization.draw_geometries(
    [mesh],
    window_name="STL Viewer",
    width=1000,
    height=800,
    mesh_show_back_face=True
)