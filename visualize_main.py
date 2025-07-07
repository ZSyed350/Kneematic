import open3d as o3d

PATH = "JB_FYDP.stl"

mesh = o3d.io.read_triangle_mesh(PATH)

print("Hello world")