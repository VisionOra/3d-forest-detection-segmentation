import open3d as o3d
import itertools
import numpy as np


def create_vizualizer():
    # Create visualizer object
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    render_opts = vis.get_render_option()
    render_opts.point_size = 2
    render_opts.background_color = np.zeros(3)
    return vis


def get_3d_box_from_coords(x1, x2, y1, y2, z1, z2, color=(1, 0, 0), type="oriented"):
    bounds = [[x1, x2], [y1, y2], [z1, z2]]  # set the bounds
    bbox_points = list(itertools.product(*bounds))  # create limit points
    assert type in ["axis_aligned", "oriented"]
    if type == "axis_aligned":
        bbox = o3d.geometry.AxisAlignedBoundingBox.create_from_points(
            o3d.utility.Vector3dVector(bbox_points)
        )  # create bounding box object
    else:
        bbox = o3d.geometry.OrientedBoundingBox.create_from_points(
            o3d.utility.Vector3dVector(bbox_points)
        )  # create bounding box object
    bbox.color = color
    return bbox


def adding_bb_to_pcd(
    boxes_array: np.asarray, vis: o3d.visualization.Visualizer
) -> o3d.visualization.Visualizer:
    """
    :param boxes_array: np.array Nx4:  [[ymin xmin ymax xmax]] in proportions
    :return: None
    """
    for box in boxes_array:
        box_3d = get_3d_box_from_coords(box[1], box[3], box[0], box[2], 0, 15)
        vis.add_geometry(box_3d)
    return vis


def read_bbox(path: str) -> np.array:
    """
    :param path: path to bbox file
    :return: np.array Nx4:  [[ymin xmin ymax xmax]] in proportions
    """
    box = np.loadtxt(path)
    return box


x1, x2, y1, y2 = 151.38401955, 16.74398851, 158.34157139, 21.40118313
# box = [2, 10, 4, 15, 0, 2]  # x1, x2, y1, y2, z1, z3
box_3d = get_3d_box_from_coords(x1, y1, x2, y2, 0, 2)

pcd = o3d.io.read_point_cloud(
    "/home/sohaib/Documents/kodifly/3d_forest/forest_3d_app/data/150_2-160_2_slam.pcd"
)

bbox_path = (
    "/home/sohaib/Documents/kodifly/3d_forest/forest_3d_app/data/150_2-160_2_slam.txt"
)
# Reading BBOX
bbox = read_bbox(bbox_path)
print(bbox)
# Reading PCD
new_pcd = o3d.geometry.PointCloud()
new_pcd.points = pcd.points

# Visualizing
vis = create_vizualizer()
vis.clear_geometries()

# Adding PCD to visualizer
vis.add_geometry(new_pcd)

# Adding BBOX to visualizer
vis = adding_bb_to_pcd(bbox, vis)

# Updating and running visualizer
vis.poll_events()
vis.update_renderer()
vis.run()
