import os
import matplotlib.pyplot as plt
import json
import numpy as np
from sklearn.decomposition import PCA
import pickle
import laspy
import sys

sys.path.insert(0, "../")
from forest3D import lidar_IO, treeDetector, ground_removal, utilities, detection_tools
from forest3D.pcd import read_pcd
from forest3D import detection_tools, processLidar
from forest3D import lidar_IO, treeDetector, ground_removal, utilities, detection_tools
from forest3D.object_detectors import detectObjects_yolov3 as detectObjects
import matplotlib.pyplot as plt
from pathlib import Path
import glob
import time


# creating a new directory called pythondirectory
def make_folder_if_not_exists(path):
    Path(path).mkdir(parents=True, exist_ok=True)


# Global Vars
offset = [0, 0, 0]
box_store = np.zeros((1, 4))
counter = 0
windowSize = [50, 50]
stepSize = 40
grid_size = [600, 600, 1000]
grid_res = 0.1


# Setting config
config_dict = {
    "doHisteq": [False, True],
    "raster_layers": ["vertical_density", "canopy_height"],
    "gridSize": grid_size,
    "res": grid_res,
    "support_window": [11, 7, 1],
    "normalisation": "rescale+histeq",
}
rasterMaker = treeDetector.RasterDetector(**config_dict)


# PCD file Path
pcd_file_path = (
    "/home/sohaib/Downloads/2023_08_11_02_00_57 (1)/2023_08_11_02_00_57/stations/*/"
)

# temporary folder path to store ground and read again
temp = "../../data/temp/"

# Store slice images here
raster_images_path = "../../data/dataset/2023_08_11_02_00_57/"

# Making folders if not exists
make_folder_if_not_exists(temp)
make_folder_if_not_exists(raster_images_path)

for i in glob.glob(pcd_file_path + "*.laz"):
    # PCD file
    pcd_files_path, file_name = i.rsplit("/", 1)
    print("Reading File: ", file_name, " From Path: ", pcd_file_path)
    #     pcd_file = read_pcd(pcd_files_path + "/" + file_name)
    #     xyz_data = pcd_file['points'][['x','y','z']].values
    # Laz file
    infile = laspy.read(i)
    xyz_data = np.vstack((infile.x, infile.y, infile.z)).transpose()

    # Removing Ground
    xyz_data_gr = ground_removal.removeGround(
        xyz_data, offset, thresh=2.0, proc_path=temp
    )
    ground_pts = ground_removal.load_ground_surface(
        os.path.join(temp, "_ground_surface.ply")
    )

    counter = 0
    slices = detection_tools.sliding_window_3d(
        xyz_data, stepSize=stepSize, windowSize=windowSize
    )

    for x, y, window in slices:  # stepsize 100
        # track progress
        counter += 1
        totalCount = len(
            range(int(np.min(xyz_data[:, 0])), int(np.max(xyz_data[:, 0])), stepSize)
        ) * len(
            range(int(np.min(xyz_data[:, 1])), int(np.max(xyz_data[:, 1])), stepSize)
        )
        sys.stdout.write("\r%d out of %d tiles" % (counter, totalCount))
        sys.stdout.flush()

        if window is not None:
            image, center = rasterMaker._rasterise(window, ground_pts=ground_pts)
            print(image.shape)
            image = np.uint8(image * 255)
            plt.imshow(image)
            plt.show()

            image_path = "{} slice_no_{}_{}.jpeg".format(
                file_name.split(".")[0], counter, int(time.time() * 100)
            )
            plt.imsave(raster_images_path + image_path, image)

            print("Image saved at: ", image_path)
    del infile
