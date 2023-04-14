import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import rlbench.transformation as tf
import scipy

def combinePointClouds(obs):
    cloud_front = obs.front_point_cloud.reshape(-1, 3)
    cloud_overhead = obs.overhead_point_cloud.reshape(-1, 3)
    cloud_wrist = obs.front_point_cloud.reshape(-1, 3)
    cloud_left_shoulder = obs.left_shoulder_point_cloud.reshape(-1, 3)
    cloud_right_shoulder = obs.right_shoulder_point_cloud.reshape(-1, 3)
    cloud = np.concatenate((cloud_front, cloud_overhead, cloud_wrist, cloud_left_shoulder, cloud_right_shoulder))
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(cloud)
    # # mesh_box = o3d.geometry.TriangleMesh.create_box(width=1.0, height=1.0, depth=0.8)
    # # mesh_box.paint_uniform_color([0.9, 0.1, 0.1])
    # # o3d.visualization.draw_geometries([pcd, mesh_box])
    # o3d.visualization.draw_geometries([pcd])
    return cloud


def interpolate(depth):
    """
    Fill nans in depth image
    """
    # a boolean array of (width, height) which False where there are missing values and True where there are valid (non-missing) values
    mask = np.logical_not(np.isnan(depth))
    # array of (number of points, 2) containing the x,y coordinates of the valid values only
    xx, yy = np.meshgrid(np.arange(depth.shape[1]), np.arange(depth.shape[0]))
    xym = np.vstack((np.ravel(xx[mask]), np.ravel(yy[mask]))).T

    # the valid values in the first, second, third color channel,  as 1D arrays (in the same order as their coordinates in xym)
    data0 = np.ravel(depth[:, :][mask])

    # three separate interpolators for the separate color channels
    interp0 = scipy.interpolate.NearestNDInterpolator(xym, data0)

    # interpolate the whole image, one color channel at a time
    result0 = interp0(np.ravel(xx), np.ravel(yy)).reshape(xx.shape)

    return result0


def getProjectImg(cloud, target_size, img_size, camera_pos):
    z_min = 0
    cloud = cloud[(cloud[:, 2] < max(camera_pos[2], z_min + 0.05))]
    view_matrix = tf.euler_matrix(0, np.pi, 0).dot(np.eye(4))
    # view_matrix = np.eye(4)
    view_matrix[:3, 3] = [camera_pos[0], -camera_pos[1], camera_pos[2]]
    view_matrix = tf.euler_matrix(0, 0, -np.pi / 2).dot(view_matrix)
    augment = np.ones((1, cloud.shape[0]))
    pts = np.concatenate((cloud.T, augment), axis=0)
    projection_matrix = np.array([
        [1 / (target_size / 2), 0, 0, 0],
        [0, 1 / (target_size / 2), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    tran_world_pix = np.matmul(projection_matrix, view_matrix)
    pts = np.matmul(tran_world_pix, pts)
    # pts[1] = -pts[1]
    pts[0] = (pts[0] + 1) * img_size / 2
    pts[1] = (pts[1] + 1) * img_size / 2

    pts[0] = np.round_(pts[0])
    pts[1] = np.round_(pts[1])
    mask = (pts[0] >= 0) * (pts[0] < img_size) * (pts[1] > 0) * (pts[1] < img_size)
    pts = pts[:, mask]
    # dense pixel index
    mix_xy = (pts[1].astype(int) * img_size + pts[0].astype(int))
    # lexsort point cloud first on dense pixel index, then on z value
    ind = np.lexsort(np.stack((pts[2], mix_xy)))
    # bin count the points that belongs to each pixel
    bincount = np.bincount(mix_xy)
    # cumulative sum of the bin count. the result indicates the cumulative sum of number of points for all previous pixels
    cumsum = np.cumsum(bincount)
    # rolling the cumsum gives the ind of the first point that belongs to each pixel.
    # because of the lexsort, the first point has the smallest z value
    cumsum = np.roll(cumsum, 1)
    cumsum[0] = bincount[0]
    cumsum[cumsum == np.roll(cumsum, -1)] = 0
    # pad for unobserved pixels
    cumsum = np.concatenate((cumsum, -1 * np.ones(img_size * img_size - cumsum.shape[0]))).astype(int)

    depth = pts[2][ind][cumsum]
    depth[cumsum == 0] = np.nan
    depth = depth.reshape(img_size, img_size)
    # fill nans
    depth = interpolate(depth)
    return depth
