import cv2
import numpy as np

def transformation_matrix(lidar_coordinates, total_station_coordinates):
    """Estimation of 3D transformation between lidar point cloud and total station
    surveyed coordinates.
    Parameters
    ----------
    lidar_coordinates: np.ndarray
        Lidar 3D coordinations of shape `(N x 3)`.
    total_staiton_coodinates: np.ndarray
        Total station survyed coordinates corresponding to lidar coordinates of 
        shape `(N x 3)`.
    
    Returns
    -------
    Rt: np.ndarray
        Transformation matrix of shape `(3x4)`.
    """
    retval, Rt, inliers = cv2.estimateAffine3D(lidar_coordinates, total_station_coordinates)
    return Rt

def voxel_grid_sampling(fname, voxel_size=0.02, subsampling=True, Rt=None, transformation=True):
    """a voxel grid structure is created and a representative data point is selected.
    Sub-sampling procedure is copied from:
        towardsdatascience.com/how-to-automate-lidar-point-cloud-processing-with-python-a027454a536c
    parameters
    ----------
    fname: str
        Point cloud filename to read.
    voxel_size: float
        The length, width and height of the voxel (which is equal) in meter.
    Rt: np.ndarray
        Matrix transformation of shape `(3x4)`.
    transformation: bool
        If True, also return the transformed coordinates of sub-sampled point cloud.
    
    returns
    -------
    grid_candidate_center: np.ndarray
        The closest candidate to the barycenter of each voxel.
    grid_barycenter: np.ndarray
        The barycenter of the points in voxels.
    """
    point_cloud= np.loadtxt(fname, delimiter=",")
    print(f"point cloud shape: {point_cloud.shape}")

    points = point_cloud[:, :3]
    colors = point_cloud[:, 3:]

    if subsampling:
        nb_vox = np.ceil((np.max(points, axis=0) - np.min(points, axis=0)) / voxel_size)

        non_empty_voxel_keys, inverse, nb_pts_per_voxel = np.unique(((points - np.min(points, axis=0)) // voxel_size).astype(int), axis=0, return_inverse=True, return_counts=True)
        idx_pts_vox_sorted = np.argsort(inverse)
        voxel_grid = {}
        grid_barycenter, grid_candidate_center = [], []
        last_seen = 0

        for idx, vox in enumerate(non_empty_voxel_keys):
            voxel_grid[tuple(vox)] = points[idx_pts_vox_sorted[last_seen:last_seen+nb_pts_per_voxel[idx]]]
            grid_barycenter.append(np.mean(voxel_grid[tuple(vox)],axis=0))
            grid_candidate_center.append(voxel_grid[tuple(vox)][np.linalg.norm(voxel_grid[tuple(vox)] - np.mean(voxel_grid[tuple(vox)],axis=0),axis=1).argmin()])
            last_seen+=nb_pts_per_voxel[idx]

        if transformation:
            assert Rt is not None, "There is no Transformation Matrix!"
            sampling_size = idx + 1
            ones = np.ones(sampling_size).reshape(-1, 1)
            grid_candidate_center = np.hstack((np.array(grid_candidate_center), ones))
            grid_barycenter = np.hstack((np.array(grid_barycenter), ones))
            return np.dot(Rt, grid_candidate_center.T).T, np.dot(Rt, grid_barycenter.T).T
        else:
            return np.array(grid_candidate_center), np.array(grid_barycenter)
    else:
        if transformation:
            assert Rt is not None, "There is no Transformation Matrix!"

            ones = np.ones(len(points)).reshape(-1, 1)
            points = np.hstack((np.array(points), ones))
            transformed_points = np.dot(Rt, points.T).T
            transformed_points = np.hstack((transformed_points, colors))
            return transformed_points


def main():
    total_station = np.array([
        (-1.006, 6.638, 1.574),
        (1.134, 5.872, 1.532),
        (3.255, 5.206, 1.522),
        (5.017, 4.639, 1.501),
        (6.625, 4.273, 1.954),
        (-1.183, 4.85, 0.161),
        (0.661, 4.13, 0.145),
        (2.652, 3.387, 0.166),
        (3.694, 3.219, 0.191),
        (4.988, 2.741, 0.212),
        (6.007, 3.391, 1.114),
        (4.358, 4.075, 1.158),
        (3.027, 4.498, 1.171),
        (0.936, 5.318, 1.163),
        (-1.186, 5.946, 1.059)
    ])

    lidar_coor = np.array([
        (-0.181, -3.826, 0.440),
        (-0.277, -1.642, 0.419),
        (-0.474, 0.536, 0.412),
        (-0.623, 2.345, 0.387),
        (-0.900, 3.949, 0.812),
        (1.461, -3.367, -0.916),
        (1.479, -1.439, -0.923),
        (1.403, 0.627, -0.902),
        (1.183, 1.653, -0.890),
        (1.115, 3.014, -0.877),
        (0.087, 3.711, 0.009),
        (0.098, 1.954, 0.051),
        (0.191, 0.605, 0.062),
        (0.249, -1.625, 0.046),
        (0.453, -3.764, -0.045)
    ])

    Rt = transformation_matrix(lidar_coor, total_station)
    print(f"transformation_matrix is: {Rt}")
    grid_candidate_center, grid_barycenter = voxel_grid_sampling("./left_levee_segment2.xyz", subsampling=True, voxel_size=0.05, Rt=Rt, transformation=True)
    np.savetxt('left_levee2_gcc.xyz', grid_candidate_center, delimiter=',')
    np.savetxt('left_levee2_gbc.xyz', grid_barycenter, delimiter=',')
    print(f"The number of closest candidate to the barycenter of each voxel after sub-sampling: {grid_candidate_center.shape}")
    print(f"The number of barycenter of the points in voxels after sub-sampling: {grid_barycenter.shape}")

if __name__ == "__main__":
    main()
    # import open3d as o3d
    #
    # point_cloud = np.loadtxt("../point_cloud/left_levee_gbc.xyz", delimiter=",")
    #
    # print(point_cloud.shape)
    # # points = point_cloud[:, :3]
    # # colors = point_cloud[:, 3:6]
    #
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(point_cloud)
    # # pcd.colors = o3d.utility.Vector3dVector(colors / 255)
    # # pcd.normals = o3d.utility.Vector3dVector(normals)
    #
    # o3d.visualization.draw_geometries([pcd])