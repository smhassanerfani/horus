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
        Transformation matrix of shape `(3, 4)`.
    """
    retval, Rt, inliers = cv2.estimateAffine3D(lidar_coordinates, total_station_coordinates)
    return Rt

def voxel_grid_sampling(fname, voxel_size=0.02):
    """a voxel grid structure is created and a representative data point is selected.
    link: towardsdatascience.com/how-to-automate-lidar-point-cloud-processing-with-python-a027454a536c
    parameters
    ----------
    fname: str
        Point cloud filename to read.
    voxel_size: float
        The length, width and height of the voxel (which is equal) in meter.
    
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

    return np.array(grid_candidate_center), np.array(grid_barycenter)
