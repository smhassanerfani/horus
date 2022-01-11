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

def voxel_grid_sampling(fname, voxel_size=0.02, Rt=None, transformation=True):
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
    
def main():
    total_station = np.array([
        (0.000, 0.000, 0.000),
        (-0.14,0.197,0.401), 
        (-0.237,0.406,0.082), 
        (0.194,0.725,0.386),
        (0.247, 0.298, 0.728), 
        (0.409, 1.000, 0.05)
    ])

    lidar_coor = np.array([
        (-0.255972,0.78928,-0.877316),
        (-0.025972,0.707672,-0.477316),
        (0.199295,0.670137,-0.792316),
        (0.374028,1.177497,-0.492316),
        (-0.040972,1.098403,-0.147316),
        (0.584028,1.452949,-0.827316)
    ])

    Rt = transformation_matrix(lidar_coor, total_station)
    print(f"transformation_matrix is: {Rt}")
    grid_candidate_center, grid_barycenter = voxel_grid_sampling("EveryPoint.txt", voxel_size=0.02, Rt=Rt, transformation=True)
    print(f"The number of closest candidate to the barycenter of each voxel after sub-sampling: {grid_candidate_center.shape}")
    print(f"The number of barycenter of the points in voxels after sub-sampling: {grid_barycenter.shape}")

if __name__ == "__main__":
    main()