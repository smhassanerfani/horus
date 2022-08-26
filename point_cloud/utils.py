import cv2
import numpy as np
import json


def read_coordinates(json_file_path="./json_file.json", shore_name="right_shore"):
    """Read the json file to generate np.ndarray `(N x 3)` for
    lidar point cloud and total station surveyed coordinates.
    Parameters
    ----------
    json_file_path: str
        Json file name to read.
    shore_name: str
        Name of creek shores, `left_shore` or `right_shore`.

    Returns
    -------
    total_station_coordinates: np.ndarray
        The total station coordinates `(N x 3)`.
    lidar_coordinates: np.ndarray
        The LiDAR  coordinates `(N x 3)`.
    """

    with open(json_file_path, 'r') as jf:
        licenses = json.load(jf)

    total_station_coordinates = list()
    lidar_coordinates = list()

    for index in licenses[shore_name]["total_station"].keys():
        total_station_coordinates.append(licenses[shore_name]["total_station"][index])
        lidar_coordinates.append(licenses[shore_name]["lidar"][index])

    return np.array(total_station_coordinates, dtype=np.float64), np.array(lidar_coordinates, dtype=np.float64)


def affine_transformation_matrix(lidar_coordinates, total_station_coordinates):
    """Estimation of 3D Affine transformation between lidar point cloud and total station
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


def voxel_grid_sampling(fname, voxel_size=0.02, Rt=None):
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
    
    returns
    -------
    grid_candidate_center: np.ndarray
        The closest candidate to the barycenter of each voxel.
    grid_barycenter: np.ndarray
        The barycenter of the points in voxels.
    """
    assert (Rt is not None or voxel_size is not None), "This function is for transformation or subsampling 3D point cloud."

    point_cloud = np.loadtxt(fname, delimiter=",")
    print(f"point cloud shape: {point_cloud.shape}")

    points = point_cloud[:, :3]
    colors = point_cloud[:, 3:]

    if voxel_size is not None:
        nb_vox = np.ceil((np.max(points, axis=0) - np.min(points, axis=0)) / voxel_size)

        non_empty_voxel_keys, inverse, nb_pts_per_voxel = np.unique(((points - np.min(points, axis=0)) // voxel_size).astype(int), axis=0, return_inverse=True, return_counts=True)
        idx_pts_vox_sorted = np.argsort(inverse)
        voxel_grid = {}
        grid_barycenter, grid_candidate_center = [], []
        last_seen = 0

        for idx, vox in enumerate(non_empty_voxel_keys):
            voxel_grid[tuple(vox)] = points[idx_pts_vox_sorted[last_seen:last_seen+nb_pts_per_voxel[idx]]]
            grid_barycenter.append(np.mean(voxel_grid[tuple(vox)], axis=0))
            grid_candidate_center.append(voxel_grid[tuple(vox)][np.linalg.norm(voxel_grid[tuple(vox)] - np.mean(voxel_grid[tuple(vox)], axis=0), axis=1).argmin()])
            last_seen += nb_pts_per_voxel[idx]

        if Rt is not None:
            assert (Rt is not None), "There is no Transformation Matrix!"
            sampling_size = idx + 1
            ones = np.ones(sampling_size).reshape(-1, 1)
            grid_candidate_center = np.hstack((np.array(grid_candidate_center), ones))
            grid_barycenter = np.hstack((np.array(grid_barycenter), ones))
            return np.dot(Rt, grid_candidate_center.T).T, np.dot(Rt, grid_barycenter.T).T
        else:
            return np.array(grid_candidate_center), np.array(grid_barycenter)
    else:
        assert (Rt is not None), "There is no Transformation Matrix!"
        ones = np.ones(len(points)).reshape(-1, 1)
        points = np.hstack((np.array(points), ones))
        transformed_points = np.dot(Rt, points.T).T
        transformed_points = np.hstack((transformed_points, colors))
        return transformed_points, None


def main():

    total_station_coordinates, lidar_coordinates = \
        read_coordinates(json_file_path="./json_file.json", shore_name="left_shore")

    Rt = affine_transformation_matrix(lidar_coordinates, total_station_coordinates)
    print(f"affine transformation_matrix is: {Rt}")

    # if voxel_size is not None: (grid_candidate_center, grid_barycenter)
    # if voxel_size is None: (grid_candidate_center, None)
    grid_transformed_pt = voxel_grid_sampling("./left_levee_segment.xyz", voxel_size=0.05, Rt=Rt)

    np.savetxt('left_levee2_gcc.xyz', grid_transformed_pt[0], delimiter=',')
    np.savetxt('left_levee2_gbc.xyz', grid_transformed_pt[1], delimiter=',')


if __name__ == "__main__":
    main()