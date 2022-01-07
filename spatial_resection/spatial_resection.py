import cv2
import numpy as np
from utils import aruco_3D_to_dict, find_aruco_markers, plot_aruco_markers, load_coefficients

def pair_coordinates(image, aruco_3D_coordinates, plot=False):
    bboxs, ids = find_aruco_markers(image, marker_size=5, total_markers=250, draw=True)

    if plot:
        plot_aruco_markers(image, bboxs, ids)

    ids = np.squeeze(ids)
    img_pts = []
    obj_pts = []

    for idx, (id, bbox) in enumerate(zip(ids, bboxs)):
        img_pts.append([bbox[0][0][0], bbox[0][0][1]])
        obj_pts.append(aruco_3D_coordinates[int(id)])

    return ids, np.array(img_pts, dtype=np.float32), np.array(obj_pts, dtype=np.float32)

def spatial_resection(camera_properties, image_points, object_points):    
    camera_matrix, dist_coeffs, _, _ = load_coefficients(camera_properties)

    success, rotation_vector, translation_vector = cv2.solvePnP(object_points, image_points, camera_matrix, dist_coeffs, cv2.SOLVEPNP_ITERATIVE)

    print(f"Rotation Vector:\n {rotation_vector}")
    print(f"Translation Vector:\n {translation_vector}")
    return rotation_vector, translation_vector

def main(image_path, aruco_coordinates_path, camera_config_path):
    aruco_3D_dict = aruco_3D_to_dict(aruco_coordinates_path)
    image = cv2.imread(image_path)
    ids, image_points, object_points = pair_coordinates(image, aruco_3D_dict, plot=False)
    
    if len(ids) >= 6:
        camera_properties = camera_config_path
        spatial_resection(camera_properties, image_points, object_points)

if __name__ == "__main__":
    main("./2021-12-02-1504.jpg", "./aruco_object_coordinates.txt", "./camera_config.yml")