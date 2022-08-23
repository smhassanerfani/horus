import cv2
from cv2 import aruco as aruco
import numpy as np
import os

def aruco_3D_to_dict(object_points):
    object_ndarray = np.loadtxt(object_points, delimiter=",")

    return {int(array[0]): array[1:].tolist() for array in object_ndarray}

def load_coefficients(path):
    '''Loads camera matrix and distortion coefficients.'''
    # FILE_STORAGE_READ
    cv_file = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)

    camera_matrix = cv_file.getNode('K').mat()
    dist_coefs = cv_file.getNode('D').mat()
    rot_vecs = cv_file.getNode('R').mat()
    tran_vecs = cv_file.getNode('T').mat()

    cv_file.release()
    return camera_matrix, dist_coefs, rot_vecs, tran_vecs

def plot_aruco_markers(img, bboxs, ids):
    if (ids is not None):
        for bbox, id in zip(bboxs, ids):
            cv2.putText(img, str(id), (int(bbox[0][0][0]), int(bbox[0][0][1])), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 255), 1)

    resized_img = cv2.resize(img, (1920, 1080))
    cv2.imshow("Image", resized_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def find_aruco_markers(img, marker_size=5, total_markers=50, draw=True):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    key = getattr(aruco, f"DICT_{marker_size}X{marker_size}_{total_markers}")
    aruco_dict = aruco.Dictionary_get(key)
   
    aruco_params = aruco.DetectorParameters_create()
    bboxs, ids, rejected = aruco.detectMarkers(gray_img, aruco_dict, parameters=aruco_params)

    if draw and len(bboxs) != 0:
        aruco.drawDetectedMarkers(img, bboxs)

    return bboxs, ids


def pair_coordinates(image, aruco_3D_coordinates, plot=False, save_path="./"):
    bboxs, ids = find_aruco_markers(image, marker_size=5, total_markers=250, draw=True)

    if plot:
        plot_aruco_markers(image, bboxs, ids)

        x = input('Do you want to save:')
        if (x.lower() == 'yes') or (x.lower() == 'y'):
            file_name = input('Please enter the file name:')
            save_path = os.path.join(save_path, f"{file_name}.png")
            cv2.imwrite(save_path, image)

    ids = np.squeeze(ids)
    useless_idxs = []
    img_pts = []
    obj_pts = []


    for idx, (id, bbox) in enumerate(zip(ids, bboxs)):
        if id in aruco_3D_coordinates.keys():
            img_pts.append([bbox[0][0][0], bbox[0][0][1]])
            obj_pts.append(aruco_3D_coordinates[int(id)])
        else:
            useless_idxs.append(idx)

    return np.delete(ids, useless_idxs), np.array(img_pts, dtype=np.float32), np.array(obj_pts, dtype=np.float32)

def spatial_resection(camera_properties, image_points, object_points):
    """For further information visit following website:
    learnopencv.com/head-pose-estimation-using-opencv-and-dlib/

    Parameters
    ----------
    camera_properties: yml
        Intrinsic and extrinsic parameters of the camera.
    image_points: np.ndarray
        2D coordinates of ArUco markers.
    object_points: np.ndarray
        3D locations of the same ArUco markers.
    
    Returns
    -------
    rotation_vector: np.ndarray
        Rotation vector.
    translation_vector: np.ndarray
        Translation vector.
    """    
    
    # Intrinsic parameters of the camera, distortion coefficients
    camera_matrix, dist_coeffs, rvec, tvec = load_coefficients(camera_properties)
    
    # Initial approximations of the rotation and translation vectors
    rvec = np.array([1.34596194, -1.64632315, 1.0735686])
    tvec = np.array([1.54266924, 1.94512742, 4.21663091])

    # The Direct Linear Transform (DLT) solution followed by Levenberg-Marquardt optimization
    success, rotation_vector, translation_vector = cv2.solvePnP(
        object_points,
        image_points,
        camera_matrix,
        dist_coeffs,
        rvec,
        tvec,
        useExtrinsicGuess=True,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    print(f"Rotation Vector:\n {rotation_vector}")
    print(f"Translation Vector:\n {translation_vector}")
    return camera_matrix, dist_coeffs, rotation_vector, translation_vector

def perspective_projection(point_cloud, rvec, tvec, camera_matrix, dist_coeffs):
    """Projects 3D points to an image plane.
    Parameters
    ----------
    point_cloud: np.ndarray
        Transformed and sub-sampled point cloud of shape (N x 3).
    rvec: np.ndarray
        The rotation vector which is estimated by `spatial resection`.
    tvec: np.ndarray
        The translation vector which is estimated by `spatial resection`.
    camera_matrix: np.ndarray
        Camera intrinsic matrix. 
    dist_coeffs: np.ndarray
        Input vector of distortion coefficients.
    
    Returns
    -------
    projected_points: np.ndarray
        Output array of image points of shape (N x 2).
    """
    
    point_cloud2D = []
    for coordinate in point_cloud:
        (point2D, jacobian) = cv2.projectPoints(np.array(coordinate), rvec, tvec, camera_matrix, dist_coeffs)
        point_cloud2D.append(point2D)

    return np.array(np.squeeze(point_cloud2D))

def main():
    cap = cv2.VideoCapture(0)

    while (True):
        success, img = cap.read()
        bboxs, ids = find_aruco_markers(img)

        if (ids is not None):
            for bbox, id in zip(bboxs, ids):
                cv2.putText(img, str(id), (bbox[0][0][0], bbox[0][0][1]), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 255), 1)

        cv2.imshow("Image", img)
        if  cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
