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

    # note we also have to specify the type to retrieve other wise we only get a
    # FileNode object back instead of a matrix
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

    cv2.imshow("Image", img)
    cv2.waitKey(0)


def find_aruco_markers(img, marker_size=5, total_markers=50, draw=True):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    key = getattr(aruco, f"DICT_{marker_size}X{marker_size}_{total_markers}")
    aruco_dict = aruco.Dictionary_get(key)
   
    aruco_params = aruco.DetectorParameters_create()
    bboxs, ids, rejected = aruco.detectMarkers(gray_img, aruco_dict, parameters=aruco_params)

    if draw and len(bboxs) != 0:
        aruco.drawDetectedMarkers(img, bboxs)

    return bboxs, ids

def main():
    cap = cv2.VideoCapture(0)

    while (True):
        success, img = cap.read()
        #img = cv2.rotate(img, cv2.ROTATE_180) #cv2.ROTATE_90_COUNTERCLOCKWISE
        #img = cv2.flip(img, 0)
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
