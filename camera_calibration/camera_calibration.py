import os
import numpy as np
import cv2
from io import BytesIO
from time import sleep
from picamera import PiCamera

def calibrate_chessboard(imgs_path="./images", width=6, height=9, mode="RT"):
    
    """Estimate the intrinsic and extrinsic properties of a camera.
    This code is written according to:
    1- OpenCV Documentation
    # https://docs.opencv.org/3.4/dc/dbb/tutorial_py_calibration.html
    2- Fernando Souza codes
    # https://medium.com/vacatronics/3-ways-to-calibrate-your-camera-using-opencv-and-python-395528a51615
    Parameters
    ----------
    imgs_path: str
        The path directory of chessboard images.
    width: int
        Number of corners (from top-to-bottom).
    height: int
       Number of corners (from left-to-right).
    mode: str
        Mode of calibration, when `RT` the calibration and capturing images are done
        at the same time, in a real-time manner.

    Returns
    -------
    ret: np.ndarray
        The root mean square (RMS) re-projection error.
    mtx: np.ndarray
        3x3 floating-point camera intrinsic matrix.
    dist: np.ndarray
        Vector of distortion coefficients.
    rvecs: np.ndarray
        Vector of rotation vectors (Rodrigues ) estimated for each pattern view.
    tvecs: np.ndarray
        Vector of translation vectors estimated for each pattern view.
    """

    try:
        os.makedirs(os.path.join(imgs_path, "cimages"))
    except FileExistsError:
        pass

    camera = PiCamera()
    camera.resolution = (1024, 768)

    # cunstruct a stream to hold image data
    image_stream = BytesIO()

    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((width * height, 3), np.float32)
    objp[:, :2] = np.mgrid[0:6, 0:9].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    if mode == "RT":
        counter = 0
        while counter < 20:

            camera.start_preview()
            sleep(3)
            camera.capture(image_stream, format="jpeg")

            image_stream.seek(0)
            file_bytes = np.asarray(bytearray(image_stream.read()), dtype=np.uint8)
            
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            image_stream.seek(0)
            image_stream.truncate()
            
            # Find the chess board corners
            ret, corners = cv2.findChessboardCorners(gray, (width, height), None)

            # If found, add object points, image points (after refining them)
            if ret == True:
                print(f"Image #{counter:02d} is captured.")
                cv2.imwrite(f'images/img-{counter:02d}.jpeg', img)
                
                objpoints.append(objp)
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                imgpoints.append(corners2)
                
                # Draw and display the corners exit
                cv2.drawChessboardCorners(img, (width, height), corners2, ret)
                cv2.imwrite(f'{imgs_path}/cimages/cimg-{counter:02d}.jpeg', img)
                counter += 1

        camera.stop_preview()
    
    else:
        images = []
        for root, dirs, files in os.walk(imgs_path):
            for file in files:
                if file.endswith(".jpeg"):
                    images.append(os.path.join(root, file))

        for fname in images:
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Find the chess board corners
            ret, corners = cv2.findChessboardCorners(gray, (width, height), None)

            # If found, add object points, image points (after refining them)
            if ret == True:
                print(f"Image {fname} is captured.")
                
                objpoints.append(objp)
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                imgpoints.append(corners2)
                
                # Draw and display the cornersexit
                cv2.drawChessboardCorners(img, (width, height), corners2, ret)
                cv2.imwrite(f'{imgs_path}/cimages/c{fname.split("/")[-1].split(".")[0]}.jpeg', img)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    rvecs = [arr.T for arr in rvecs]
    tvecs = [arr.T for arr in tvecs]
    rvecs = np.concatenate(rvecs, axis=0)
    tvecs = np.concatenate(tvecs, axis=0)

    print("Re-projection error estimation:")
    
    mean_error = 0
    for idx, objpoint in enumerate(objpoints):
        imgpoints2, _ = cv2.projectPoints(objpoint, rvecs[idx], tvecs[idx], mtx, dist)
        error = cv2.norm(imgpoints[idx], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        mean_error += error
    print(f"Total error: {mean_error / len(objpoints):.4f}")

    return (ret, mtx, dist, rvecs, tvecs)


