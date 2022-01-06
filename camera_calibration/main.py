import cv2
from camera_calibration import calibrate_chessboard
from utils import save_coefficients, load_coefficients

def undistortion_test(image, camera_config):

    img = cv2.imread(image)
    mtx, dist, rvecs, tvecs = load_coefficients(camera_config)

    h,  w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

    # undistort
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

    # crop the image
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]
    cv2.imwrite("./images/undistorted_image.jpeg", dst)
    
    print("Undistorted images is dumped!")

if __name__ == "__main__":
    ret, mtx, dist, rvecs, tvecs = calibrate_chessboard(mode="nRT")
    save_coefficients(mtx, dist, rvecs, tvecs, './camera_config.yml')

    # image = "./images/img-06.jpeg"
    # camera_config = "./camera_config.yml"
    # undistortion_test(image, camera_config)