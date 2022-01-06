import cv2
from camera_calibration import calibrate_chessboard
from utils import save_coefficients, load_coefficients


# exit()

### Undistortion ###
img_path = "./images/img-06.jpeg"
mtx, dist, rvecs, tvecs = load_coefficients('./camera_config.yml')
img = cv2.imread(img_path)

h,  w = img.shape[:2]
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

# undistort
dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

# crop the image
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv2.imwrite('./images/ccimg-06.png', dst)
print("undistorted images is dumped!")

if __name__ == "__main__":
    ret, mtx, dist, rvecs, tvecs = calibrate_chessboard(mode="RT")
    save_coefficients(mtx, dist, rvecs, tvecs, './camera_config.yml')