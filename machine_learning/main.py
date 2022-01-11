import cv2
import numpy as np
from utils import KNearestNeighbor

classifier = KNearestNeighbor()
classifier.train(img_pts, ids)
y_pred = classifier.predict(point_cloud2D, k=3)
# print(point_cloud[sum(y_pred, [])])
# exit(0)
# a = point_cloud2D[(y_pred == 9) | (y_pred == 3)]

for idx, vals in enumerate(y_pred):
    for val in vals:
        # yellow points
        cv2.putText(img, ".", (int(point_cloud2D[val][0]), int(point_cloud2D[val][1])), cv2.FONT_HERSHEY_PLAIN, 2, (75, 235, 235), 2)
    
    # Blue points (OpenCV `BRG` color code)
    cv2.putText(img, ".", (int(img_pts[idx, 0]), int(img_pts[idx, 1])), cv2.FONT_HERSHEY_PLAIN, 2, (255, 25, 25), 2)  

cv2.imshow("Image", img)
cv2.waitKey(0)