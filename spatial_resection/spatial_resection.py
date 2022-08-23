import os
import cv2
import numpy as np
import argparse
from utils import aruco_3D_to_dict, pair_coordinates, spatial_resection, perspective_projection


def get_arguments(
        images_path="../machine_learning/dataset/deployment/2022-08-19/images",
        camera_config_path="./camera_config_beena.yml",
        aruco_coordinates_path="./aruco_markers_3D.txt",
        point_cloud3D="./total_levee_gbc.xyz",
        save_path="../machine_learning/results/deployment/2022-08-19",
        plot_permission=True
        ):
    
    parser = argparse.ArgumentParser(description=f"Spatial resection on captured images.")
    parser.add_argument("--images-path", type=str, default=images_path,
                        help="Path to the file of the captured image.")
    parser.add_argument("--camera-config-path", type=str, default=camera_config_path,
                        help="Path to the .yml file of the camera config.")
    parser.add_argument("--aruco-coordinates-path", type=str, default=aruco_coordinates_path,
                        help="Path to the .txt file of the 3D aruco markers coordinates.")
    parser.add_argument("--point-cloud3D", type=str, default=point_cloud3D,
                        help="Path to the .xyz file of the 3D point cloud.")
    parser.add_argument("--save-path", type=str, default=save_path,
                        help="Path to save results.")
    parser.add_argument("--plot-permission", default=plot_permission, action="store_true",
                        help="Permission for plotting.")

    return parser.parse_args()


def main(args):

    for root, dir, files in os.walk(args.images_path, topdown=True):
        for file in files:
            if file.endswith(".jpg"):

                image = cv2.imread(os.path.join(root, file))
                aruco_3D_dict = aruco_3D_to_dict(args.aruco_coordinates_path)

                ids, image_points, object_points = pair_coordinates(image, aruco_3D_dict,
                                                                    plot=args.plot_permission, save_path=args.save_path)

                if len(ids) >= 6:

                    camera_matrix, dist_coeffs, rvec, tvec = spatial_resection(args.camera_config_path,
                                                                               image_points, object_points)

                    point_cloud3D = np.loadtxt(args.point_cloud3D, delimiter=",")

                    if point_cloud3D.shape[1] > 3:
                        point_cloud3D = point_cloud3D[:, :3]
                    
                    point_cloud2D = perspective_projection(point_cloud3D, rvec, tvec, camera_matrix, dist_coeffs)

                    save_path = os.path.join(args.save_path, f'{file.split(".")[0]}.pts')
                    np.savetxt(save_path, point_cloud2D, delimiter=',')

                    if args.plot_permission:
                        for val in point_cloud2D:
                            cv2.putText(image, ".", (int(val[0]), int(val[1])), cv2.FONT_HERSHEY_PLAIN, 1, (75, 20, 20), 1)


                        resized_image = cv2.resize(image, (1920, 1080))
                        cv2.imshow("Image", resized_image)
                        cv2.waitKey(0)
                        cv2.destroyAllWindows()
                        
                        x = input('Do you want to save:')
                        if (x.lower() == 'yes') or (x.lower() == 'y'):
                            file_name = input('Please enter the file name:')
                            save_path = os.path.join(args.save_path, f"{file_name}.png")
                            cv2.imwrite(save_path, image)

                    print(f"Point Cloud 2D has been created based on image: {file}")
                    exit()

if __name__ == "__main__":
    args = get_arguments()
    main(args)