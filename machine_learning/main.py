import cv2
import numpy as np
from scipy import stats
import argparse
from utils.knn import KNearestNeighbor
import matplotlib.pyplot as plt
import os


def get_arguments(
        edges_path="./results/deployment/2022-08-19/edges",
        point_cloud3D="../spatial_resection/total_levee_gbc.xyz",
        point_cloud2D="./results/deployment/2022-08-19/2022-08-19-0906.pts",
        save_path="../machine_learning/results/deployment/2022-08-19",
        plot_permission=True
        ):

    parser = argparse.ArgumentParser(description=f"Spatial resection on captured images.")

    parser.add_argument("--edges-path", type=str, default=edges_path,
                        help="Path to the file of the estimated water contours.")
    parser.add_argument("--point-cloud3D", type=str, default=point_cloud3D,
                        help="Path to the .xyz file of the 3D point cloud.")
    parser.add_argument("--point-cloud2D", type=str, default=point_cloud2D,
                        help="Path to the .pts file of the projected 2D point cloud.")
    parser.add_argument("--save-path", type=str, default=save_path,
                        help="Path to save results.")
    parser.add_argument("--plot-permission", default=plot_permission, action="store_true",
                        help="Permission for plotting.")

    return parser.parse_args()


def estimate_stage(image, point_cloud2d, point_cloud3d):

    edge_pts = np.asarray(image[:, :, 0], dtype=np.double)
    edge_pts[:600] = 0 # keep 2D points for the region of interest

    edge_pts_idxs = np.transpose(np.nonzero(edge_pts)) # Indices of edge points that are non-zero.
    edge_pts_idxs[:, [0, 1]] = edge_pts_idxs[:, [1, 0]] # (h, w) -> (w, h)

    labels = np.arange(0, edge_pts_idxs.shape[0]) # unique label for each non-zero edge point index
    
    classifier = KNearestNeighbor()
    classifier.train(edge_pts_idxs, labels)

    # Indices of the nearest 2D point cloud coordinates to edge points
    point_cloud2d_idxs = classifier.predict(point_cloud2d, k=1)
    pt_idxs = sum(point_cloud2d_idxs, []) # flattened list
    point_cloud2d = point_cloud2d[pt_idxs]

    # The nearest 3D point cloud coordinates according to the 2D indices
    point_cloud3d = point_cloud3d[pt_idxs]
    point_cloud3d = np.concatenate((point_cloud3d, point_cloud2d, np.array(pt_idxs).reshape(-1, 1)), axis=1)

    right_shore = point_cloud3d[point_cloud3d[:, 1] < 0]
    left_shore = point_cloud3d[point_cloud3d[:, 1] > 0]

    return left_shore, right_shore

def plot_stats(file_name, save_path, left_shore, right_shore):
    
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(16,9))
    
    axes[0, 0].scatter(right_shore[:, 0], right_shore[:, 2])
    axes[0, 0].set_title('Stage Fluctuation along Right Shore')
    axes[0, 0].set_xlabel('Distance (m)')
    axes[0, 0].set_ylabel('Stage (m)')
    axes[0, 0].grid(True)
    
    axes[0, 1].boxplot(right_shore[:, 2])
    axes[0, 1].set_title('Stage Fluctuation along Right Shore')
    axes[0, 1].set_ylabel('Stage (m)')
    
    axes[1, 0].scatter(left_shore[:, 0], left_shore[:, 2])
    axes[1, 0].set_title('Stage Fluctuation along Left Shore')
    axes[1, 0].set_xlabel('Distance (m)')
    axes[1, 0].set_ylabel('Stage (m)')
    axes[1, 0].grid(True)
    
    axes[1, 1].boxplot(left_shore[:, 2])
    axes[1, 1].set_title('Stage Fluctuation along Left Shore')
    axes[1, 1].set_ylabel('Stage (m)')

    try:
        os.makedirs(os.path.join(save_path, "stats_plots"))
    except FileExistsError:
        pass

    save_path = os.path.join(save_path, "stats_plots", f"{file_name.split('.')[0]}.png")
    plt.savefig(save_path)


def visualization(image, file_name, save_path, point_cloud2d, left_shore, right_shore):

    for counter, coordinates in enumerate(left_shore):
        if counter % 50 == 0:

            # yellow points (OpenCV `BRG` color code)
            cv2.putText(image, f"({coordinates[0]:.3f}, {coordinates[2]:.3f})", (int(coordinates[3]+50), int(coordinates[4])), cv2.FONT_HERSHEY_PLAIN, 1, (75, 235, 235), 2)

            # Blue points (OpenCV `BRG` color code)
            cv2.putText(image, f"{int(coordinates[5])}", (int(coordinates[3]), int(coordinates[4])), cv2.FONT_HERSHEY_PLAIN, 1, (250, 250, 0), 2)

    for counter, coordinates in enumerate(right_shore):
        if counter % 50 == 0:

            # yellow points (OpenCV `BRG` color code)
            cv2.putText(image, f"({coordinates[0]:.3f}, {coordinates[2]:.3f})", (int(coordinates[3]-120), int(coordinates[4])), cv2.FONT_HERSHEY_PLAIN, 1, (75, 235, 235), 2)

            # Blue points (OpenCV `BRG` color code)
            cv2.putText(image, f"{int(coordinates[5])}", (int(coordinates[3]-170), int(coordinates[4])), cv2.FONT_HERSHEY_PLAIN, 1, (250, 250, 0), 2)

    try:
        os.makedirs(os.path.join(save_path, "vis_water_stage"))
    except FileExistsError:
        pass

    save_path = os.path.join(save_path, "vis_water_stage", f"{file_name.split('.')[0]}.png")
    cv2.imwrite(save_path, image)
    cv2.destroyAllWindows()


def main(args):

    point_cloud3d = np.loadtxt(args.point_cloud3D, delimiter=",")
    point_cloud2d = np.loadtxt(args.point_cloud2D, delimiter=",")

    records = dict()
    for root, dir, files in os.walk(args.edges_path, topdown=True):
        for file in files:
            if file.endswith(".png"):

                print(f"processing: {file}")
                image = cv2.imread(os.path.join(root, file))
                left_shore, right_shore = estimate_stage(image, point_cloud2d, point_cloud3d)

                if args.plot_permission:
                    plot_stats(file, args.save_path, left_shore, right_shore)
                    visualization(image, file, args.save_path, point_cloud2d, left_shore, right_shore)

                records[file.split(".")[0]] = [left_shore[:, 2].mean(), stats.mode(left_shore[:, 2])[0][0], right_shore[:, 2].mean(), stats.mode(right_shore[:, 2])[0][0]]

    save_path = os.path.join(args.save_path, "water_stage.csv")
    with open(save_path, 'w') as f:
        for key, value in records.items():
            f.write(f"{key}, {value[0]}, {value[1]}, {value[2]}, {value[3]}\n")

if __name__ == "__main__":
    args = get_arguments()
    main(args)