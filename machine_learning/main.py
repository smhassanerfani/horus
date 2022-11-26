import cv2
import numpy as np
from scipy import stats
import argparse
from utils.knn import KNearestNeighbor
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os


def get_arguments(
        edges_path="./results/deployment/2022-11-11/edges",
        point_cloud3D="../spatial_resection/total_levee_gbc.xyz",
        point_cloud2D="./results/deployment/2022-11-11/2022-11-11-0911/2022-11-11-0911.pts",
        save_path="../machine_learning/results/deployment/2022-11-11",
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
    
    fig = plt.figure(figsize=(10, 9))

    plt.subplots_adjust(wspace=0.25, hspace=0.25)

    sub1 = fig.add_subplot(2, 2, 1)  # two rows, two columns, fist cell
    sub1.boxplot([left_shore[:, 2], right_shore[:, 2]])
    sub1.set_xticklabels(['Cam-L-BL', 'Cam-R-BL'])
    sub1.tick_params(axis='x', labelrotation=45, labelsize=12)
    sub1.set_title('Water Level Fluctuation')
    sub1.set_ylabel('Water Level (m)', fontsize=14)
    sub1.set_ylim(0, 0.80)
    sub1.grid(True)

    # Create second axes, the top-right plot
    sub2 = fig.add_subplot(2, 2, 2)  # two rows, two columns, second cell
    right_shore_3to4m = right_shore[(right_shore[:, 0] < 4.0) & (right_shore[:, 0] > 3.0)]
    left_shore_3to4m = left_shore[(left_shore[:, 0] < 4.0) & (left_shore[:, 0] > 3.0)]
    sub2.boxplot([left_shore_3to4m[:, 2], right_shore_3to4m[:, 2]])
    sub2.set_xticklabels(['Cam-L-BL', 'Cam-R-BL'])
    sub2.tick_params(axis='x', labelrotation=45, labelsize=12)
    sub2.set_title('Water Level Fluctuation (3.0 to 4.0 m)')
    sub2.set_ylabel('Water Level (m)', fontsize=14)
    sub2.set_ylim(0, 0.80)
    sub2.grid(True)

    # Create third axes, a combination of third and fourth cell
    sub3 = fig.add_subplot(2, 2, (3, 4))  # two rows, two colums, combined third and fourth cell
    sub3.scatter(right_shore[:, 0], right_shore[:, 2], color="lightcoral", label="Right Bank")
    sub3.scatter(left_shore[:, 0], left_shore[:, 2], color="teal", label="Left Bank")
    # sub3.set_title('Stage Fluctuation along the Creek')
    sub3.set_xlabel('Distance (m)', fontsize=14)
    sub3.set_ylabel('Water Level (m)')
    sub3.set_ylim(0, 0.80)
    sub3.set_xlim(1.0, 6.5)
    sub3.grid(True)
    sub3.legend()

    try:
        os.makedirs(os.path.join(save_path, "stats_plots"))
    except FileExistsError:
        pass

    save_path = os.path.join(save_path, "stats_plots", f"{file_name.split('.')[0]}.png")
    plt.savefig(save_path)


def plot_section(file_name, save_path, left_shore, right_shore):
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(12, 9))

    x = [-2.0, 2.0, 3.5, -3.5]
    y = [0.0, 0.0, 1.70, 1.70]
    axes.add_patch(patches.Polygon(xy=list(zip(x, y)), fill=False, linewidth=3))

    right_shore = right_shore[(right_shore[:, 0] < 4.0) & (right_shore[:, 0] > 3.0)]
    left_shore = left_shore[(left_shore[:, 0] < 4.0) & (left_shore[:, 0] > 3.0)]

    x_values = [-right_shore[:, 1].mean(), -left_shore[:, 1].mean()]
    y_values = [right_shore[:, 2].mean(), left_shore[:, 2].mean()]
    axes.plot(x_values, y_values, 'b', linestyle="--", linewidth=3)

    axes.set_ylim([0.0, 1.70])
    axes.tick_params(axis='y', labelsize=20)

    axes.set_xlim([-3.50, 3.50])
    axes.tick_params(axis='x', labelsize=20)

    time = file_name.split(".")[0].split("-")[-1]
    axes.set_title(f'Water Level on Aug 18, 2022 [{time[:2]}:{time[2:]}]', fontsize=24)
    axes.set_xlabel('Cross Section (m)', fontsize=24)
    axes.set_ylabel('Gauge Stage (m)', fontsize=24)
    axes.grid(True)

    try:
        os.makedirs(os.path.join(save_path, "section_plots_inverse"))
    except FileExistsError:
        pass

    save_path = os.path.join(save_path, "section_plots_inverse", f"{file_name.split('.')[0]}.png")
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
                    plot_section(file, args.save_path, left_shore, right_shore)
                    visualization(image, file, args.save_path, point_cloud2d, left_shore, right_shore)

                right_shore = right_shore[(right_shore[:, 0] < 4.0) & (right_shore[:, 0] > 3.0)]
                left_shore = left_shore[(left_shore[:, 0] < 4.0) & (left_shore[:, 0] > 3.0)]

                records[file.split(".")[0]] = [left_shore[:, 2].mean(), stats.mode(left_shore[:, 2])[0][0], right_shore[:, 2].mean(), stats.mode(right_shore[:, 2])[0][0]]

    save_path = os.path.join(args.save_path, "water_level.csv")
    with open(save_path, 'w') as f:
        for key, value in records.items():
            f.write(f"{key}, {value[0]}, {value[1]}, {value[2]}, {value[3]}\n")


if __name__ == "__main__":
    args = get_arguments()
    main(args)