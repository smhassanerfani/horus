import numpy as np
import argparse
import json
from PIL import Image
from os.path import join
import os


def _fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)


def _iIoU(hist):
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))


def _iACC(hist):
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.diag(hist) / hist.sum(axis=1)


def _mACC(hist):
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.diag(hist).sum() / hist.sum()


def main(args):

    # get the gt and pred
    gt_imgs = []
    pred_imgs = []

    for root, dirs, files in os.walk(args.gt_dir, topdown=True):
        for file in files:
            if file.endswith(".png"):
                gt_imgs.append(os.path.join(root, file))
                pred_imgs.append(os.path.join(args.pred_dir, file))

    with open(join(args.json_dir, 'labels_info.json'), 'r') as jf:
        info = json.load(jf)

    num_classes = len(info)
    name_classes = [entity["name"] for entity in info]

    hist = np.zeros((num_classes, num_classes))

    for ind in range(len(gt_imgs)):
        try:
            pred = np.array(Image.open(pred_imgs[ind]))
            label = np.array(Image.open(gt_imgs[ind]))

        except:
            print("We don't have the prediction of ",
                  gt_imgs[ind].split('/')[-1])
            continue

        hist += _fast_hist(label.flatten(), pred.flatten(), num_classes)

    iIoUs = _iIoU(hist)
    iACCs = _iACC(hist)

    for ind_class in range(num_classes):
        print(f"{name_classes[ind_class]}\t {100*iIoUs[ind_class]:.2f}\t {100*iACCs[ind_class]:.2f}")

    print(f"mIoU\t {100*np.nanmean(iIoUs):.2f}")
    print(f"mACC\t {100*_mACC(hist):.2f}")


def get_arguments(
        gt_dir="../dataset/masks/test",
        pred_dir="../results/PSPNet/test_visualization",
        json_dir="./"
    ):

    parser = argparse.ArgumentParser(description="Calculation of mIoU")
    parser.add_argument('-gt', '--gt-dir', default=gt_dir,
                        type=str, help='directory of inconsistency analysis data')
    parser.add_argument('-pd', '--pred-dir', default=pred_dir,
                        type=str, help='directory of annotator')
    parser.add_argument('-j', '--json-dir', default=json_dir,
                        type=str, help='labels\' ID information')
    return parser.parse_args()


if __name__ == "__main__":
    args = get_arguments()
    main(args)
