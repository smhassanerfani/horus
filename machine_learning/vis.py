import os
import argparse
import numpy as np
from skimage import io
from PIL import Image
from tqdm import tqdm
from utils.palette import colorize_mask
from dataloader import Horus


def get_arguments(
        split="val",
        padding_size=False, # (1440, 1920)
        data_directory="./dataset",
        pred_directory="./results/PSPNet/val_visualization",
        save_path="./results/PSPNet/val_visualization_v2/"
    ):
    
    parser = argparse.ArgumentParser(description=f"Visualization on Horus dataset.")
    parser.add_argument("--split", type=str, default=split,
                        help="Horus 'test' set.")
    parser.add_argument("--padding-size", type=int, default=padding_size,
                        help="Integer number determining the height and width of model output.")
    parser.add_argument("--data-directory", type=str, default=data_directory,
                        help="Path to the directory containing the source dataset.")
    parser.add_argument("--pred-directory", type=str, default=pred_directory,
                        help="Path to the directory containing the prediction masks.")
    parser.add_argument("--save-path", type=str, default=save_path,
                        help="Path to save results.")
    return parser.parse_args()


def main(args):

    try:
        os.makedirs(args.save_path)
    except FileExistsError:
        pass

    dataset = Horus(args.data_directory, split=args.split, joint_transform=None, transform=False)


    for image, mask, name, width, height in tqdm(dataset):

        rgb_mask = colorize_mask(mask, padding_size=args.padding_size)
        rgb_mask = np.asarray(rgb_mask.convert('RGB'), dtype=np.uint8)

        mask_img = (rgb_mask*0.7 + image*0.3)
        mask_img = Image.fromarray(mask_img.astype(np.uint8))

        mask_prd = io.imread(os.path.join(args.pred_directory, name.replace(".jpg", ".png")))
        mask_prd = colorize_mask(mask_prd, padding_size=args.padding_size)
        mask_prd = np.asarray(mask_prd.convert('RGB'), dtype=np.uint8)

        mask_prd = (mask_prd*0.7 + image*0.3)
        mask_prd = Image.fromarray(mask_prd.astype(np.uint8))

        mask_img.save('%s/%s_gt.png' % (args.save_path, name[:-4]))
        mask_prd.save('%s/%s_pr.png' % (args.save_path, name[:-4]))

    print("finish")


if __name__ == "__main__":
    args = get_arguments()
    main(args)