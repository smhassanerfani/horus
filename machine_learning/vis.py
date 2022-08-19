import os
import argparse
import numpy as np
from skimage import io
from PIL import Image
from tqdm import tqdm
from utils.palette import colorize_mask
from dataloader import Horus


def get_arguments(
        split="2022-08-17",
        padding_size=False, # (1440, 1920)
        data_directory="./dataset",
        pred_directory="./results/SegFormer-B5/2022-08-17",
        save_path="./results/SegFormer-B5/2022-08-17_vis/"
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

    dataset = Horus(args.data_directory, split=args.split, joint_transform=None, transform="ndarray")

    for image, name, width, height in tqdm(dataset):

        pred = io.imread(os.path.join(args.pred_directory, name.replace(".jpg", ".png")))

        pred = (pred*0.7 + image*0.3)
        pred = Image.fromarray(pred.astype(np.uint8))

        pred.save(f"{args.save_path}/{name[:-4]}.png")

    print("finish")


if __name__ == "__main__":
    args = get_arguments()
    main(args)