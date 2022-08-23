import os
import argparse
import numpy as np
from skimage.io import imsave
from PIL import Image
from tqdm import tqdm
import torch
import torch.backends.cudnn as cudnn
from utils.palette import colorize_mask
from dataloader import Horus
from torch.utils.data import DataLoader
from joint_transforms import Resize


def get_arguments(
        model="TransUNet",
        split="test",
        num_classes=2,
        input_size =448,
        padding_size=(1440, 1920),
        batch_size=1,
        num_workers=1,
        data_directory="./dataset",
        restore_from="./results/TransUNet/snapshots/epoch30.pth",
        save_path="./results/TransUNet/test_visualization/"
    ):
    
    parser = argparse.ArgumentParser(description=f"Testing {model} on Horus 'test' set.")
    parser.add_argument("--model", type=str, default=model,
                        help=f"Model name: {model}.")
    parser.add_argument("--split", type=str, default=split,
                        help="ATLANTIS 'test' set.")
    parser.add_argument("--num-classes", type=int, default=num_classes,
                        help="Number of classes to predict, excluding background.")
    parser.add_argument("--input-size", type=int, default=input_size,
                        help="Integer number determining the height and width of model input.")
    parser.add_argument("--padding-size", type=int, default=padding_size,
                        help="Integer number determining the height and width of model output.")
    parser.add_argument("--batch-size", type=int, default=batch_size,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--num-workers", type=int, default=num_workers,
                        help="Number of workers for multithread data loading.")
    parser.add_argument("--data-directory", type=str, default=data_directory,
                        help="Path to the directory containing the source dataset.")
    parser.add_argument("--restore-from", type=str, default=restore_from,
                        help="Where model restores parameters from.")
    parser.add_argument("--save-path", type=str, default=save_path,
                        help="Path to save results.")
    return parser.parse_args()


def main(args):
    cudnn.enabled = True
    cudnn.benchmark = True


    if args.model == "TransUNet":
        from models.vit_seg_modeling import VisionTransformer as ViT_seg
        from models.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
        config_vit = CONFIGS_ViT_seg["R50-ViT-B_16"]
        config_vit.n_classes = args.num_classes
        model = ViT_seg(config_vit, img_size=args.input_size, num_classes=config_vit.n_classes)

    saved_state_dict = torch.load(args.restore_from)
    model.load_state_dict(saved_state_dict)

    model.eval()
    model.cuda()

    try:
        os.makedirs(args.save_path)
    except FileExistsError:
        pass

    test_dataset = Horus(args.data_directory, split=args.split, joint_transform=Resize(args.input_size))
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                 num_workers=args.num_workers, pin_memory=True, drop_last=False)

    interpolation = torch.nn.Upsample(size=args.padding_size, mode="bilinear", align_corners=True)
    with torch.no_grad():
        for image, mask, name, width, height in tqdm(test_dataloader):

            # GPU deployment
            image = image.cuda()

            # Compute prediction and loss
            pred = model(image)

            pred = interpolation(pred).squeeze(0).detach().cpu().numpy().transpose(1, 2, 0)

            pred = np.argmax(pred, axis=2).astype(np.uint8)
            mask = mask.squeeze(0).numpy()

            rgb_pred = colorize_mask(pred)
            rgb_mask = colorize_mask(mask, padding_size=args.padding_size)

            imsave('%s/%s.png' % (args.save_path, name[0][:-4]), pred, check_contrast=False)

            if args.split != "test":
                rgb_pred.save('%s/%s_color.png' % (args.save_path, name[0][:-4]))
                rgb_mask.save('%s/%s_gt.png' % (args.save_path, name[0][:-4]))

        print("finish")


if __name__ == "__main__":
    args = get_arguments()
    print(f"{args.model} is deployed on {torch.cuda.get_device_name(0)}")
    main(args)
