import os
import argparse
import numpy as np
from skimage.io import imsave
from tqdm import tqdm
import torch
import torch.backends.cudnn as cudnn
from utils.palette import colorize_mask
from dataloader import Horus
from torch.utils.data import DataLoader
import json
import cv2


def get_arguments(
        model="SegFormer",
        split="2022-08-17",
        num_classes=2,
        padding_size=(1440, 1920),
        batch_size=1,
        num_workers=1,
        data_directory="./dataset",
        model_config="nvidia/segformer-b5-finetuned-cityscapes-1024-1024",
        # model_config="nvidia/segformer-b0-finetuned-ade-512-512",
        restore_from="./results/SegFormer-B5/snapshots/epoch25.pth",
        masks_path="./results/SegFormer-B5/masks/",
        edges_path="./results/SegFormer-B5/edges/",
        LABELS_INFO="utils/labels_info.json"
        ):
    
    parser = argparse.ArgumentParser(description=f"Testing {model} on Horus 'test' set.")
    parser.add_argument("--model", type=str, default=model,
                        help=f"Model name: {model}.")
    parser.add_argument("--split", type=str, default=split,
                        help="ATLANTIS 'test' set.")
    parser.add_argument("--num-classes", type=int, default=num_classes,
                        help="Number of classes to predict, excluding background.")
    parser.add_argument("--padding-size", type=int, default=padding_size,
                        help="Integer number determining the height and width of model output.")
    parser.add_argument("--batch-size", type=int, default=batch_size,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--num-workers", type=int, default=num_workers,
                        help="Number of workers for multithread data loading.")
    parser.add_argument("--data-directory", type=str, default=data_directory,
                        help="Path to the directory containing the source dataset.")
    parser.add_argument("--model-config", type=str, default=model_config,
                        help="Where model restores configs from.")
    parser.add_argument("--restore-from", type=str, default=restore_from,
                        help="Where model restores parameters from.")
    parser.add_argument("--masks-path", type=str, default=masks_path,
                        help="Path to save masks.")
    parser.add_argument("--edges-path", type=str, default=edges_path,
                        help="Path to save edges.")
    parser.add_argument("--labels-info", type=str, default=LABELS_INFO,
                        help="Path to the directory containing list and id of labels.")
    return parser.parse_args()


def main(args):
    cudnn.enabled = True
    cudnn.benchmark = True


    if args.model == "SegFormer":

        with open(args.labels_info, 'r') as jf:
            horus = json.load(jf)

        id2label = {label["id"]: label["name"] for label in horus}
        label2id = {label["name"]: label["id"] for label in horus}

        # Loading model
        if args.model == "SegFormer":
            from transformers import SegformerFeatureExtractor
            from transformers import SegformerForSemanticSegmentation
            feature_extractor = SegformerFeatureExtractor.from_pretrained(args.model_config, reduce_labels=False)
            model = SegformerForSemanticSegmentation.from_pretrained(args.model_config,
                                                                     ignore_mismatched_sizes=True,
                                                                     num_labels=args.num_classes,
                                                                     id2label=id2label, label2id=label2id)

    saved_state_dict = torch.load(args.restore_from)
    model.load_state_dict(saved_state_dict)

    model.eval()
    model.cuda()

    try:
        os.makedirs(args.masks_path)
        os.makedirs(args.edges_path)
    except FileExistsError:
        pass

    test_dataset = Horus(args.data_directory, split=args.split,
                         transform=None, segformer=feature_extractor)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                 num_workers=args.num_workers, pin_memory=True, drop_last=False)

    interpolation = torch.nn.Upsample(size=args.padding_size, mode="bilinear", align_corners=False)
    with torch.no_grad():
        for encoded_inputs, name, width, height in tqdm(test_dataloader):

            # GPU deployment
            image = encoded_inputs["pixel_values"].cuda()

            # Compute prediction
            outputs = model(pixel_values=image)
            pred = interpolation(outputs.logits).squeeze(0).detach().cpu().numpy().transpose(1, 2, 0)
            pred = np.argmax(pred, axis=2).astype(np.uint8)

            # Save prediction as RGB
            pred = colorize_mask(pred).convert('RGB')
            pred.save(f"{args.masks_path}/{name[0][:-4]}.png")

            # Save edges
            edge = cv2.Canny(np.array(pred, dtype=np.uint8), 95, 100)
            cv2.imwrite(f"{args.edges_path}/{name[0][:-4]}.png", edge)

        print("finish")


if __name__ == "__main__":
    args = get_arguments()
    print(f"{args.model} is deployed on {torch.cuda.get_device_name(0)}")
    main(args)
