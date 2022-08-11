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


def get_arguments(
        model="SegFormer",
        split="val",
        num_classes=2,
        padding_size=(1440, 1920),
        batch_size=1,
        num_workers=1,
        data_directory="./dataset",
        model_config="nvidia/segformer-b0-finetuned-ade-512-512",
        restore_from="./results/SegFormer/model_weights/epoch29.pth",
        save_path="./results/SegFormer/val_visualization/",
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
    parser.add_argument("--save-path", type=str, default=save_path,
                        help="Path to save results.")
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
        os.makedirs(args.save_path)
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

            # Compute prediction and loss
            outputs = model(pixel_values=image)

            pred = interpolation(outputs.logits).squeeze(0).detach().cpu().numpy().transpose(1, 2, 0)

            pred = np.argmax(pred, axis=2)
            mask = encoded_inputs["labels"].squeeze(0).numpy()

            rgb_pred = colorize_mask(pred)
            rgb_mask = colorize_mask(mask, padding_size=args.padding_size)

            imsave('%s/%s.png' % (args.save_path, name[0][:-4]), pred)

            if args.split != "test":
                rgb_pred.save('%s/%s_color.png' % (args.save_path, name[0][:-4]))
                rgb_mask.save('%s/%s_gt.png' % (args.save_path, name[0][:-4]))

        print("finish")


if __name__ == "__main__":
    args = get_arguments()
    print(f"{args.model} is deployed on {torch.cuda.get_device_name(0)}")
    main(args)
