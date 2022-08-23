import os
import numpy as np
import torch
from torch.utils import data
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Normalize


class Horus(data.Dataset):

    def __init__(self, root, split, joint_transform=None, transform="tensor", segformer=None):
        super(Horus, self).__init__()
        self.root = root
        self.split = split
        self.images_base = os.path.join(self.root, self.split, "images")
        self.items_list = self.get_images_list(self.images_base)

        self.joint_transform = joint_transform
        self.transform = transform
        mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        image_transforms_list = [ToTensor(), Normalize(*mean_std)]
        self.image_transforms = Compose(image_transforms_list)

        self.segformer = segformer

    def get_images_list(self, images_base):
        items_list = []
        for root, dirs, files in os.walk(images_base, topdown=True):

            for name in files:
                if name.endswith(".jpg"):
                    img_file = os.path.join(root, name)
                    items_list.append({
                        "image": img_file,
                        "name": name
                    })
        return items_list

    def __getitem__(self, index):
        image_path = self.items_list[index]["image"]
        name = self.items_list[index]["name"]
        image = Image.open(image_path).convert("RGB")
        width, height = image.size

        if self.joint_transform:
            image = self.joint_transform(image)

        if self.transform == "tensor":
            image = self.image_transforms(image)

        elif self.transform == "ndarray":
            image = np.asarray(image, dtype=np.uint8)

        if self.segformer:
            encoded_inputs = self.segformer(image, None, return_tensors="pt")

            for k, v in encoded_inputs.items():
                encoded_inputs[k].squeeze_()  # remove batch dimension

            return encoded_inputs, name, width, height

        return image, name, width, height

    def __len__(self):
        return len(self.items_list)


def main():
    from transformers import SegformerFeatureExtractor
    feature_extractor = SegformerFeatureExtractor(reduce_labels=True)

    dataset = Horus("./dataset", "2022-08-17", segformer=feature_extractor)
    print(len(dataset))
    dataiter = iter(dataset)
    encoded_inputs, name, width, height = next(dataiter)
    print(encoded_inputs["pixel_values"].shape)


if __name__ == "__main__":
    main()
