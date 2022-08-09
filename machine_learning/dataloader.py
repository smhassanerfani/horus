import os
import numpy as np
import torch
from torch.utils import data
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Normalize


class MaskToTensor(object):
    def __call__(self, img):
        return torch.from_numpy(np.array(img, dtype=np.int32)).long()


class Horus(data.Dataset):

    def __init__(self, root, split, joint_transform=None, transform=True):
        super(Horus, self).__init__()
        self.root = root
        self.split = split
        self.images_base = os.path.join(self.root, "images", self.split)
        self.masks_base = os.path.join(self.root, "masks", self.split)
        self.items_list = self.get_images_list(self.images_base, self.masks_base)

        self.joint_transform = joint_transform
        self.transform = transform
        mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        image_transforms_list = [ToTensor(), Normalize(*mean_std)]
        self.image_transforms = Compose(image_transforms_list)
        self.label_transforms = MaskToTensor()


    def get_images_list(self, images_base, masks_base):
        items_list = []
        for root, dirs, files in os.walk(images_base, topdown=True):

            for name in files:
                if name.endswith(".jpg"):
                    mask_name = name.replace(".jpg", ".png")
                    img_file = os.path.join(root, name)
                    lbl_file = os.path.join(masks_base, mask_name)
                    items_list.append({
                        "image": img_file,
                        "label": lbl_file,
                        "name": name
                    })
        return items_list

    def __getitem__(self, index):
        image_path = self.items_list[index]["image"]
        label_path = self.items_list[index]["label"]
        name = self.items_list[index]["name"]
        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        label = Image.open(label_path)

        if self.joint_transform:
            image, label = self.joint_transform(image, label)

        if self.transform:
            image = self.image_transforms(image)
            label = self.label_transforms(label)
        else:
            image = np.asarray(image, dtype=np.uint8)
            label = np.asarray(label, dtype=np.uint8)

        return image, label, name, width, height

    def __len__(self):
        return len(self.items_list)


def main():
    dataset = Materials("./dataset", "val")
    print(len(dataset))
    dataiter = iter(dataset)
    image, mask, name, width, height = next(dataiter)
    print(image.shape, mask.shape, name, (height, width))


if __name__ == "__main__":
    main()
