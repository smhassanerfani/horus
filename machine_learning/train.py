import argparse
import os
import torch
import torch.backends.cudnn as cudnn
from dataloader import Horus
from torch.utils.data import DataLoader
import joint_transforms as joint_transforms
from utils.plrds import AdjustLearningRate

def get_arguments(
        MODEL="SegFormer",
        NUM_CLASSES=2,
        SNAPSHOT_DIR="./results/SegFormer/model_weights",
        DATA_DIRECTORY="./dataset",
        INPUT_SIZE=448,
        BATCH_SIZE=2,
        NUM_WORKERS=4,
        LEARNING_RATE=0.00006,
        MOMENTUM=0.9,
        WEIGHT_DECAY=0.0001,
        NUM_EPOCHS=30,
        POWER=0.9,
        RESTORE_FROM="nvidia/segformer-b0-finetuned-ade-512-512"
    ):

    parser = argparse.ArgumentParser(description=f"Training {MODEL} on ATLANTIS.")
    parser.add_argument("--model", type=str, default=MODEL,
                        help=f"Model Name: {MODEL}")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict, excluding background.")
    parser.add_argument("--snapshot-dir", type=str, default=SNAPSHOT_DIR,
                        help="Where to save snapshots of the model.")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where to restore the model parameters.")
    parser.add_argument("--input-size", type=int, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of s")
    parser.add_argument("--data-directory", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the source dataset.")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS,
                        help="Number of workers for multithreading dataloader.")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--momentum", type=float, default=MOMENTUM,
                        help="Momentum component of the optimizer.")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY,
                        help="Regularisation parameter for L2-loss.")
    parser.add_argument("--num-epochs", type=int, default=NUM_EPOCHS,
                        help="Number of epochs for training.")
    parser.add_argument("--power", type=float, default=POWER,
                        help="Decay parameter to compute the learning rate.")

    return parser.parse_args()


def train_loop(dataloader, model, loss_fn, optimizer, lr_estimator, interpolation):

    model.train()
    running_loss = 0.0
    for batch, (encoded_inputs, _, _, _) in enumerate(dataloader, 1):

        # GPU deployment
        images = encoded_inputs["pixel_values"].cuda()
        masks = encoded_inputs["labels"].cuda()
        
        # Compute prediction
        outputs = model(pixel_values=images, labels=masks)
        loss, logits = outputs.loss, outputs.logits

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Adjusting learning rate decay
        lr_estimator.num_of_iterations += len(images)
        lr = lr_estimator(lr_estimator.num_of_iterations)

        # Statistics
        running_loss += torch.nan_to_num(loss.item(), nan=0.0) * images.size(0)

        if batch % 100 == 0:
            loss, current = loss.item(), lr_estimator.num_of_iterations
            print(f"loss: {loss:.5f}, lr = {lr:.6f} [{current:6d}/{lr_estimator.max_iter:6d}]")

    epoch_loss = running_loss / len(dataloader.dataset)
    print(f"Training loss: {epoch_loss:>8f}")

def val_loop(dataloader, model, loss_fn, interpolation):

    running_loss = 0.0
    with torch.no_grad():
        for encoded_inputs, _, _, _ in dataloader:

            # GPU deployment
            images = encoded_inputs["pixel_values"].cuda()
            masks = encoded_inputs["labels"].cuda()

            # Compute prediction and loss
            outputs = model(pixel_values=images, labels=masks)
            loss, logits = outputs.loss, outputs.logits

            running_loss += torch.nan_to_num(loss.item(), nan=0.0) * images.size(0)

        val_loss = running_loss / len(dataloader.dataset)
        print(f"Validation loss: {val_loss:>8f} \n")


def main(args):
    cudnn.enabled = True
    cudnn.benchmark = True

    # Loading model
    if args.model == "SegFormer":
        from transformers import SegformerFeatureExtractor
        from transformers import SegformerForSemanticSegmentation
        feature_extractor = SegformerFeatureExtractor(reduce_labels=True).from_pretrained(args.restore_from)
        model = SegformerForSemanticSegmentation.from_pretrained(args.restore_from)

    try:
        os.makedirs(args.snapshot_dir)
    except FileExistsError:
        pass


    model = model.cuda()

    # Dataloader
    train_joint_transform_list = [
        joint_transforms.RandomSizeAndCrop(
            args.input_size,
            False,
            pre_size=None,
            scale_min=0.5,
            scale_max=2.0,
            ignore_index=0),
        joint_transforms.Resize(args.input_size),
        joint_transforms.RandomHorizontallyFlip()]

    train_joint_transform = joint_transforms.Compose(train_joint_transform_list)

    train_dataset = Horus(args.data_directory, split="train",
                          joint_transform=train_joint_transform, segformer=feature_extractor)
    val_dataset = Horus(args.data_directory, split="val",
                        joint_transform=train_joint_transform, segformer=feature_extractor)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.num_workers, pin_memory=True, drop_last=False)

    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True,
                                num_workers=args.num_workers, pin_memory=True, drop_last=False)

    # Initializing the loss function and optimizer
    loss_fn = None

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    # optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate,
    #                             momentum=args.momentum, weight_decay=args.weight_decay)

    interpolation = torch.nn.Upsample(size=(args.input_size, args.input_size), mode="bilinear",
                                      align_corners=True)

    max_iter = args.num_epochs * len(train_dataloader.dataset)
    lr_poly = AdjustLearningRate(optimizer, args.learning_rate, max_iter, args.power)

    for epoch in range(args.num_epochs):
        print(f"Epoch {epoch + 1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer, lr_poly, interpolation)
        val_loop(val_dataloader, model, loss_fn, interpolation)
        torch.save(model.state_dict(),
                   os.path.join(args.snapshot_dir, "epoch" + str(epoch + 1) + ".pth"))
    print("Done!")


if __name__ == "__main__":
    args = get_arguments()
    print(f"{args.model} is deployed on {torch.cuda.get_device_name(0)}")
    main(args)
