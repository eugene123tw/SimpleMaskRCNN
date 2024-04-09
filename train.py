import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from argparse import ArgumentParser
from torchvision.transforms import v2 as T
import torch
import utils
from engine import train_one_epoch, evaluate
from datumaro import Dataset as DmDataset
from dataset import DatumaroDataset
from coco_utils import get_coco, pre_filtering
from pathlib import Path
import time
import datetime

import torch
from torch.profiler import profile, record_function, ProfilerActivity


def get_transform(train, size_image=(512, 512)):
    transforms = []
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    transforms.append(T.Resize(size_image))
    transforms.append(T.ToTensor())
    # ImageNet mean and std
    transforms.append(T.Normalize((123.675, 116.28, 103.53), (58.395, 57.12, 57.375)))
    return T.Compose(transforms)


def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=torchvision.models.detection.MaskRCNN_ResNet50_FPN_Weights.DEFAULT)
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = model.roi_heads.mask_predictor.conv5_mask.out_channels
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask,
        hidden_layer,
        num_classes
    )

    return model

def main():
    parser = ArgumentParser()
    parser.add_argument("--epochs", default=50, type=int, required=False, help="number of total epochs to run")
    parser.add_argument("--data-root", type=str, required=True, help="path to dataset")
    parser.add_argument("--device", default="cuda", type=str, required=False, help="device to use for training")
    parser.add_argument("--print-freq", default=10, type=int, required=False, help="print frequency")
    parser.add_argument('--image-size', nargs='+', type=int, default=[512, 512], required=False, help='input image size')
    parser.add_argument('--batch-size', type=int, default=8, required=False, help='batch size')
    parser.add_argument('--num-workers', type=int, default=4, required=False, help='number of workers')
    parser.add_argument('--lr', type=float, default=0.007, required=False, help='learning rate')
    parser.add_argument('--wd', type=float, default=0.001, required=False, help='weight decay')
    parser.add_argument('--step-lr', type=int, default=-1, required=False, help="step learning rate. If -1 passed "
                                                                                "the step lr will be difined automatically as the 0.7 of the epochs."
                                                                                " if passed 0 no step lr will be used")
    parser.add_argument('--warmup', type=bool, default=True, required=False, help='use warmup learning rate scheduler')
    parser.add_argument('--profile', default=False, action="store_true", help='run in profile mode')
    args = parser.parse_args()

    # define arguments
    path_to_train_data = args.data_root
    num_epochs = args.epochs
    device = args.device
    if device == "xpu":
        import intel_extension_for_pytorch as ipex

    print_freq = args.print_freq
    image_size = tuple(args.image_size)
    batch_size = args.batch_size
    num_workers = args.num_workers
    learning_rate = args.lr
    weight_decay = args.wd
    warmup = args.warmup
    if args.step_lr == -1:
        step_lr = int(args.epochs * 0.7)
    else:
        step_lr = args.step_lr

    assert device in ("cuda", "cpu", "xpu"), "device should be 'cuda', 'cpu', or 'xpu'"
    assert Path(path_to_train_data).exists(), "path to train data does not exist"
    assert len(image_size) == 2, "image size should be a tuple of length 2"

    # use our dataset and defined transformations
    dataset = DmDataset.import_from(path_to_train_data, format="coco")
    dataset = pre_filtering(dataset, "coco", 0)
    subsets = dataset.subsets()
    dataset_train = DatumaroDataset(subsets["train"], get_transform(train=True, size_image=image_size))
    dataset_test = DatumaroDataset(subsets["val"], get_transform(train=False, size_image=image_size))
    # dataset_train, num_classes = get_coco(path_to_train_data, image_set="train", transforms=get_transform(train=True, size_image=image_size))
    # dataset_test, _ = get_coco(path_to_train_data, image_set="val", transforms=get_transform(train=True, size_image=image_size))
    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=utils.collate_fn
    )

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=utils.collate_fn
    )

    # load a model pre-trained on COCO
    # get the model using our helper function
    # model = get_model_instance_segmentation(dataset_train.num_classes)
    model = get_model_instance_segmentation(dataset_train.num_classes)

    # move model to the right device
    model.to(device)
    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=learning_rate,
        momentum=0.9,
        weight_decay=weight_decay
    )

    # and a learning rate scheduler
    if step_lr > 0:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=step_lr,
            gamma=0.1
        )

    if device == "xpu":
        model.train()
        model, optimizer = torch.xpu.optimize(model, optimizer=optimizer, dtype=torch.float32)

    if args.profile:
        activity = ProfilerActivity.CPU
        if device == "cuda":
            activity = ProfilerActivity.CUDA
        elif device == "xpu":
            activity = ProfilerActivity.XPU

        model.eval()
        with profile(activities=[activity], record_shapes=True) as prof:
            with record_function("model_inference"):
                inputs = torch.randn(batch_size, 3, *args.image_size).to(torch.device(device))
                model(inputs)

        print(prof.key_averages().table(sort_by=f"{device}_time_total", row_limit=20))
        model.train()
        return

    print(f"Starting training for {num_epochs} epochs")
    start_time = time.time()
    best_map_50_bb, best_map_50_seg = 0, 0
    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=print_freq, max_epochs=num_epochs, warmup=warmup)
        # update the learning rate
        if step_lr > 0:
            lr_scheduler.step()
        # evaluate on the test dataset
        best_map_50_bb, best_map_50_seg = evaluate(model, data_loader_test, device=device,
                                                   best_map_50_bb=best_map_50_bb, best_map_50_seg=best_map_50_seg)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Training time: {total_time_str}")
    print(f"Best metric bbox: {best_map_50_bb}, Best metric segm: {best_map_50_seg}")

if __name__ == "__main__":
    main()
