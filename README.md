# SimpleMaskRCNN
Simple MaskRCNN Instance Segmentation pipeline with training support on XPU (using Intel Extension For Pytorch), GPU and CPU

# Installation

1. Install torch either for cuda or for XPU (IPEX)

    ``
    pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118
    ``

2. Install requirements

    `` pip install -r requirements.txt ``

# Dataset format

[Datumaro COCO format](https://openvinotoolkit.github.io/datumaro/latest/docs/data-formats/formats/coco.html)

# How to run

```
usage: train.py [-h] [--epochs EPOCHS] --data-root DATA_ROOT [--device DEVICE] [--print-freq PRINT_FREQ] [--image-size IMAGE_SIZE [IMAGE_SIZE ...]] [--batch-size BATCH_SIZE] [--num-workers NUM_WORKERS] [--lr LR] [--wd WD] [--step-lr STEP_LR] [--warmup WARMUP]

options:
  -h, --help            show this help message and exit
  --epochs EPOCHS       number of total epochs to run
  --data-root DATA_ROOT
                        path to dataset
  --device DEVICE       device to use for training
  --print-freq PRINT_FREQ
                        print frequency
  --image-size IMAGE_SIZE [IMAGE_SIZE ...]
                        input image size
  --batch-size BATCH_SIZE
                        batch size
  --num-workers NUM_WORKERS
                        number of workers
  --lr LR               learning rate
  --wd WD               weight decay
  --step-lr STEP_LR     step learning rate. If -1 passed the step lr will be difined automatically as the 0.7 of the epochs. if passed 0 no step lr will be used
  --warmup WARMUP       use warmup learning rate scheduler
  ```


