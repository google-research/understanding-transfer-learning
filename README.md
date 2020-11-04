# What is being transferred in transfer learning?

This repo contains the code for the following paper:

> Behnam Neyshabur*, Hanie Sedghi*, Chiyuan Zhang*. What is being transferred in transfer learning?. *equal contribution. Advances in Neural Information Processing Systems (NeurIPS), 2020.

**Disclaimer**: this is not an officially supported Google product.

# Setup

## Library dependencies

This code has the following dependencies

- pytorch (1.4.0 is tested)
- gin-config
- tqdm
- wget (the python package)

GPUs are needed to run most of the experiments.

## Data

CheXpert data (the `train` and `valid` folders) needs to be placed in
`/mnt/data/CheXpert-v1.0-img224`. If your data is in a different place,
you can specify the `data.image_path` parameter (see `configs/p100_chexpert.py`).
We pre-resized all the CheXpert images to reduce the burden of data pre-processing
using the following script:

```bash
#!/bin/bash

NEWDIR=CheXpert-v1.0-img224
mkdir -p $NEWDIR/{train,valid}

cd CheXpert-v1.0

echo "Prepare directory structure..."
find . -type d | parallel mkdir -p ../$NEWDIR/{}

echo "Resize all images to have at least 224 pixels on each side..."
find . -name "*.jpg" | parallel convert {} -resize "'224^>'" ../$NEWDIR/{}

cd ..
```

The DomainNet data will be automatically downloaded from the Internet upon
first run. By default, it will download to `/mnt/data`, which can be changed
with the `data_dir` config (see `configs/p100_domain_net.py`).

# Common Experiments

## Training jobs

CheXpert training from random init. We use 2 Nvidia V100 GPUs for CheXpert training.
If you run into out-of-memory error, you can try to reduce the batch size.

```
CUDA_VISIBLE_DEVICES=0,1 python chexpert_train.py -k train/chexpert/fixup_resnet50_nzfc/randinit-lr0.1-bs256
```

CheXpert finetuning from ImageNet pre-trained checkpoint. The code tries to load
the ImageNet pre-trained chexpoint from `/mnt/data/logs/imagenet-lr01/ckpt-E090.pth.tar`.
Or you can customize the path to checkpoint (see `configs/p100_chexpert.py`).

```
CUDA_VISIBLE_DEVICES=0,1 python chexpert_train.py -k train/chexpert/fixup_resnet50_nzfc/finetune-lr0.02-bs256
```

Similarly, DomainNet training can be executed using the script `imagenet_train.py` (replace `real` with `clipart`
and `quickdraw` to run on different domains).

```
# randinit
CUDA_VISIBLE_DEVICES=0 python imagenet_train.py -k train/DomainNet_real/fixup_resnet50_nzfc/randinit-lr0.1-MstepLR

# finetune
CUDA_VISIBLE_DEVICES=0 python imagenet_train.py -k train/DomainNet_real/fixup_resnet50_nzfc/finetune-lr0.02-MstepLR
```

## Training with shuffled blocks

The training jobs with block-shuffled images are defined in `configs/p200_pix_shuffle.py`. Run

```
python -m configs pix_shuffle
```

To see the keys of all the training jobs with pixel shuffling. Similarly,

```
python -m configs blk7_shuffle
```

list all the jobs with 7x7 block-shuffled images. You can run any of those jobs using the `-k` command line
argument. For example:

```
CUDA_VISIBLE_DEVICES=0 python imagenet_train.py \
    -k blk7_shuffle/DomainNet_quickdraw/fixup_resnet50_nzfc_noaug/randinit-lr0.1-MstepLR/seed0
```

## Finetuning from different pre-training checkpoints

The config file `configs/p200_finetune_ckpt.py` defines training jobs that finetune from different
ImageNet pre-training checkpoints along the pre-training optimization trajectory.

## Linear interpolation between checkpoints (performance barrier)

The script `ckpt_interpolation.py` performs the experiment of linearly interpolating between different
solutions. The file is self-contained. You can edit the file directly to specify which combinations
of checkpoints are to be used. The command line argument `-a compute` and `-a plot` can be used to
switch between doing the computation and making the plots based on computed results.


# General Documentation

This codebase uses `gin-config` to customize the behavior of the program, and allows
us to easily generate a large number of similar configurations with Python loops.
This is especially useful for hyper-parameter sweeps.

## Running a job

A script mainly takes a config `key` in the commandline, and it will pull the detailed
configurations according to this key from the pre-defined configs. For example:

```
python3 imagenet_train.py -k train/cifar10/fixup_resnet50/finetune-lr0.02-MstepLR
```

### Query pre-defined configs

You can list all the pre-defined config keys matching a given regex with the following command:

```
python3 -m configs <regex>
```

For example:

```
$ python3 -m configs cifar10
2 configs found ====== with regex: cifar10
    0) train/cifar10/fixup_resnet50/randinit-lr0.1-MstepLR
    1) train/cifar10/fixup_resnet50/finetune-lr0.02-MstepLR
```

## Defining new configs

All the configs are in the directory `configs`, with the naming convention `pXXX_YYY.py`. Here
`XXX` are digits, which allows ordering between configs (so when defining configs we can reference
and extend previously defined configs).

To add a new config file:

1. create `pXXX_YYY.py` file.
2. edit `__init__.py` to import this file.
3. in the newly added file, define functions to registery new configs. All the functions with the
   name `register_blah` will be automatically called.

## Customing new functions

To customize the behavior of a new function, make that function gin configurable by

```
@gin.configurable('config_name')
def my_func(arg1=gin.REQUIRED, arg2=0):
  # blah
```

Then in the pre-defined config files, you can specify the values by

```
spec['gin']['config_name.arg1'] = # whatever python objects
spec['gin']['config_name.arg2'] = 2
```

See [gin-config](https://github.com/google/gin-config) for more details.
