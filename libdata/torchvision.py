# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os

import numpy as np

import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from . import domain_net
from . import utils

__all__ = ['torchvision_get_data']


def torchvision_get_data(name, configs=None):
  if configs is None:
    configs = dict()

  if name == 'imagenet':
    # we assume imagenet is already available locally somewhere
    data_dir = configs['imagenet_path']
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_dataset = datasets.ImageFolder(
        train_dir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))
    val_dataset = datasets.ImageFolder(val_dir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))
    train_dataset_eval_mode = None
    meta = {'num_classes': 1000}
  elif name == 'cifar10':
    data_dir = os.path.join(configs.get('data_dir', 'data'), 'cifar10')
    normalize = transforms.Normalize(
            mean=np.array([0.4914, 0.4822, 0.4465]),
            std=np.array([0.2470, 0.2435, 0.2616]))

    # NOTE: we are using ImageNet sizes 224x224, because
    # we would like to use ImageNet architectures on this dataset
    train_dataset = datasets.CIFAR10(
        data_dir, train=True, download=True, 
        transform=transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))
    trans_for_eval = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])
    val_dataset = datasets.CIFAR10(
        data_dir, train=False, download=True,
        transform=trans_for_eval)
    train_dataset_eval_mode = datasets.CIFAR10(
        data_dir, train=True, download=True,
        transform=trans_for_eval)
    meta = {'num_classes': 10}
  elif name.startswith('DomainNet_'):
    real_name = name.split('_')[1]
    assert real_name in ['clipart','infograph', 'painting', 'quickdraw', 'real', 'sketch']
    data_dir = configs.get('data_dir', 'data')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    image_size = configs.get('image_size', 224)
    pre_crop_size = configs.get('pre_crop_size', 256)
    image_chnn = 3
    print(f'Loading DomainNet ({real_name}) with image_size={image_size}...')

    extra_trans = []

    shuffle_pixel_cfg = configs.get('shuffle_pixel_config', None)
    if shuffle_pixel_cfg is not None:
      print('Enabling random pixel shuffling!')
      pixel_shuffler = utils.RandomPixelShuffle(image_chw=(image_chnn, image_size, image_size),
                                                **shuffle_pixel_cfg)
      extra_trans.append(pixel_shuffler)

    shuffle_block_cfg = configs.get('shuffle_block_config', None)
    if shuffle_block_cfg is not None:
      assert shuffle_pixel_cfg is None
      print('Enabling random block shuffling!')
      block_shuffler = utils.RandomBlockShuffle(image_size=image_size,
                                                **shuffle_block_cfg)
      extra_trans.append(block_shuffler)
    trans_for_eval = transforms.Compose([
            transforms.Resize(pre_crop_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            normalize,
        ] + extra_trans)
    if configs.get('disable_aug', False):
      print('Disabling data augmentation!')
      trans_for_train = trans_for_eval
    else:
      trans_for_train = transforms.Compose([
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ] + extra_trans)

    train_dataset = domain_net.DomainNet(
        data_dir, real_name, train=True, download=True, 
        subsample=configs.get('subsample', None),
        transform=trans_for_train)
    val_dataset = domain_net.DomainNet(data_dir, real_name, train=False,
                                       download=True, transform=trans_for_eval,
                                       subsample=configs.get('subsample', None))
    train_dataset_eval_mode = domain_net.DomainNet(data_dir, real_name, train=True,
                                                   download=True, transform=trans_for_eval,
                                                   subsample=configs.get('subsample', None))
    meta = {'num_classes': len(train_dataset.classes)}

  else:
    raise KeyError(f'Unknown dataset {name}')
  return dict(train=train_dataset, test=val_dataset, eval_mode_train=train_dataset_eval_mode), meta
