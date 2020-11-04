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
import collections
import copy

from . import base


def register_domain_net_training(registry):
  for finetune_from in [None, '/mnt/data/logs/imagenet-lr01/ckpt-E090.pth.tar']:
    for base_lr in [0.02, 0.1]:
      for data_name in ['DomainNet_clipart', 'DomainNet_real', 'DomainNet_quickdraw']:
        for disable_aug in [True, False]:
          spec = domainnet_sgd_spec(data_name, base_lr=base_lr, finetune_from=finetune_from, 
                                    disable_aug=disable_aug)
          spec['key'] += '-MstepLR'
          spec['gin']['lr_scheduler.name'] = 'MultiStepLR' 
          spec['gin']['lr_scheduler.kwargs'] = dict(milestones=[30, 60, 90], gamma=0.1)

          registry.register(spec)


def domainnet_sgd_spec(data_name, base_lr=0.1, finetune_from=None, disable_aug=False):
  num_gpu = 1

  finetune_tag = 'finetune' if finetune_from is not None else 'randinit'
  no_aug_tag = '_noaug' if disable_aug else ''
  spec = base.config_tmpl(
    key=f'train/{data_name}/fixup_resnet50_nzfc{no_aug_tag}/{finetune_tag}-lr{base_lr}', 
    script='imagenet_train.py')

  div_factor = 8

  spec['gpu_spec'] = f'{num_gpu}xV100'
  spec['gin']['data.name'] = data_name
  spec['gin']['data.configs'] = dict(subsample={'size': 50000, 'seed': 2020},
                                     data_dir='/mnt/data', disable_aug=disable_aug)
  spec['gin']['model.arch'] = 'fixup_resnet50'
  spec['gin']['model.kwargs'] = dict(zero_fc_init=False)
  spec['gin']['train.batch_size'] = 256 // div_factor
  spec['gin']['train.epochs'] = 100
  spec['gin']['train.data_workers'] = 16
  spec['gin']['train.finetune_from'] = finetune_from
  spec['gin']['optimizer.name'] = 'SGD'
  spec['gin']['optimizer.base_lr'] = base_lr / div_factor
  spec['gin']['optimizer.weight_decay'] = 1e-4
  spec['gin']['optimizer.kwargs'] = dict(momentum=0.9)

  return spec
