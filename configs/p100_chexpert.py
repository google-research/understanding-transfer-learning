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


def register_chexpert_training(registry):
  # random init
  for base_lr in [0.1, 0.02]:
    spec = chexpert_sgd_spec(base_lr=base_lr, epochs=400, finetune_from=None)
    spec['key'] = spec['key'].replace('/fixup_resnet50/', '/fixup_resnet50_nzfc/')
    spec['gin']['model.kwargs'] = dict(zero_fc_init=False)
    registry.register(spec)

  # finetune
  for base_lr in [0.1, 0.02]:
    spec = chexpert_sgd_spec(base_lr=base_lr, epochs=200, 
                             finetune_from='/mnt/data/logs/imagenet-lr01/ckpt-E090.pth.tar')
    spec['key'] = spec['key'].replace('/fixup_resnet50/', '/fixup_resnet50_nzfc/')
    spec['gin']['model.kwargs'] = dict(zero_fc_init=False)
    registry.register(spec)


def chexpert_sgd_spec(base_lr=0.1, epochs=200, finetune_from=None):
  batch_size = 256
  num_gpu = 2

  finetune_tag = 'finetune' if finetune_from is not None else 'randinit'
  spec = base.config_tmpl(key=f'train/chexpert/fixup_resnet50/{finetune_tag}-lr{base_lr}-bs{batch_size}', 
                          script='chexpert_train.py')

  spec['gpu_spec'] = f'{num_gpu}xV100'

  spec['gin']['data.image_path'] = '/mnt/data/CheXpert-v1.0-img224'
  spec['gin']['data.data_config'] = {'cache_bitmap': False}
  spec['gin']['model.arch'] = 'fixup_resnet50'
  spec['gin']['train.batch_size'] = batch_size
  spec['gin']['train.epochs'] = epochs
  spec['gin']['train.data_workers'] = 8
  spec['gin']['train.finetune_from'] = finetune_from
  spec['gin']['optimizer.name'] = 'SGD'
  spec['gin']['optimizer.base_lr'] = base_lr * batch_size / 256
  spec['gin']['optimizer.weight_decay'] = 1e-4
  spec['gin']['optimizer.kwargs'] = dict(momentum=0.9)

  return spec
