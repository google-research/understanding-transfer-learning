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
"""Finetune from different checkpoints of pre-training."""

import collections
import copy


def register_chexpert_finetune_sweeps(registry):
  for data_name in ['DomainNet_clipart', 'DomainNet_real', 'DomainNet_quickdraw']:
    for base_key in [f'train/{data_name}/fixup_resnet50_nzfc/finetune-lr0.02-MstepLR',
                     f'train/{data_name}/fixup_resnet50_nzfc/finetune-lr0.1-MstepLR']:
      for ckpt_idx in [0, 1, 2, 5, 10, 29, 30, 31, 59, 60, 61, 89, 90]:
        base_spec = registry.get_config(base_key)
        spec = copy.deepcopy(base_spec)
        spec['key'] = spec['key'].replace('train/', 'finetune_ckpt/') + f'-ckpt{ckpt_idx:03d}'
        spec['gin']['train.finetune_from'] = f'/mnt/data/logs/imagenet-lr01/ckpt-E{ckpt_idx:03d}.pth.tar'
        registry.register(spec)

