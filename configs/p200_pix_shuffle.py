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

def register_domain_net_shuffle_pixel(registry):
  aug_tag = '_noaug'
  data_names = ['DomainNet_clipart', 'DomainNet_real', 'DomainNet_quickdraw']
  base_keys = [f'train/{data_name}/fixup_resnet50_nzfc{aug_tag}/finetune-lr0.02-MstepLR'
               for data_name in data_names]
  base_keys += [f'train/{data_name}/fixup_resnet50_nzfc{aug_tag}/finetune-lr0.1-MstepLR'
                for data_name in data_names]
  base_keys += [f'train/{data_name}/fixup_resnet50_nzfc{aug_tag}/randinit-lr0.02-MstepLR'
                for data_name in data_names]
  base_keys += [f'train/{data_name}/fixup_resnet50_nzfc{aug_tag}/randinit-lr0.1-MstepLR'
                for data_name in data_names]
  for base_key in base_keys:
    for seed in [0, 1, 2]:
      # pixel shuffling
      base_spec = registry.get_config(base_key)
      spec = copy.deepcopy(base_spec)
      spec['key'] = spec['key'].replace('train/', 'pix_shuffle/') + f'/seed{seed}'
      spec['gin']['data.configs']['shuffle_pixel_config'] = {'seed': seed + 1234}
      registry.register(spec)

      # block shuffling
      for block_size in [1, 2, 4, 7, 8, 14, 16, 28, 32, 56, 112]:
        spec = copy.deepcopy(base_spec)
        spec['key'] = spec['key'].replace('train/', f'blk{block_size}_shuffle/') + f'/seed{seed}'
        spec['gin']['data.configs']['shuffle_block_config'] = {'seed': seed + 1234, 
                                                               'block_size': block_size}
        registry.register(spec)

