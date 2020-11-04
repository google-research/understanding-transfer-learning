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
import gin

import torch
import torchvision.models as tv_models

import libarch


@gin.configurable('model', blacklist=['num_classes'])
def build_model(arch=gin.REQUIRED, num_classes=None, kwargs=None):
  assert num_classes is not None

  # create model
  if arch.startswith('torchvision/'):
    arch_name = arch.split('/')[1]
    ctor = tv_models.__dict__[arch_name]
  else:
    ctor = libarch.__dict__[arch]

  if kwargs is None:
    kwargs = dict()
  kwargs['num_classes'] = num_classes

  model = ctor(**kwargs)
  model = torch.nn.DataParallel(model).cuda()
  return model
