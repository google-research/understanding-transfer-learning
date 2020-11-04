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
import numpy as np
import torch


class RandomPixelShuffle(object):
  """Randomly shuffle pixel and channels within an image."""

  def __init__(self, image_chw, seed):
    rng = np.random.RandomState(seed=seed)
    total_dim = np.prod(image_chw)

    self.image_chw = image_chw
    self.perm = torch.from_numpy(rng.permutation(total_dim))

  def __call__(self, tensor):
    return torch.reshape(torch.flatten(tensor)[self.perm], self.image_chw)
    

class RandomBlockShuffle(object):
  """Randomly shuffle blocks within an image."""

  def __init__(self, image_size, block_size, seed):
    if image_size % block_size != 0:
      raise KeyError(f'RandomBlockShuffle: image size {image_size} cannot be divided by block size {block_size}')

    self.image_size = image_size
    self.block_size = block_size
    self.n_blocks = (image_size // block_size)**2
    rng = np.random.RandomState(seed=seed)
    self.perm = torch.from_numpy(rng.permutation(self.n_blocks))

    self.unfold_op = torch.nn.Unfold(kernel_size=block_size, stride=block_size)
    self.fold_op = torch.nn.Fold(output_size=image_size, kernel_size=block_size, stride=block_size)

  def __call__(self, tensor):
    blocks = self.unfold_op(tensor.unsqueeze(0))  # (1, block_size, n_blocks)
    assert blocks.size(2) == self.n_blocks
    blocks = blocks[..., self.perm]  # shuffle blocks
    tensor = self.fold_op(blocks)  # (1, C, H, W)
    return tensor.squeeze(0)

