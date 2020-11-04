#!/usr/bin/python
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


with open('train.csv', 'r') as in_f:
    lines = in_f.readlines()

header = lines[0]
lines = lines[1:]
rng = np.random.RandomState(seed=1234)
lines = rng.permutation(lines)

def export(fn, start_idx, stop_idx):
    with open(fn, 'w') as out_f:
        out_f.write(header)
        for line in lines[start_idx:stop_idx]:
            out_f.write(line)

# 50k trainingset
export('train.0-50k.csv', 0, 50000)

# 50k for testset
export('train.50k-100k.csv', 50000, 100000)
