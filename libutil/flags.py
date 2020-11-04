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
import argparse


def get_argparser():
  parser = argparse.ArgumentParser()
  parser.add_argument('-k', '--config-key', type=str, required=True, help='Config key.')
  parser.add_argument('-b', '--base-dir', type=str, default='experiments', help='Base work directory.')
  parser.add_argument('-w', '--work-dir', default=None, 
                      help='Work dir, by default decided by base-dir and config-key.')
  return parser
