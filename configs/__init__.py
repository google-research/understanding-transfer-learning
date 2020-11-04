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
import inspect
import re
import sys

from . import p100_domain_net
from . import p100_chexpert

from . import p200_finetune_ckpt
from . import p200_pix_shuffle


class Registry(object):
  """Registry holding all model specs in the model zoo."""
  registry = None

  @classmethod
  def build_registry(cls):
    """Build registry, called upon first zoo query."""
    if cls.registry is not None:
      return

    cls.registry = collections.OrderedDict()

    mod_self = sys.modules[__name__]
    config_mods = [getattr(mod_self, x) for x in dir(mod_self)
                    if re.match(r'^p[0-9]+_', x)]
    config_mods = [x for x in config_mods if isinstance(x, type(sys))]
    config_mods.sort(key=str)
    for p_mod in config_mods:
      register_funcs = [getattr(p_mod, x) for x in dir(p_mod)
                        if x.startswith('register_')]
      register_funcs = filter(callable, register_funcs)

      for func in register_funcs:
        func(cls)

  @classmethod
  def register(cls, config):
    key = config['key']
    if key in cls.registry:
      raise KeyError('duplicated config key: {}'.format(key))
    cls.registry[key] = config

  @classmethod
  def list_configs(cls, regex):
    cls.build_registry()
    configs = [cls.registry[key] for key in cls.registry.keys()
               if re.search(regex, key)]
    return configs

  @classmethod
  def print_configs(cls, regex):
    configs = cls.list_configs(regex)
    print('{} configs found ====== with regex: {}'.format(
        len(configs), regex))
    for i, config in enumerate(configs):
      print('  {:>3d}) {}'.format(i, config['key']))

  @classmethod
  def get_config(cls, key):
    cls.build_registry()
    return cls.registry[key]
