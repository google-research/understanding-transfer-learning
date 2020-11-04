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
import logging

import gin


def setup(args, my_config):
  if not args.work_dir:
    args.work_dir = os.path.join(args.base_dir, args.config_key)

  os.makedirs(args.work_dir, exist_ok=True)

  for key, val in my_config['gin'].items():
    gin.bind_parameter(key, val)

  gin_export_fn = os.path.join(args.work_dir, my_config['script'] + '.gin')
  with open(gin_export_fn, 'w') as out_f:
    out_f.write(gin.config_str())

  logfile = os.path.join(args.work_dir, my_config['script'] + '.log')

  log_format = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
  formatter = logging.Formatter(log_format)

  logging.basicConfig(level=logging.INFO, datefmt='%m-%d %H:%M')
  file_stream = open(logfile, 'w')
  handlers = [logging.StreamHandler(file_stream)]

  # define a Handler which writes INFO messages or higher to the sys.stderr
  # handlers.append(logging.StreamHandler())

  for handler in handlers:
    handler.setLevel(logging.INFO)
    # tell the handler to use this format
    handler.setFormatter(formatter)
    # add the handler to the root logger
    logging.getLogger('').addHandler(handler)

  logging.info('Starting %s...', my_config['script'])
  logging.info(':::::: Commandline args:\n%s', args)
  logging.info(':::::: Gin configurations:\n%s', gin.config_str())


