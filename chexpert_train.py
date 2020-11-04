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
import time

import numpy as np
import sklearn.metrics

from easydict import EasyDict as edict

import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
import gin
import easydict

import libutil
import libdata
import libmodel
import libtrain
import configs


def main():
  parser = libutil.flags.get_argparser()
  args = parser.parse_args()

  my_cfg = configs.Registry.get_config(args.config_key)
  libutil.setup.setup(args, my_cfg)

  run_train(args)


@gin.configurable('data')
def get_data(image_path=gin.REQUIRED, index_paths=None, data_config=None):
  if index_paths is None:
    index_paths = {
        'train': 'libdata/chexpert_index/train.0-50k.csv',
        'dev': 'libdata/chexpert_index/dev.csv',
        'test': 'libdata/chexpert_index/train.50k-100k.csv'
    }

  data_config_defaults = {
    'width': 224, 'height': 224, 'long_side': 224, 'fix_ratio': True,
    'pixel_mean': 128.0, 'pixel_std': 64.0, 'use_pixel_std': True, 'use_equalizeHist': True,
    'use_transforms_type': 'Aug', 'gaussian_blur': 3, 'border_pad': 'pixel_mean',
    'batch_weight': True, 'enhance_index': [2, 6], 'enhance_times': 1,
    'pos_weight': [1,1,1,1,1], 'cache_bitmap': True
  }
  if data_config is not None:
    data_config_defaults.update(data_config)
  data_config = easydict.EasyDict(data_config_defaults)

  datasets = {
      key: libdata.chexpert.ImageDataset(image_path, index_path, data_config,
                                         mode='train' if key == 'train' else 'dev')
      for key, index_path in index_paths.items()
  }
  datasets['eval_mode_train'] = libdata.chexpert.ImageDataset(
      image_path, index_paths['train'], data_config, mode='dev')

  meta = {'num_diseases': 5}
  return datasets, meta


@gin.configurable('train', blacklist=['args'])
def run_train(args, batch_size=gin.REQUIRED, epochs=gin.REQUIRED, finetune_from=None, mixup_alpha=0.0, data_workers=32):
  loaded_data, data_meta = get_data()
  model = libmodel.build_model(num_classes=data_meta['num_diseases'])

  if finetune_from:
    libtrain.load_finetune_init(model, finetune_from)
  else:
    logging.info('No finetune init weights specified.')

  optimizer = libtrain.make_optimizer(model)
  lr_scheduler = libtrain.make_lr_scheduler(optimizer)

  weight_regularizers = libtrain.make_weight_regularizers(model)

  data_loaders = {
      key: torch.utils.data.DataLoader(dset, batch_size=batch_size, num_workers=data_workers,
                                       shuffle=(key == 'train'), pin_memory=(key == 'train'))
      for key, dset in loaded_data.items()
  }
  dev_header = loaded_data['dev']._label_header
  data_to_eval = ['dev', 'test', 'eval_mode_train']
          
  tb_writer = torch.utils.tensorboard.SummaryWriter(os.path.join(args.work_dir, 'train_tensorboard'))

  def eval_and_save_ckpt(epoch, best_auc):
    for name in data_to_eval:
      logging.info(f'E{epoch:03d} Evaluating {name}...')
      results = libtrain.chexpert_test_epoch(epoch, model, data_loaders[name], dev_header, 
                                             testset_name=name, tb_writer=tb_writer)
      if name == 'test':
        is_best = results['aucs'].mean() > best_auc
        best_auc = max(results['aucs'].mean(), best_auc)
  
    libtrain.save_checkpoint({
        'epoch': epoch, 'state_dict': model.state_dict(), 'best_auc': best_auc, 'optimizer' : optimizer.state_dict(),
    }, is_best=is_best, ckpt_dir=os.path.join(args.work_dir, 'checkpoints'))
    return best_auc

  best_auc = eval_and_save_ckpt(epoch=0, best_auc=-float('inf'))
  for epoch in range(epochs):
    for i_grp, param_group in enumerate(optimizer.param_groups):
      tb_writer.add_scalar(f'learning_rate/group{i_grp}', param_group['lr'], epoch + 1)

    # train for one epoch
    train_epoch(epoch, model, data_loaders['train'], optimizer, mixup_alpha, weight_regularizers, tb_writer)

    best_auc = eval_and_save_ckpt(epoch=epoch+1, best_auc=best_auc)

    if lr_scheduler is not None:
      lr_scheduler.step()

    tb_writer.flush()

  tb_writer.close()


def train_epoch(epoch, model, dataloader, optimizer, mixup_alpha, weight_regularizers, tb_writer=None):
  torch.set_grad_enabled(True)
  model.train()

  time_now = time.time()
  losses = libtrain.AverageMeter()
  weight_regu_losses = libtrain.AverageMeter()
  accs = libtrain.AverageMeter()

  def mixup_bceloss(pred, target, lam):
    return torch.nn.functional.binary_cross_entropy_with_logits(
      pred, target * lam.view((-1, 1)), reduction='mean')

  the_tqdm = tqdm(dataloader, disable=None, desc=f'Train E{epoch+1:03d}')
  for step, (image, target) in enumerate(the_tqdm):
    image = image.cuda(non_blocking=True)
    target = target.cuda(non_blocking=True)

    if mixup_alpha > 0.0:
      # using mixup
      inputs, targets_a, targets_b, lam = libutil.mixup.mixup_data(image, target, mixup_alpha, use_cuda=True)
      output = model(inputs)
      loss_func = libutil.mixup.mixup_criterion(targets_a, targets_b, lam)
      loss = loss_func(mixup_bceloss, output)
    else:
      output = model(image)
      loss = torch.nn.functional.binary_cross_entropy_with_logits(
          output, target, reduction='mean')

    weight_regu_loss = libtrain.eval_weight_regularizers(weight_regularizers)
    loss += weight_regu_loss
    losses.update(loss.item(), image.size(0))
    weight_regu_losses.update(weight_regu_loss.item(), image.size(0))

    with torch.no_grad():
      pred = output >= 0
      correct = pred.eq(target).float()

      avg_acc = correct.mean(1)
      accs.update(avg_acc.mean(0).item(), image.size(0))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    the_tqdm.set_description(f'Train E{epoch+1:03d} Acc={accs.avg:.3f} Loss={losses.avg:.4f} WReguL={weight_regu_losses.avg:.4f}')

  time_finish = time.time()
  logging.info(f'E{epoch+1:03d} Training finished in {time_finish-time_now:.3f} secs, Acc={accs.avg:.3f}, Loss={losses.avg:.4f}')
  if tb_writer is not None:
    tb_writer.add_scalar('train/loss', losses.avg, epoch + 1)
    tb_writer.add_scalar('train/weight_regu_loss', weight_regu_losses.avg, epoch + 1)
    tb_writer.add_scalar('train/acc', accs.avg, epoch + 1)


if __name__ == '__main__':
    main()

