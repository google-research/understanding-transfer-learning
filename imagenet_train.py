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
import time
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.tensorboard

from tqdm import tqdm
import gin

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
def get_data(name=gin.REQUIRED, configs=None):
  return libdata.torchvision_get_data(name, configs)


@gin.configurable('train', blacklist=['args'])
def run_train(args, batch_size=gin.REQUIRED, epochs=gin.REQUIRED, finetune_from=None, mixup_alpha=0.7, data_workers=32):
  loaded_data, data_meta = get_data()
  model = libmodel.build_model(num_classes=data_meta['num_classes'])

  if finetune_from:
    libtrain.load_finetune_init(model, finetune_from)
  else:
    logging.info('No finetune init weights specified.')

  cel = nn.CrossEntropyLoss()
  def criterion(pred, target, lam):
    """Criterion suitable for mixup training."""
    return (-F.log_softmax(pred, dim=1) * torch.zeros(pred.size()).cuda().scatter_(1, target.data.view(-1, 1), lam.view(-1, 1))).sum(dim=1).mean()

  optimizer = libtrain.make_optimizer(model)
  lr_scheduler = libtrain.make_lr_scheduler(optimizer)
  weight_regularizers = libtrain.make_weight_regularizers(model)

  train_loader = torch.utils.data.DataLoader(
      loaded_data['train'], batch_size=batch_size, shuffle=True,
      num_workers=data_workers, pin_memory=True)

  val_loader = torch.utils.data.DataLoader(
      loaded_data['test'], batch_size=batch_size, shuffle=False,
      num_workers=data_workers, pin_memory=False)
  data_to_eval = [('test', val_loader)]
  if loaded_data['eval_mode_train'] is not None:
    eval_mode_train_loader = torch.utils.data.DataLoader(
        loaded_data['eval_mode_train'], batch_size=batch_size, shuffle=False,
        num_workers=data_workers, pin_memory=False)
    data_to_eval.append(('train', eval_mode_train_loader))

  tb_writer = torch.utils.tensorboard.SummaryWriter(os.path.join(args.work_dir, 'train_tensorboard'))

  def eval_and_save_ckpt(epoch, best_acc1):
    for name, loader in data_to_eval:
      logging.info(f'E{epoch:03d} Evaluating {name}...')
      results = libtrain.imagenet_test_epoch(loader, model, cel)
      logging.info(f'E{epoch:03d} eval-{name}: Acc@1 {results["top1"]:.3f} Loss {results["loss"]:.4f}')
      tb_writer.add_scalar(f'eval/{name}_acc', results['top1'], epoch)
      if name == 'test':
        is_best = results['top1'] > best_acc1
        best_acc1 = max(results['top1'], best_acc1)
  
    libtrain.save_checkpoint({
        'epoch': epoch, 'state_dict': model.state_dict(), 'best_acc1': best_acc1, 'optimizer' : optimizer.state_dict(),
    }, is_best=is_best, ckpt_dir=os.path.join(args.work_dir, 'checkpoints'))
    return best_acc1

  best_acc1 = eval_and_save_ckpt(epoch=0, best_acc1=-float('inf'))
  for epoch in range(epochs):
    for i_grp, param_group in enumerate(optimizer.param_groups):
      tb_writer.add_scalar(f'learning_rate/group{i_grp}', param_group['lr'], epoch + 1)

    # train for one epoch
    train_epoch(train_loader, model, criterion, optimizer, epoch, mixup_alpha, weight_regularizers, tb_writer)

    best_acc1 = eval_and_save_ckpt(epoch=epoch+1, best_acc1=best_acc1)

    if lr_scheduler is not None:
      lr_scheduler.step()

    tb_writer.flush()

  tb_writer.close()


def train_epoch(train_loader, model, criterion, optimizer, epoch, mixup_alpha, weight_regularizers, tb_writer=None):
  batch_time = libtrain.AverageMeter()
  data_time = libtrain.AverageMeter()
  losses = libtrain.AverageMeter()
  top1 = libtrain.AverageMeter()

  # switch to train mode
  model.train()

  end = time.time()
  the_tqdm = tqdm(train_loader, disable=None, desc=f'Train E{epoch+1:03d}')
  for i, (inputs, targets) in enumerate(the_tqdm):
    # measure data loading time
    data_time.update(time.time() - end)

    inputs = inputs.cuda(non_blocking=True)
    targets = targets.cuda(non_blocking=True)

    inputs, targets_a, targets_b, lam = libutil.mixup.mixup_data(inputs, targets, mixup_alpha, use_cuda=True)

    # compute output
    output = model(inputs)
    loss_func = libutil.mixup.mixup_criterion(targets_a, targets_b, lam)
    loss = loss_func(criterion, output)

    loss += libtrain.eval_weight_regularizers(weight_regularizers)

    # measure accuracy and record loss
    acc1 = libtrain.calc_accuracy(output, targets, topk=(1,))[0]
    losses.update(loss.item(), inputs.size(0))
    top1.update(acc1.item(), inputs.size(0))

    # compute gradient and do SGD step
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # measure elapsed time
    batch_time.update(time.time() - end)
    end = time.time()

    the_tqdm.set_description(f'Train E{epoch+1:03d} Acc={top1.avg:.3f} Loss={losses.avg:.4f}')

  logging.info(f'E{epoch:03d} train: Acc@1 {top1.avg:.3f} Loss {losses.avg:.4f}')
  if tb_writer is not None:
    tb_writer.add_scalar('train/loss', losses.avg, epoch + 1)
    tb_writer.add_scalar('train/acc', top1.avg, epoch + 1)


if __name__ == '__main__':
  main()
