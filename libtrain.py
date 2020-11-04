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
import re
import time
import logging

import gin
from tqdm import tqdm

import numpy as np
import sklearn.metrics

import torch
import torch.utils.tensorboard

import libutil


def load_finetune_init(model, checkpoint_path):
  logging.info('Loading finetune inits from %s', checkpoint_path)

  state_dict = torch.load(checkpoint_path)['state_dict']
  keys_to_exclude = ['module.fc.weight', 'module.fc.bias']

  for key in keys_to_exclude:
    del state_dict[key]
  ret = model.load_state_dict(state_dict, strict=False)
  assert ret.missing_keys == keys_to_exclude
  assert ret.unexpected_keys == []


@gin.configurable('lr_scheduler', blacklist=['optimizer'])
def make_lr_scheduler(optimizer, name=None, kwargs=None):
  if name is None:
    return None

  if kwargs is None:
    kwargs = dict()
  ctor = getattr(torch.optim.lr_scheduler, name)
  return ctor(optimizer, **kwargs)


@gin.configurable('optimizer', blacklist=['model'])
def make_optimizer(model, name='SGD', weight_decay=gin.REQUIRED, base_lr=gin.REQUIRED, custom_lrs=None, kwargs=None):
  # TODO: this could potentially change batch norm bias learning rate if batch norm is used
  # we should make this more specific for fixup architectures
  parameters_bias = [p[1] for p in model.named_parameters() if 'bias' in p[0]]
  parameters_scale = [p[1] for p in model.named_parameters() if 'scale' in p[0]]

  # custom_lrs can be specified as
  # [
  #   (param_regex, base_lr),
  #   (param_regex2, base_lr2),
  # ]
  if custom_lrs is None:
    custom_lrs = []

  param_groups = [{'params': []} for _ in range(len(custom_lrs) + 1)]
  for p_name, p_var in model.named_parameters():
    if 'bias' in p_name or 'scale' in p_name:
      continue  # ignore, fixup related parameters already handled above
    matched = False
    for i, (regex, lr) in enumerate(custom_lrs):
      if re.search(regex, p_name):
        param_groups[i]['params'].append(p_var)
        param_groups[i]['lr'] = lr
        matched = True
        break
    if not matched:
      # the default group
      param_groups[-1]['params'].append(p_var)
      param_groups[-1]['lr'] = base_lr

  all_param_groups = [
          {'params': parameters_bias, 'lr': base_lr/10.},
          {'params': parameters_scale, 'lr': base_lr/10.}
  ] + param_groups

  def is_tensor_in(tensor, group):
    for t in group:
      if t is tensor:
        return True
    return False

  logging.info('#################################### Initialized learning rates')
  for i, p_group in enumerate(all_param_groups):
    logging.info('  GROUP %d', i)
    logging.info('    base_lr = %g', p_group['lr'])
    for p_name, p_var in model.named_parameters():
      if is_tensor_in(p_var, p_group['params']):
        logging.info('     - %s', p_name)
  logging.info('####################################')

  ctor = getattr(torch.optim, name)
  optimizer = ctor(
      all_param_groups, lr=base_lr, weight_decay=weight_decay, **(kwargs or dict()))

  return optimizer


@gin.configurable('weight_regularizers', blacklist=['init_model'])
def make_weight_regularizers(init_model, rules=None):
  if rules is None:
    return None

  # weights at initialization, keep in cpu memory
  init_states = {key: val.cpu().detach().clone() for key, val in init_model.state_dict().items()}

  # we need to define a local function to create
  # proper closure here, as python for loop rebind
  # the variable in closure
  def _make_regu(name, tensor, ref_tensor, regu):
    if regu['type'] == 'l2_to_0':
      logging.info(f'  {regu["coef"]}||w||^2 -- {name}')
      return lambda: regu['coef'] * torch.sum(tensor**2)
    elif regu['type'] == 'l2_to_w0':
      logging.info(f'  {regu["coef"]}||w - w0||^2 -- {name}')
      return lambda: regu['coef'] * torch.sum((tensor - ref_tensor.cuda())**2)
    else:
      raise KeyError(f'Unknown weight regularizer {regu["type"]}')

  regularizers = []
  logging.info('### Building weight regularizers...')
  for regex, regu in rules:
    for name, tensor in init_model.named_parameters():
      if re.search(regex, name):
        regularizers.append(_make_regu(name, tensor, init_states[name], regu))

  logging.info(f'### {len(regularizers)} weight regularizers built.')
  return regularizers


def eval_weight_regularizers(regularizers):
  if regularizers is None:
    return torch.tensor(0.0)
  return sum([regu() for regu in regularizers])


def save_checkpoint(state, is_best, ckpt_dir):
  epoch = state['epoch']
  fn_ckpt = f'ckpt-E{epoch:03d}.pth.tar'
  fn_best = 'ckpt-best.pth.tar'

  os.makedirs(ckpt_dir, exist_ok=True)
  torch.save(state, os.path.join(ckpt_dir, fn_ckpt))
  if is_best:
    torch.save(state, os.path.join(ckpt_dir, fn_best))


def imagenet_test_epoch(val_loader, model, criterion):
  batch_time = AverageMeter()
  losses = AverageMeter()
  top1 = AverageMeter()

  # switch to evaluate mode
  model.eval()

  with torch.no_grad():
    end = time.time()
    for i, (input, target) in enumerate(tqdm(val_loader, disable=None)):
      input = input.cuda(non_blocking=True)
      target = target.cuda(non_blocking=True)

      # compute output
      output = model(input)
      loss = criterion(output, target)


      # measure accuracy and record loss
      acc1 = calc_accuracy(output, target, topk=(1,))[0]
      losses.update(loss.item(), input.size(0))
      top1.update(acc1[0], input.size(0))

      # measure elapsed time
      batch_time.update(time.time() - end)
      end = time.time()

  return dict(top1=top1.avg, loss=losses.avg, batch_time=batch_time.avg)



def chexpert_test_epoch(epoch, model, dataloader, label_header, testset_name='test', tb_writer=None):
  torch.set_grad_enabled(False)
  model.eval()

  losses = AverageMeter()

  sigmoid_all = []
  target_all = []
  t_start = time.time()
  for step, (image, target) in enumerate(tqdm(dataloader, disable=None)):
    image = image.cuda(non_blocking=True)
    target = target.cuda(non_blocking=True)
       
    output = model(image)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        output, target, reduction='mean')
    losses.update(loss.item(), image.size(0))

    sigmoid_all.append(torch.sigmoid(output).cpu().numpy())
    target_all.append(target.cpu().numpy())

  logging.info('E{:03d} Eval-{} Time: {:.3f} secs, Loss : {:06f}'.format(
      epoch, testset_name, time.time() - t_start, losses.avg))
  sigmoid_all = np.concatenate(sigmoid_all, axis=0)
  target_all = np.concatenate(target_all, axis=0)
  accs = np.mean(np.equal(sigmoid_all >= 0.5, target_all), axis=0)
  auc_list = []
  for i in range(len(accs)):
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(
        target_all[:, i], sigmoid_all[:, i], pos_label=1)
    auc = sklearn.metrics.auc(fpr, tpr)
    auc_list.append(auc)
    logging.info(f'    [{i:02d}] {label_header[i]:>16s}:: Accuracy: {accs[i]:.4f}, AUC: {auc:.4f}')
  logging.info(f'    {testset_name}-Mean Accuracy: {np.mean(accs):.4f}, AUC: {np.mean(auc_list):.4f}')

  if tb_writer is not None:
    tb_writer.add_scalar(f'eval/{testset_name}_auc', np.mean(auc_list), epoch)
    tb_writer.add_scalar(f'eval/{testset_name}_loss', losses.avg, epoch)
    tb_writer.add_scalar(f'eval/{testset_name}_acc', np.mean(accs), epoch)

  return dict(aucs=np.array(auc_list), accs=accs, loss=losses.avg)


def calc_accuracy(output, target, topk=(1,)):
  """Computes the accuracy over the k top predictions for the specified values of k"""
  with torch.no_grad():
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
      correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
      res.append(correct_k.mul_(100.0 / batch_size))
    return res


class AverageMeter(object):
  """Computes and stores the average and current value"""
  def __init__(self):
    self.reset()

  def reset(self):
    self.val = 0
    self.sum = 0
    self.count = 0

  def update(self, val, n=1):
    self.val = val
    self.sum += val * n
    self.count += n

  @property
  def avg(self): 
    return self.sum / self.count
