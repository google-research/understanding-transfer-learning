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
"""Evaluate the linear interpolation of two checkpoints."""

import os
import argparse
import pickle

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

import libutil
import libdata
import libmodel
import libtrain

from tqdm import tqdm


def eval_interpolation(model, state_dict_1, state_dict_2, eval_fn, num=3):
  weights = np.linspace(0, 1, num)

  results = []
  for i in range(num):
    model.load_state_dict(
        interpolate_state_dicts(state_dict_1, state_dict_2, weights[i]))
    results.append(eval_fn(model))
  return results
    

def interpolate_state_dicts(state_dict_1, state_dict_2, weight):
  return {key: (1-weight)*state_dict_1[key] + weight*state_dict_2[key]
          for key in state_dict_1.keys()}


def load_state_dict_from_gs(gs_path):
  local_path = libutil.gcloud.download_as_tmp(gs_path)
  return torch.load(local_path)['state_dict']


def experiment_for_domain_net(dn_name, batch_size=500, data_workers=4):
  data_cfg = dict(subsample={'size': 50000, 'seed': 2020})
  loaded_data, data_meta = libdata.torchvision_get_data(f'DomainNet_{dn_name}', data_cfg)
  test_loader = torch.utils.data.DataLoader(
      loaded_data['test'], batch_size=batch_size, shuffle=False,
      num_workers=data_workers, pin_memory=False)
  model = libmodel.build_model(arch='fixup_resnet50', num_classes=data_meta['num_classes'],
                               kwargs=dict(zero_fc_init=False))

  def eval_fn(the_model):
    results = libtrain.imagenet_test_epoch(test_loader, the_model, criterion=nn.CrossEntropyLoss())
    return {key: float(val) for key, val in results.items()}

  ckpt_finetune_fn = 'fixup_resnet50_nzfc/finetune-lr0.02-MstepLR/checkpoints/ckpt-E100.pth.tar'
  ckpt_randinit_fn = 'fixup_resnet50_nzfc/randinit-lr0.1-MstepLR/checkpoints/ckpt-E100.pth.tar'
  prefix1 = 'experiments/train'
  prefix2 = 'experiments-rep2/train'
  checkpoint_paths = {
    'randinit_1': f'{prefix1}/DomainNet_{dn_name}/{ckpt_randinit_fn}',
    'randinit_2': f'{prefix2}/DomainNet_{dn_name}/{ckpt_randinit_fn}',
    'finetune_1': f'{prefix1}/DomainNet_{dn_name}/{ckpt_finetune_fn}',
    'finetune_2': f'{prefix2}/DomainNet_{dn_name}/{ckpt_finetune_fn}',
  }

  all_results = {}
  num_eval = 50
  all_keys = list(checkpoint_paths.keys())
  for i in range(len(all_keys)):
    for j in range(i+1, len(all_keys)):
      key1 = all_keys[i]
      key2 = all_keys[j]

      state_dict1 = load_state_dict_from_gs(checkpoint_paths[key1])
      state_dict2 = load_state_dict_from_gs(checkpoint_paths[key2])
      interpolate_results = eval_interpolation(
        model, state_dict1, state_dict2, eval_fn, num=num_eval)
      ret_key = f'{key1} vs {key2}'
      all_results[ret_key] = interpolate_results
      print('#' * 40)
      print(ret_key)
      print(interpolate_results)

      with open(f'experiments/ckpt_interpolation/{dn_name}.p', 'wb') as f:
        pickle.dump(all_results, f)


def experiment_for_chexpert(batch_size=256, data_workers=8):
  from chexpert_train import get_data
  loaded_data, data_meta = get_data(data_config={'cache_bitmap': False})
  model = libmodel.build_model(arch='fixup_resnet50', num_classes=data_meta['num_diseases'],
                               kwargs=dict(zero_fc_init=False))
  test_loader = torch.utils.data.DataLoader(
      loaded_data['test'], batch_size=batch_size, num_workers=data_workers,
      shuffle=False, pin_memory=False)
  dev_header = loaded_data['dev']._label_header
  
  def eval_fn(the_model):
    results = libtrain.chexpert_test_epoch(0, the_model, test_loader, dev_header, testset_name='test')
    return results

  prefix = 'experiments/train/chexpert/fixup_resnet50_nzfc'
  prefix2 = 'experiments-rep2/train/chexpert/fixup_resnet50_nzfc'
  ckpt_fn = lambda epoch: f'checkpoints/ckpt-E{epoch:03d}.pth.tar'
  checkpoint_paths = {
    'finetune-best': [f'{prefix}/finetune-lr0.02-bs256/{ckpt_fn(4)}',
                      f'{prefix}-rep1/finetune-lr0.02-bs256/{ckpt_fn(4)}'],
    'finetune-final': [f'{prefix}/finetune-lr0.02-bs256/{ckpt_fn(200)}',
                       f'{prefix}-rep1/finetune-lr0.02-bs256/{ckpt_fn(200)}'],
    'randinit-lr0.1-best': [f'{prefix}/randinit-lr0.1-bs256/{ckpt_fn(68)}',
                            f'{prefix}-rep1/randinit-lr0.1-bs256/{ckpt_fn(68)}'],
    'randinit-lr0.1-final': [f'{prefix}/randinit-lr0.1-bs256/{ckpt_fn(400)}',
                             f'{prefix}-rep1/randinit-lr0.1-bs256/{ckpt_fn(400)}'],
    'randinit-lr0.02-best': [f'{prefix}/randinit-lr0.02-bs256/{ckpt_fn(173)}',
                             f'{prefix2}/randinit-lr0.02-bs256/{ckpt_fn(176)}'],
    'randinit-lr0.02-final': [f'{prefix}/randinit-lr0.02-bs256/{ckpt_fn(400)}',
                              f'{prefix2}/randinit-lr0.02-bs256/{ckpt_fn(400)}'],
  }

  all_results = {}
  num_eval = 20
  def _do_eval(name, fn1, fn2):
    sd1 = load_state_dict_from_gs(fn1)
    sd2 = load_state_dict_from_gs(fn2)
    interpolate_results = eval_interpolation(model, sd1, sd2, eval_fn, num=num_eval)
    all_results[name] = interpolate_results
    print('#' * 40)
    print(name)
    print(interpolate_results)
  
    pkl_fn = f'experiments/ckpt_interpolation/chexpert.p'
    if os.path.exists(pkl_fn):
      with open(pkl_fn, 'rb') as f:
        existing_results = pickle.load(f)
        existing_results.update(all_results)
        all_rsults = existing_results
    with open(pkl_fn, 'wb') as f:
      pickle.dump(all_results, f)

  cps = checkpoint_paths
  _do_eval('ft-best vs ft-best', cps['finetune-best'][0], cps['finetune-best'][1])
  _do_eval('ft-final vs ft-final', cps['finetune-final'][0], cps['finetune-final'][1])
  _do_eval('ft-best vs ft-final', cps['finetune-best'][1], cps['finetune-final'][1])
  _do_eval('ri-lr0.1-best vs ri-lr0.1-best', cps['randinit-lr0.1-best'][0], cps['randinit-lr0.1-best'][1])
  _do_eval('ri-lr0.1-final vs ri-lr0.1-final', cps['randinit-lr0.1-final'][0], cps['randinit-lr0.1-final'][1])
  _do_eval('ri-lr0.1-best vs ri-lr0.1-final', cps['randinit-lr0.1-best'][1], cps['randinit-lr0.1-final'][1])
  _do_eval('ft-best vs ri-lr0.1-best', cps['finetune-best'][0], cps['randinit-lr0.1-best'][1])

  _do_eval('ft-best vs ri-lr0.02-best', cps['finetune-best'][0], cps['randinit-lr0.02-best'][1])
  _do_eval('ri-lr0.02-best vs ri-lr0.02-best', cps['randinit-lr0.02-best'][0], cps['randinit-lr0.02-best'][1])
  _do_eval('ri-lr0.02-final vs ri-lr0.02-final', cps['randinit-lr0.02-final'][0], cps['randinit-lr0.02-final'][1])
  _do_eval('ri-lr0.02-best vs ri-lr0.02-final', cps['randinit-lr0.02-best'][1], cps['randinit-lr0.02-final'][1])

def linear_coeff_xticks(num, n_ticks=None):
  if n_ticks is None:
    ticks = np.arange(num)
    tick_labels = np.array([f'{x:.2f}' for x in np.linspace(0, 1, len(ticks))])
    orig_ticks, _ = plt.xticks()
    n_ticks = len(orig_ticks) - 1
    tick_idx = np.linspace(0, len(ticks)-1, n_ticks).astype(np.int64)
    ticks = ticks[tick_idx]
    tick_labels = tick_labels[tick_idx]
  else:
    ticks = np.linspace(0, num-1, n_ticks)
    tick_labels = [f'{x:.2f}' for x in np.linspace(0, 1, n_ticks)]

  plt.xticks(ticks, tick_labels)


def plot_for_domain_net(dn_name, criterion='top1', plot_type='bar'):
  pkl_fn = f'experiments/ckpt_interpolation/{dn_name}.p'
  with open(pkl_fn, 'rb') as f:
    all_results = pickle.load(f)

  all_keys = [
    ('randinitT-randinitT', ['randinit_1 vs randinit_2']),
    ('randinitT-finetuneT', ['randinit_1 vs finetune_2']),
    ('finetuneT-finetuneT', ['finetune_1 vs finetune_2']),
  ]

  key_ret_map = {}
  for key, orig_key_list in all_keys:
    results = []
    for orig_key in orig_key_list:
      results.append([x[criterion] for x in all_results[orig_key]])
    results = np.mean(results, axis=0)
    key_ret_map[key] = results
  
  bar_gap = 0.1
  bar_w = (1-bar_gap) / len(all_keys)
  bar_offset = (1-bar_gap) / 2

  plot_keys = [x[0] for x in all_keys]
  with plt.style.context('ggplot'):
    if plot_type == 'bar':
      plt.figure(figsize=(6, 3.5))
    elif plot_type == 'plot':
      plt.figure(figsize=(3.5, 4))

    for i, key in enumerate(plot_keys):
      accs = key_ret_map[key]
      if plot_type == 'bar':
        plt.bar(np.arange(len(accs)) - bar_offset + i*bar_w, accs, bar_w, align='edge', label=key)
      elif plot_type == 'plot':
        plt.plot(np.arange(len(accs)), accs, '.-', label=key)

    if plot_type == 'bar':
      linear_coeff_xticks(len(key_ret_map[plot_keys[0]])) 
    elif plot_type == 'plot':
      linear_coeff_xticks(len(key_ret_map[plot_keys[0]]), n_ticks=6) 
    plt.xlabel('linear interpolation coefficient')
    if criterion == 'top1':
      plt.ylabel('test accuracy %')
    elif criterion == 'loss':
      plt.ylabel('test loss')
    if plot_type == 'bar':
      plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left', ncol=len(plot_keys), mode="expand", 
                borderaxespad=0., fontsize='x-small')
    elif plot_type == 'plot':
      plt.legend(fontsize='x-small')
    plt.savefig(f'experiments/ckpt_interpolation/{dn_name}-{criterion}-{plot_type}.pdf', bbox_inches='tight')


def plot_for_chexpert(criterion='aucs', plot_type='bar', randinit_lr=0.1):
  pkl_fn = f'experiments/ckpt_interpolation/chexpert.p'
  with open(pkl_fn, 'rb') as f:
    all_results = pickle.load(f)

  plot_labels = [
    'finetune*-finetune*', 'finetune*-randinit*', 'randinit*-randinit*', 
    'finetuneT-finetuneT', 'finetune*-finetuneT', 
    'randinitT-randinitT', 'randinit*-randinitT',
  ]
  key_mapping = {'finetune*': 'ft-best', 'finetuneT': 'ft-final',
                 'randinit*': f'ri-lr{randinit_lr}-best', 'randinitT': f'ri-lr{randinit_lr}-final'}
  bar_gap = 0.1
  bar_w = (1-bar_gap) / len(plot_labels)
  bar_offset = (1-bar_gap) / 2

  def get_bar_data(data_key):
    data = all_results[data_key]
    if criterion == 'aucs':
      data = [np.mean(x['aucs']) for x in data]
    elif criterion == 'loss':
      data = [x['loss'] for x in data]
    return data
  
  with plt.style.context('ggplot'):
    if plot_type == 'bar':
      plt.figure(figsize=(6, 3.5))
    elif plot_type == 'plot':
      plt.figure(figsize=(3.5, 4))

    for i, label in enumerate(plot_labels):
      key1, key2 = label.split('-')
      real_key = f'{key_mapping[key1]} vs {key_mapping[key2]}'
      bar_data = get_bar_data(real_key)
      if plot_type == 'bar':
        plt.bar(np.arange(len(bar_data)) - bar_offset + i*bar_w, bar_data, 
                bar_w, align='edge', label=label)
      elif plot_type == 'plot':
        plt.plot(np.arange(len(bar_data)), bar_data, '.-', label=label)

    if plot_type == 'plot':
      linear_coeff_xticks(len(bar_data), n_ticks=6)
      plt.legend(fontsize='x-small', ncol=2)
    elif plot_type == 'bar':
      linear_coeff_xticks(len(bar_data))
      plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left', ncol=3, mode="expand", 
                borderaxespad=0., fontsize='x-small')
    plt.xlabel('linear interpolation coefficient')
    if criterion == 'aucs':
      plt.ylabel('test AUC')
      #plt.ylim(0.47, 0.8)
    elif criterion == 'loss':
      plt.ylabel('test loss')
      #plt.ylim(0.4, 1.3)
    if randinit_lr == 0.1:
      lr_tag = ''
    else:
      lr_tag = f'-rilr{randinit_lr}'
    plt.savefig(f'experiments/ckpt_interpolation/chexpert-{criterion}-{plot_type}{lr_tag}.pdf', 
                bbox_inches='tight')
  

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('-a', '--action', choices=['compute', 'plot'])
  parser.add_argument('-e', '--experiment')
  args = parser.parse_args()

  if args.experiment.startswith('DomainNet_'):
    dn_name = args.experiment.split('_')[1]
    if args.action == 'compute':
      experiment_for_domain_net(dn_name)
    elif args.action == 'plot':
      for plot_type in ['bar', 'plot']:
        plot_for_domain_net(dn_name, 'top1', plot_type=plot_type)
        plot_for_domain_net(dn_name, 'loss', plot_type=plot_type)
  elif args.experiment == 'chexpert':
    if args.action == 'compute':
      experiment_for_chexpert()
    elif args.action == 'plot':
      for plot_type in ['bar', 'plot']:
        for randinit_lr in [0.1, 0.02]:
          plot_for_chexpert('aucs', plot_type=plot_type, randinit_lr=randinit_lr)
          plot_for_chexpert('loss', plot_type=plot_type, randinit_lr=randinit_lr)