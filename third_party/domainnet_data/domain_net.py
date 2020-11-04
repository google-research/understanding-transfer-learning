from __future__ import print_function
import os
import logging

import wget
import numpy as np

from torchvision.datasets import VisionDataset
from torchvision.datasets.utils import download_and_extract_archive

from PIL import Image

class DomainNet(VisionDataset):
  """DomainNet dataset from 'Moment Matching for Multi-Source Domain Adaptation'.
      by Peng, Bai, Xia, Huang, Saenko,  Wang (ICCV 2019).
    
    Files from http://ai.bu.edu/M3SDA/
  """
  _domain_urls = {'clipart': "http://csr.bu.edu/ftp/visda/2019/multi-source/groundtruth/clipart.zip",
                  'infograph': "http://csr.bu.edu/ftp/visda/2019/multi-source/infograph.zip",
                  'painting': "http://csr.bu.edu/ftp/visda/2019/multi-source/groundtruth/painting.zip",
                  'quickdraw': "http://csr.bu.edu/ftp/visda/2019/multi-source/quickdraw.zip",
                  'real': "http://csr.bu.edu/ftp/visda/2019/multi-source/real.zip",
                  'sketch': "http://csr.bu.edu/ftp/visda/2019/multi-source/sketch.zip"
                  }
  _file_base_url = "http://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/"


  def __init__(self, root, domain, train=True, transform=None, target_transform=None, download=False,
               subsample=None):
    self._root = root
    self.domain = domain
    
    if download:
      self.download()

    if not self._check_exists():
      raise RuntimeError('Dataset not found.' +
                           ' You can use download=True to download it')

    super(DomainNet, self).__init__(root = self.DomainNet_folder + '/' + self.domain, transform=transform,
                               target_transform=target_transform)

    self.classes, self.class_to_idx , self.idx_to_class = self._find_classes(self.root)

    if train:
      self.paths, targets = self.read_file(self.train_file_local)
    else:
      self.paths, targets = self.read_file(self.test_file_local)

    if subsample is not None:
      logging.info('# Subsampling dataset according to %s...', subsample)

      n_sample = subsample['size']
      if n_sample < len(targets):
        rng = np.random.RandomState(seed=subsample['seed'])
        subset_idx = rng.choice(len(targets), size=n_sample, replace=False)
        self.paths = self.paths[subset_idx]
        targets = targets[subset_idx]

    self.targets = targets.astype(np.int)

  def read_file(self, file):
    with open(file) as f:
      doc = f.readlines()
      paths = []
      targets = []
      for i, line in enumerate(doc):
        path, target = line.strip().split(' ')
        paths.append(path)
        targets.append(int(target))
    return np.array(paths), np.array(targets)


  def __getitem__(self, index):
    """
    Args:
      index (int): Index

    Returns:
      tuple: (image, target) where target is index of the target class.
    """
    path, target = self.paths[index], self.targets[index]

    # doing this so that it is consistent with all other datasets
    # to return a PIL Image
    try:
      img = Image.open(os.path.join(self.DomainNet_folder,path))
    except:
      raise ValueError(f'Tried to open image {path}')

    if self.transform is not None:
     img = self.transform(img)

    if self.target_transform is not None:
     target = self.target_transform(target)

    return img, target

  def _find_classes(self, dir):
    """
    Finds the class folders in a dataset.
    Args:
        dir (string): Root directory path.
    Returns:
        tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.
    Ensures:
        No class is a subdirectory of another.
    """
    classes = [d.name for d in os.scandir(dir) if d.is_dir()]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    idx_to_class = {i: classes[i] for i in range(len(classes))}
    return classes, class_to_idx, idx_to_class

  def __len__(self):
    return len(self.paths)

  @property
  def DomainNet_folder(self):
    return os.path.join(self._root, self.__class__.__name__)

  @property
  def train_file_url(self):
    return self._file_base_url + self.domain + '_train.txt'

  @property
  def test_file_url(self):
    return self._file_base_url + self.domain + '_test.txt'

  @property
  def train_file_local(self):
    return os.path.join(self.DomainNet_folder, self.domain + '_train.txt')

  @property
  def test_file_local(self):
    return os.path.join(self.DomainNet_folder, self.domain + '_test.txt')

  def _check_exists(self):
    return (os.path.exists(os.path.join(self.DomainNet_folder, self.domain)) and 
            os.path.exists(self.train_file_local) and os.path.exists(self.test_file_local))

  def download(self):
    """Download the tar.gz data file if it doesn't exist. """

    if self._check_exists():
      return

    os.makedirs(self.DomainNet_folder, exist_ok = True)

    # Download and extract zip file
    if not os.path.exists(os.path.join(self.DomainNet_folder, self.domain)):
      url = self._domain_urls[self.domain]
      filename =  self.domain + '.zip'
      print(f"Downloading {filename}...")
      download_and_extract_archive(url, download_root=self.DomainNet_folder, filename=filename)
      print(f"Download of {filename} complete")

    # Download test and train lists
    if not os.path.exists(self.train_file_local):
      print(f"Downloading {self.train_file_url}")
      wget.download(self.train_file_url, self.train_file_local)

    if not os.path.exists(self.test_file_local):
      print(f"Downloading {self.test_file_url}")
      wget.download(self.test_file_url, self.test_file_local)
    
