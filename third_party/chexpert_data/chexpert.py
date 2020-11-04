import os
import logging

import numpy as np
from torch.utils.data import Dataset
import cv2
from PIL import Image
import subprocess
import torchvision.transforms as tfs


np.random.seed(0)


def TransCommon(image):
    image = cv2.equalizeHist(image)
    image = cv2.GaussianBlur(image, (3, 3), 0)
    return image


def TransAug(image):
    img_aug = tfs.Compose([
        tfs.RandomAffine(degrees=(-15, 15), translate=(0.05, 0.05),
                         scale=(0.95, 1.05), fillcolor=128)
    ])
    image = img_aug(image)
    return image


def GetTransforms(image, target=None, type='common'):
    # taget is not support now
    if target is not None:
        raise Exception(
            'Target is not support now ! ')
    # get type
    if type.strip() == 'Common':
        image = TransCommon(image)
        return image
    elif type.strip() == 'None':
        return image
    elif type.strip() == 'Aug':
        image = TransAug(image)
        return image
    else:
        raise Exception(
            'Unknown transforms_type : '.format(type))


class ImageDataset(Dataset):
    def __init__(self, data_path, label_path, cfg, mode='train', subsample_size=-1, subsample_seed=1234):
        self.cfg = cfg
        self._label_header = None
        self.data_path = data_path
        self._image_paths = []
        self._labels = []
        self._mode = mode

        self.dict = [{'1.0': '1', '': '0', '0.0': '0', '-1.0': '0'},
                     {'1.0': '1', '': '0', '0.0': '0', '-1.0': '1'}, ]
        print(f'ImageDataset constructed with data_path = {self.data_path}')
        with open(label_path) as f:
            header = f.readline().strip('\n').split(',')
            self._label_header = [
                header[7],
                header[10],
                header[11],
                header[13],
                header[15]]
            for line in f:
                labels = []
                fields = line.strip('\n').split(',')
                image_path = os.path.join(self.data_path, os.path.expanduser(fields[0]))
                flg_enhance = False
                for index, value in enumerate(fields[5:]):
                    if index == 5 or index == 8:
                        labels.append(self.dict[1].get(value))
                        if self.dict[1].get(
                                value) == '1' and \
                                self.cfg.enhance_index.count(index) > 0:
                            flg_enhance = True
                    elif index == 2 or index == 6 or index == 10:
                        labels.append(self.dict[0].get(value))
                        if self.dict[0].get(
                                value) == '1' and \
                                self.cfg.enhance_index.count(index) > 0:
                            flg_enhance = True
                # labels = ([self.dict.get(n, n) for n in fields[5:]])
                labels = list(map(int, labels))
                self._image_paths.append(image_path)
                self._labels.append(labels)
                if flg_enhance and self._mode == 'train':
                    for i in range(self.cfg.enhance_times):
                        self._image_paths.append(image_path)
                        self._labels.append(labels)
        self._num_image = len(self._image_paths)

        # NOTE(2020.04.30) we started using explicit config of data index, so disabling this dynamic subsampling
        # features to avoid confusion.
        assert subsample_size == -1
        # if subsample_size > 0:
        #   if subsample_size > self._num_image:
        #     raise AssertionError(f'subsample_size ({subsample_size}) should be less than {self._num_image}')

        #   rng = np.random.RandomState(seed=subsample_seed)
        #   idx = rng.choice(self._num_image, size=subsample_size, replace=False)

        #   self._image_paths = [self._image_paths[i] for i in idx]
        #   self._labels = [self._labels[i] for i in idx]
        #   self._num_image = len(self._labels)

        if cfg.cache_bitmap:
            self._bitmap_cache = self._build_bitmap_cache()
        else:
            self._bitmap_cache = None

    def __len__(self):
        return self._num_image

    def _border_pad(self, image):
        h, w, c = image.shape

        if self.cfg.border_pad == 'zero':
            image = np.pad(
                image,
                ((0, self.cfg.long_side - h),
                 (0, self.cfg.long_side - w), (0, 0)),
                mode='constant', constant_values=0.0
            )
        elif self.cfg.border_pad == 'pixel_mean':
            image = np.pad(
                image,
                ((0, self.cfg.long_side - h),
                 (0, self.cfg.long_side - w), (0, 0)),
                mode='constant', constant_values=self.cfg.pixel_mean
            )
        else:
            image = np.pad(
                image,
                ((0, self.cfg.long_side - h),
                 (0, self.cfg.long_side - w), (0, 0)),
                mode=self.cfg.border_pad
            )

        return image

    def _fix_ratio(self, image):
        h, w, c = image.shape

        if h >= w:
            ratio = h * 1.0 / w
            h_ = self.cfg.long_side
            w_ = round(h_ / ratio)
        else:
            ratio = w * 1.0 / h
            w_ = self.cfg.long_side
            h_ = round(w_ / ratio)

        image = cv2.resize(image, dsize=(w_, h_),
                           interpolation=cv2.INTER_LINEAR)

        image = self._border_pad(image)

        return image

    def _build_bitmap_cache(self):
        print('Pre-loading all images...(might take a while)')
        return [self._load_image(idx) for idx in range(self._num_image)]

    def _load_image(self, idx):
        image = cv2.imread(self._image_paths[idx], 0)
        image = Image.fromarray(image)
        return image

    def __getitem__(self, idx):
        if self._bitmap_cache is not None:
            image = self._bitmap_cache[idx]
        else:
            image = self._load_image(idx)

        if self._mode == 'train':
            image = GetTransforms(image, type=self.cfg.use_transforms_type)
        image = np.array(image)
        if self.cfg.use_equalizeHist:
            image = cv2.equalizeHist(image)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB).astype(np.float32)

        if self.cfg.fix_ratio:
            image = self._fix_ratio(image)
        else:
            image = cv2.resize(image, dsize=(self.cfg.width, self.cfg.height),
                               interpolation=cv2.INTER_LINEAR)
        if self.cfg.gaussian_blur > 0:
            image = cv2.GaussianBlur(image, (self.cfg.gaussian_blur,
                                             self.cfg.gaussian_blur), 0)
        # normalization
        image -= self.cfg.pixel_mean
        # vgg and resnet do not use pixel_std, densenet and inception use.
        if self.cfg.use_pixel_std:
            image /= self.cfg.pixel_std
        # normal image tensor :  H x W x C
        # torch image tensor :   C X H X W
        image = image.transpose((2, 0, 1))
        labels = np.array(self._labels[idx]).astype(np.float32)

        path = self._image_paths[idx]

        if self._mode == 'train' or self._mode == 'dev':
            return (image, labels)
        elif self._mode == 'test':
            return (image, path)
        elif self._mode == 'heatmap':
            return (image, path, labels)
        else:
            raise Exception('Unknown mode : {}'.format(self._mode))
