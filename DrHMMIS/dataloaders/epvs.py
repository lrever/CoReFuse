import os
import torch
import numpy as np
from glob import glob
from torch.utils.data import Dataset
import h5py
import itertools
from torch.utils.data.sampler import Sampler


class epvs_001(Dataset):
    """ BraTS2019 Dataset """

    def __init__(self, base_dir=None, split='train', num=None, transform=None):
        self._base_dir = base_dir
        self.transform = transform
        self.sample_list = []
        #self._base_dir+'/train4.txt'
        
        if split == 'train':
            train_path = os.path.join(self._base_dir,'train1.txt')
            absolute_path = os.path.abspath(train_path)
            print(absolute_path)
            with open(train_path, 'r') as f:
                self.image_list = f.readlines()
        elif split == 'val':
            test_path = os.path.join(self._base_dir,'val0.txt')
            with open(test_path, 'r') as f:
                self.image_list = f.readlines()

        if hasattr(self, 'image_list'):  # 检查 self.image_list 是否存在
            self.image_list = [item.replace('\n', '').split(",")[0] for item in self.image_list]
            print(self.image_list)
            print("total {} samples".format(len(self.image_list)))
        else:
            raise AttributeError("image_list is not initialized")

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_name = self.image_list[idx]
        h5f = h5py.File(self._base_dir + "/{}".format(image_name), 'r')
        FLAIR = h5f['flair'][:]
        T1 = h5f['t1'][:]
        T2=h5f['t2'][:]
        label = h5f['seg'][:]
        mask = h5f['mask'][:]
        #print(FLAIR.shape,T1.shape,T2.shape,label.shape)
        sample = {'flair': FLAIR,'t1':T1,'t2':T2,'seg': label.astype(np.uint8),'mask': mask.astype(np.uint8)}
        if self.transform:
            sample = self.transform(sample)
        return sample,image_name


class CenterCrop(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        FLAIR,T1w,T2w, label = sample['FLAIR'], sample['T1w'],sample['T2w'],sample['label']

        # pad the sample if necessary
        if label.shape[0] <= self.output_size[0] or label.shape[1] <= self.output_size[1] or label.shape[2] <= \
                self.output_size[2]:
            pw = max((self.output_size[0] - label.shape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - label.shape[1]) // 2 + 3, 0)
            pd = max((self.output_size[2] - label.shape[2]) // 2 + 3, 0)
            FLAIR = np.pad(FLAIR, [(pw, pw), (ph, ph), (pd, pd)],
                           mode='constant', constant_values=0)
            T1w = np.pad(T1w, [(pw, pw), (ph, ph), (pd, pd)],
                           mode='constant', constant_values=0)
            T2w = np.pad(T2w, [(pw, pw), (ph, ph), (pd, pd)],
                           mode='constant', constant_values=0)
            label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)],
                           mode='constant', constant_values=0)

        (w, h, d) = FLAIR.shape

        w1 = int(round((w - self.output_size[0]) / 2.))
        h1 = int(round((h - self.output_size[1]) / 2.))
        d1 = int(round((d - self.output_size[2]) / 2.))

        label = label[w1:w1 + self.output_size[0], h1:h1 +
                      self.output_size[1], d1:d1 + self.output_size[2]]
        FLAIR = FLAIR[w1:w1 + self.output_size[0], h1:h1 +
                      self.output_size[1], d1: d1 + self.output_size[2]]
        T1w = T1w[w1:w1 + self.output_size[0], h1:h1 +
                      self.output_size[1], d1: d1 + self.output_size[2]]
        T2w = T2w[w1:w1 + self.output_size[0], h1:h1 +
                      self.output_size[1], d1:d1 + self.output_size[2]]

        return {'FLAIR': FLAIR,'T1w':T1w,'T2w':T2w, 'label': label}


class RandomCrop(object):
    """
    Crop randomly the image in a sample
    Args:
    output_size (int): Desired output size
    """

    def __init__(self, output_size, with_sdf=False):
        self.output_size = output_size
        self.with_sdf = with_sdf

    def __call__(self, sample):
        FLAIR,T1w,T2w, label,mask = sample['flair'],sample['t1'],sample['t2'], sample['seg'],sample['mask']
        if self.with_sdf:
            sdf = sample['sdf']

        # pad the sample if necessary
        if label.shape[0] <= self.output_size[0] or label.shape[1] <= self.output_size[1] or label.shape[2] <= \
                self.output_size[2]:
            pw = max((self.output_size[0] - label.shape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - label.shape[1]) // 2 + 3, 0)
            pd = max((self.output_size[2] - label.shape[2]) // 2 + 3, 0)
            FLAIR = np.pad(FLAIR, [(pw, pw), (ph, ph), (pd, pd)],
                           mode='constant', constant_values=0)
            T1w = np.pad(T1w, [(pw, pw), (ph, ph), (pd, pd)],
                           mode='constant', constant_values=0)
            T2w = np.pad(T2w, [(pw, pw), (ph, ph), (pd, pd)],
                           mode='constant', constant_values=0)
            label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)],
                           mode='constant', constant_values=0)
            mask = np.pad(mask, [(pw, pw), (ph, ph), (pd, pd)],
                           mode='constant', constant_values=0)
            if self.with_sdf:
                sdf = np.pad(sdf, [(pw, pw), (ph, ph), (pd, pd)],
                             mode='constant', constant_values=0)

        (w, h, d) = FLAIR.shape
        # if np.random.uniform() > 0.33:
        #     w1 = np.random.randint((w - self.output_size[0])//4, 3*(w - self.output_size[0])//4)
        #     h1 = np.random.randint((h - self.output_size[1])//4, 3*(h - self.output_size[1])//4)
        # else:
        w1 = np.random.randint(0, w - self.output_size[0])
        h1 = np.random.randint(0, h - self.output_size[1])
        d1 = np.random.randint(0, d - self.output_size[2])

        label = label[w1:w1 + self.output_size[0], h1:h1 +
                      self.output_size[1], d1:d1 + self.output_size[2]]
        FLAIR = FLAIR[w1:w1 + self.output_size[0], h1:h1 +
                      self.output_size[1], d1: d1 + self.output_size[2]]
        T1w = T1w[w1:w1 + self.output_size[0], h1:h1 +
                      self.output_size[1], d1: d1 + self.output_size[2]]
        T2w = T2w[w1:w1 + self.output_size[0], h1:h1 +
                      self.output_size[1], d1:d1 + self.output_size[2]]
        mask = mask[w1:w1 + self.output_size[0], h1:h1 +
                      self.output_size[1], d1:d1 + self.output_size[2]]
        if self.with_sdf:
            sdf = sdf[w1:w1 + self.output_size[0], h1:h1 +
                      self.output_size[1], d1:d1 + self.output_size[2]]
            return {'flair': FLAIR,'t1':T1w,'t2':T2w, 'seg': label, 'sdf': sdf,'mask':mask}
        else:
            return {'flair': FLAIR,'t1':T1w,'t2':T2w, 'seg': label,'mask':mask}


class RandomRotFlip(object):
    """
    Crop randomly flip the dataset in a sample
    Args:
    output_size (int): Desired output size
    """

    def __call__(self, sample):
        FLAIR,T1w,T2w, label,mask = sample['flair'],sample['t1'],sample['t2'], sample['seg'],sample['mask']
        k = np.random.randint(0, 4)
        FLAIR = np.rot90(FLAIR, k)
        T1w = np.rot90(T1w, k)
        T2w = np.rot90(T2w, k)
        label = np.rot90(label, k)
        mask = np.rot90(mask, k)
        axis = np.random.randint(0, 2)
        FLAIR = np.flip(FLAIR, axis=axis).copy()
        T1w = np.flip(T1w, axis=axis).copy()
        T2w = np.flip(T2w, axis=axis).copy()
        label = np.flip(label, axis=axis).copy()
        mask = np.flip(mask, axis=axis).copy()

        return {'flair': FLAIR,'t1':T1w,'t2':T2w, 'seg': label,'mask':mask}


class RandomNoise(object):
    def __init__(self, mu=0, sigma=0.1):
        self.mu = mu
        self.sigma = sigma

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        noise = np.clip(self.sigma * np.random.randn(
            image.shape[0], image.shape[1], image.shape[2]), -2*self.sigma, 2*self.sigma)
        noise = noise + self.mu
        image = image + noise
        return {'image': image, 'label': label}


class CreateOnehotLabel(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        onehot_label = np.zeros(
            (self.num_classes, label.shape[0], label.shape[1], label.shape[2]), dtype=np.float32)
        for i in range(self.num_classes):
            onehot_label[i, :, :, :] = (label == i).astype(np.float32)
        return {'image': image, 'label': label, 'onehot_label': onehot_label}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        FLAIR = sample['flair']
        T1w = sample['t1']
        T2w=sample['t2']
        FLAIR = FLAIR.reshape(
            1, FLAIR.shape[0], FLAIR.shape[1], FLAIR.shape[2]).astype(np.float32)
        T1w = T1w.reshape(
            1, T1w.shape[0], T1w.shape[1], T1w.shape[2]).astype(np.float32)
        T2w = T2w.reshape(
            1, T2w.shape[0], T2w.shape[1], T2w.shape[2]).astype(np.float32)
        if 'onehot_label' in sample:
            return {'flair': torch.from_numpy(FLAIR), 't1':torch.from_numpy(T1w),'t2':torch.from_numpy(T2w),'seg': torch.from_numpy(sample['seg']).long(),"mask":torch.from_numpy(sample['mask']).long(),
                    'onehot_label': torch.from_numpy(sample['onehot_label']).long()}
        else:
            return {'flair': torch.from_numpy(FLAIR), 't1':torch.from_numpy(T1w),'t2':torch.from_numpy(T2w),'seg': torch.from_numpy(sample['seg']).long(),"mask":torch.from_numpy(sample['mask']).long()}


class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices

    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """

    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch)
            in zip(grouper(primary_iter, self.primary_batch_size),
                   grouper(secondary_iter, self.secondary_batch_size))
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size


def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)
    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)