import os
import torch
import numpy as np
from glob import glob
from torch.utils.data import Dataset
import h5py
import itertools
from torch.utils.data.sampler import Sampler
import random

class BraTS2019(Dataset):
    """ BraTS2019 Dataset """

    def __init__(self, base_dir=None, split='train', num=None, transform=None,train_txt=None):
        self._base_dir = base_dir
        self.transform = transform
        self.sample_list = []

        train_path = os.path.join(self._base_dir,train_txt)#self._base_dir+'/train4.txt'
        test_path = self._base_dir+'/val_0.txt'

        if split == 'train':
            absolute_path = os.path.abspath(train_path)
            print(absolute_path)
            with open(train_path, 'r') as f:
                self.image_list = f.readlines()
        elif split == 'val':
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
        t1ce=h5f['t1ce'][:]
        label = h5f['seg'][:]
        #sample = {'flair': FLAIR,'t1':T1,'t2':T2,'t1ce':t1ce,'seg': label.astype(np.uint8)}
        sample = {'flair': FLAIR,'t1':T1,'t2':T2,'t1ce':t1ce,'seg': label.astype(np.uint8),'flair_ultra':FLAIR,'t1_ultra':T1,'t2_ultra':T2,'t1ce_ultra':t1ce}
        if self.transform:
            sample = self.transform(sample)
        return sample

class RandomMaskProcessor:
    def __init__(self, mask_size=(16, 16, 16), mask_number=8):
        """
        初始化随机掩码生成器。

        :param mask_size: 掩码的大小 (w, h, d)
        :param num_masks: 要生成的掩码数量
        """
        self.mask_size = mask_size
        self.num_masks = mask_number

    def __call__(self, image_data):
        """
        在输入图像数据上生成随机均匀分布的掩码，并返回掩码和处理后的图像数据。

        :param image_data: 输入的图像数据 (numpy array)
        :return: (掩码矩阵, 应用掩码后的图像数据)
        """
        mask = self.generate_random_masks(image_data.shape)
        masked_image = self.apply_mask_to_image(image_data, mask)
        return  masked_image

    def generate_random_masks(self, image_shape):
        """
        随机生成均匀分布的掩码矩阵，确保掩码互不重叠。

        :param image_shape: 原始图像的形状 (W, H, D)
        :return: 与原图形状相同的掩码矩阵
        """
        mask = np.zeros(image_shape, dtype=np.uint8)
        w, h, d = image_shape
        mw, mh, md = self.mask_size
        #print(w,h,d,mw,mh,md)
        positions = set()  # 用于记录掩码起始点，避免重复

        while len(positions) < self.num_masks:
            start_w = random.randint(0, w - mw)
            start_h = random.randint(0, h - mh)
            start_d = random.randint(0, d - md)
            position = (start_w, start_h, start_d)
            if position not in positions:
                positions.add(position)
                mask[start_w:start_w + mw, start_h:start_h + mh, start_d:start_d + md] = 1

        return mask

    def apply_mask_to_image(self, image, mask):
        """
        将掩码应用到原始图像上，将掩码区域设置为 0。

        :param image: 原始图像数据 (numpy array)
        :param mask: 掩码矩阵 (numpy array)
        :return: 应用掩码后的图像
        """
        masked_image = image.copy()
        masked_image[mask == 1] = 0  # 将掩码区域设置为 0（黑色）
        return masked_image
    


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

    def __init__(self, output_size,mask_size ,mask_number,with_sdf=False):
        self.output_size = output_size
        self.with_sdf = with_sdf
        self.mask=RandomMaskProcessor(mask_size=mask_size,mask_number=mask_number)
    def __call__(self, sample):
        FLAIR,T1w,T2w, label,T1ce = sample['flair'],sample['t1'],sample['t2'], sample['seg'],sample['t1ce']
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
            T1ce=np.pad(T1ce, [(pw, pw), (ph, ph), (pd, pd)],
                           mode='constant', constant_values=0)
            label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)],
                           mode='constant', constant_values=0)
            if self.with_sdf:
                sdf = np.pad(sdf, [(pw, pw), (ph, ph), (pd, pd)],
                             mode='constant', constant_values=0)

        (w, h, d) = FLAIR.shape
        midline=w//2
        # if np.random.uniform() > 0.33:
        #     w1 = np.random.randint((w - self.output_size[0])//4, 3*(w - self.output_size[0])//4)
        #     h1 = np.random.randint((h - self.output_size[1])//4, 3*(h - self.output_size[1])//4)
        # else:
        w1 = np.random.randint(0, w - self.output_size[0])
        h1 = np.random.randint(0, h - self.output_size[1])
        d1 = np.random.randint(0, d - self.output_size[2])
        label_1 = label[w1:w1 + self.output_size[0], h1:h1 +
                      self.output_size[1], d1:d1 + self.output_size[2]]
        FLAIR_1 = FLAIR[w1:w1 + self.output_size[0], h1:h1 +
                      self.output_size[1], d1: d1 + self.output_size[2]]
        T1w_1 = T1w[w1:w1 + self.output_size[0], h1:h1 +
                      self.output_size[1], d1: d1 + self.output_size[2]]
        T2w_1 = T2w[w1:w1 + self.output_size[0], h1:h1 +
                      self.output_size[1], d1:d1 + self.output_size[2]]
        T1ce_1=T1ce[w1:w1 + self.output_size[0], h1:h1 +
                      self.output_size[1], d1:d1 + self.output_size[2]]
        symmetric_w1 = 2 * midline - (w1 + self.output_size[0])
        FLAIR_symmetry=FLAIR[symmetric_w1:symmetric_w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        T1w_symmetry=T1w[symmetric_w1:symmetric_w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        T2w_symmetry=T2w[symmetric_w1:symmetric_w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        T1ce_symmetry=T1ce[symmetric_w1:symmetric_w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        FLAIR_symmetry=np.flip(FLAIR_symmetry,axis=0)
        T1w_symmetry=np.flip(T1w_symmetry,axis=0)
        T2w_symmetry=np.flip(T2w_symmetry,axis=0)
        T1ce_symmetry=np.flip(T1ce_symmetry,axis=0)
        T1_sym_mask=self.mask(T1w_symmetry)
        FLAIR_symmetry_mask=self.mask(FLAIR_symmetry)
        if self.with_sdf:
            sdf = sdf[w1:w1 + self.output_size[0], h1:h1 +
                      self.output_size[1], d1:d1 + self.output_size[2]]
            return {'flair': FLAIR_1,'t1':T1w_1,'t2':T2w_1, 'seg': label_1, 'sdf': sdf,'t1ce':T1ce_1,'flair_sym':FLAIR_symmetry,'t1_sym':T1w_symmetry,'t2_sym':T2w_symmetry,'t1ce_sym':T1ce_symmetry,'t1_sym_mask':T1_sym_mask,'flair_sym_mask':FLAIR_symmetry_mask}
        else:
            return {'flair': FLAIR_1,'t1':T1w_1,'t2':T2w_1, 'seg': label_1,'t1ce':T1ce_1,'flair_sym':FLAIR_symmetry,'t1_sym':T1w_symmetry,'t2_sym':T2w_symmetry,'t1ce_sym':T1ce_symmetry,'t1_sym_mask':T1_sym_mask,'flair_sym_mask':FLAIR_symmetry_mask}


class RandomRotFlip(object):
    """
    Crop randomly flip the dataset in a sample
    Args:
    output_size (int): Desired output size
    """

    def __call__(self, sample):
        FLAIR,T1w,T2w, label,T1ce,FLAIR_sym,t1_sym,t2_sym,t1ce_sym,t1_sym_mask,flair_sym_mask = sample['flair'],sample['t1'],sample['t2'], sample['seg'],sample['t1ce'],sample['flair_sym'],sample['t1_sym'],sample['t2_sym'],sample['t1ce_sym'],sample['t1_sym_mask'],sample['flair_sym_mask']
        k = np.random.randint(0, 4)
        FLAIR = np.rot90(FLAIR, k)
        T1w = np.rot90(T1w, k)
        T2w = np.rot90(T2w, k)
        T1ce=np.rot90(T1ce, k)
        label = np.rot90(label, k)
        FLAIR_sym = np.rot90(FLAIR_sym, k)
        T1w_sym = np.rot90(t1_sym, k)
        T2w_sym = np.rot90(t2_sym, k)
        T1ce_sym=np.rot90(t1ce_sym, k)
        t1_sym_mask=np.rot90(t1_sym_mask,k)
        flair_sym_mask=np.rot90(flair_sym_mask,k)
        axis = np.random.randint(0, 2)
        FLAIR = np.flip(FLAIR, axis=axis).copy()
        T1w = np.flip(T1w, axis=axis).copy()
        T2w = np.flip(T2w, axis=axis).copy()
        T1ce=np.flip(T1ce, axis=axis).copy()
        label = np.flip(label, axis=axis).copy()
        FLAIR_sym= np.flip(FLAIR_sym, axis=axis).copy()
        T1w_sym = np.flip(T1w_sym, axis=axis).copy()
        T2w_sym = np.flip(T2w_sym, axis=axis).copy()
        T1ce_sym=np.flip(T1ce_sym, axis=axis).copy()
        t1_sym_mask=np.flip(t1_sym_mask, axis=axis).copy()
        flair_sym_mask=np.flip(flair_sym_mask, axis=axis).copy()
        return {'flair': FLAIR,'t1':T1w,'t2':T2w, 'seg': label,'t1ce':T1ce,'flair_sym':FLAIR_sym,'t1_sym':T1w_sym,'t2_sym':T2w_sym,"t1ce_sym":T1ce_sym,'t1_sym_mask':t1_sym_mask,'flair_sym_mask':flair_sym_mask}


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
        T1ce=sample['t1ce']
        FLAIR_sym=sample['flair_sym']
        T1w_sym=sample['t1_sym']
        T2w_sym=sample['t2_sym']
        T1ce_sym=sample['t1ce_sym']
        t1_sym_mask=sample['t1_sym_mask']
        flair_sym_mask=sample['flair_sym_mask']
        FLAIR = FLAIR.reshape(
            1, FLAIR.shape[0], FLAIR.shape[1], FLAIR.shape[2]).astype(np.float32)
        T1w = T1w.reshape(
            1, T1w.shape[0], T1w.shape[1], T1w.shape[2]).astype(np.float32)
        T2w = T2w.reshape(
            1, T2w.shape[0], T2w.shape[1], T2w.shape[2]).astype(np.float32)
        T1ce=T1ce.reshape(
            1, T1ce.shape[0], T1ce.shape[1], T1ce.shape[2]).astype(np.float32)
        FLAIR_sym = FLAIR_sym.reshape(
            1, FLAIR_sym.shape[0], FLAIR_sym.shape[1], FLAIR_sym.shape[2]).astype(np.float32)
        T1w_sym = T1w_sym.reshape(
            1, T1w_sym.shape[0], T1w_sym.shape[1], T1w_sym.shape[2]).astype(np.float32)
        T2w_sym = T2w_sym.reshape(
            1, T2w_sym.shape[0], T2w_sym.shape[1], T2w_sym.shape[2]).astype(np.float32)
        T1ce_sym=T1ce_sym.reshape(
            1, T1ce_sym.shape[0], T1ce_sym.shape[1], T1ce_sym.shape[2]).astype(np.float32)
        t1_sym_mask=t1_sym_mask.reshape(
            1, t1_sym_mask.shape[0], t1_sym_mask.shape[1],t1_sym_mask.shape[2]).astype(np.float32)
        flair_sym_mask=flair_sym_mask.reshape(
            1, flair_sym_mask.shape[0], flair_sym_mask.shape[1],flair_sym_mask.shape[2]).astype(np.float32)
        if 'onehot_label' in sample:
            return {'flair': torch.from_numpy(FLAIR), 't1':torch.from_numpy(T1w),'t2':torch.from_numpy(T2w),'seg': torch.from_numpy(sample['seg']).long(),'t1ce':torch.from_numpy(T1ce),'flair_sym': torch.from_numpy(FLAIR_sym), 't1_sym':torch.from_numpy(T1w_sym),'t2_sym':torch.from_numpy(T2w_sym),'t1ce_sym':torch.from_numpy(T1ce_sym),'t1_sym_mask':torch.from_numpy(t1_sym_mask),'flair_sym_mask':torch.from_numpy(flair_sym_mask),
                    'onehot_label': torch.from_numpy(sample['onehot_label']).long()}
        else:
            return {'flair': torch.from_numpy(FLAIR), 't1':torch.from_numpy(T1w),'t2':torch.from_numpy(T2w),'seg': torch.from_numpy(sample['seg']).long(),'t1ce':torch.from_numpy(T1ce),'flair_sym': torch.from_numpy(FLAIR_sym), 't1_sym':torch.from_numpy(T1w_sym),'t2_sym':torch.from_numpy(T2w_sym),'t1ce_sym':torch.from_numpy(T1ce_sym),'t1_sym_mask':torch.from_numpy(t1_sym_mask),'flair_sym_mask':torch.from_numpy(flair_sym_mask)}


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