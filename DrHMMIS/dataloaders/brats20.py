import os
import torch
import numpy as np
from glob import glob
from torch.utils.data import Dataset
import h5py
import itertools
from torch.utils.data.sampler import Sampler


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
        sample = {'flair': FLAIR,'t1':T1,'t2':T2,'t1ce':t1ce,'seg': label.astype(np.uint8)}
        if self.transform:
            sample = self.transform(sample)
        return sample
class BraTS2020_Pretrain_Augmented_Dataset(Dataset):
    """
    加载已包含伪标签的H5文件，用于高速预训练。
    [最终修正版]: 增加了稳健的维度检查和修正。
    """
    def __init__(self, base_dir, list_dir, split, transform=None):
        self.transform = transform
        self.base_dir = base_dir
        self.sample_list = []
        list_path = os.path.join(list_dir, split + '.txt')
        with open(list_path, 'r') as f:
            self.sample_list = [line.strip() for line in f.readlines()]
        print(f"成功为高速预训练的 '{split}' 部分加载了 {len(self.sample_list)} 个样本。")

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case_name = self.sample_list[idx]
        h5_path = os.path.join(self.base_dir, case_name, f"{case_name}.h5")
        
        try:
            with h5py.File(h5_path, 'r') as hf:
                image = np.array(hf['image']).astype(np.float32)
                et_mask = np.array(hf['et_mask']).astype(np.float32)
                ed_mask = np.array(hf['ed_mask']).astype(np.float32)
                ncr_mask = np.array(hf['ncr_mask']).astype(np.float32)
                unsym_mask=np.array(hf['unsym_mask']).astype(np.float32)
                # [*** 核心修正点 ***]
                # 强制检查并移除任何多余的、长度为1的维度
                image = np.squeeze(image)
                et_mask = np.squeeze(et_mask)
                ed_mask = np.squeeze(ed_mask)
                ncr_mask = np.squeeze(ncr_mask)
                unsym_mask=np.squeeze(unsym_mask)

                # 确保image最终是4D, mask是3D
                if image.ndim != 4:
                    # 可以在这里添加更详细的错误处理或打印
                    print(f"警告: 样本 {case_name} 的 'image' 维度不为4，实际为 {image.shape}")
                if et_mask.ndim != 3:
                    print(f"警告: 样本 {case_name} 的 'et_mask' 维度不为3，实际为 {et_mask.shape}")
                
                sample = {
                    'image': image,
                    'et_mask': et_mask,
                    'ed_mask': ed_mask,
                    'ncr_mask': ncr_mask,
                    'unsym_mask':unsym_mask
                }

        except Exception as e:
            print(f"错误：无法从 {h5_path} 加载数据。错误信息: {e}")
            return None
        print(sample['image'].shape,sample['ed_mask'].shape)
        if self.transform:
            sample = self.transform(sample)
        print(sample)
        return sample
class BraTS2020_Pretrain_Dataset(Dataset):
    """
    用于BraTS自监督重建预训练的数据集 (适配指定的文件结构)。
    - 从 .txt 文件读取样本列表 (子文件夹名)。
    - 从每个子文件夹中加载对应的 .h5 文件。
    - 将4个模态堆叠成一个4通道图像。
    - 应用预训练所需的数据变换。
    """
    def __init__(self, base_dir, list_dir, split, transform=None):
        """
        初始化.
        Args:
            base_dir (str): 存放所有病人子文件夹的根目录。
            list_dir (str): 存放 train.txt, val.txt 等文件的目录。
            split (str): 'train', 'val', 'test' 等，用于选择对应的 .txt 文件。
            transform (callable, optional): 应用于样本的数据变换。
        """
        self.transform = transform
        self.base_dir = base_dir
        self.sample_list = []
        
        # 构建并读取 .txt 文件
        list_path = os.path.join(list_dir, split + '.txt')
        with open(list_path, 'r') as f:
            self.sample_list = [line.strip() for line in f.readlines()]
            
        print(f"成功为 '{split}' 部分加载了 {len(self.sample_list)} 个样本。")

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        # 获取子文件夹名, 例如 "BraTS2021_00000"
        case_name = self.sample_list[idx]
        
        # [*** 核心修改点 ***]
        # 根据您描述的规则构建完整的文件路径
        # 例如: /base_dir/BraTS2021_00000/BraTS2021_00000h5file.h5
        file_name = case_name + "h5file.h5"
        h5_path = os.path.join(self.base_dir, case_name, file_name)

        try:
            with h5py.File(h5_path, 'r') as hf:
                # 从h5文件中读取4个模态的图像数据
                t1 = np.array(hf['t1'])
                t1ce = np.array(hf['t1ce'])
                t2 = np.array(hf['t2'])
                flair = np.array(hf['flair'])
                
                # 将4个单通道图像堆叠成一个4通道图像
                # 形状: (4, D, H, W)
                image = np.stack([t1, t1ce, t2, flair], axis=0).astype(np.float32)

        except Exception as e:
            print(f"错误：无法从 {h5_path} 加载数据。错误信息: {e}")
            # 返回一个空样本或进行其他错误处理
            return None

        # 创建一个包含4通道图像的字典，以供MONAI变换流程使用
        sample = {'image': image}
        
        # 应用数据变换
        if self.transform:
            sample = self.transform(sample)

        return sample

        return sample




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
        T1ce=T1ce[w1:w1 + self.output_size[0], h1:h1 +
                      self.output_size[1], d1:d1 + self.output_size[2]]
        if self.with_sdf:
            sdf = sdf[w1:w1 + self.output_size[0], h1:h1 +
                      self.output_size[1], d1:d1 + self.output_size[2]]
            return {'flair': FLAIR,'t1':T1w,'t2':T2w, 'seg': label, 'sdf': sdf,'t1ce':T1ce}
        else:
            return {'flair': FLAIR,'t1':T1w,'t2':T2w, 'seg': label,'t1ce':T1ce}


class RandomRotFlip(object):
    """
    Crop randomly flip the dataset in a sample
    Args:
    output_size (int): Desired output size
    """

    def __call__(self, sample):
        FLAIR,T1w,T2w, label,T1ce = sample['flair'],sample['t1'],sample['t2'], sample['seg'],sample['t1ce']
        k = np.random.randint(0, 4)
        FLAIR = np.rot90(FLAIR, k)
        T1w = np.rot90(T1w, k)
        T2w = np.rot90(T2w, k)
        T1ce=np.rot90(T1ce, k)
        label = np.rot90(label, k)
        axis = np.random.randint(0, 2)
        FLAIR = np.flip(FLAIR, axis=axis).copy()
        T1w = np.flip(T1w, axis=axis).copy()
        T2w = np.flip(T2w, axis=axis).copy()
        T1ce=np.flip(T1ce, axis=axis).copy()
        label = np.flip(label, axis=axis).copy()

        return {'flair': FLAIR,'t1':T1w,'t2':T2w, 'seg': label,'t1ce':T1ce}


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
        FLAIR = FLAIR.reshape(
            1, FLAIR.shape[0], FLAIR.shape[1], FLAIR.shape[2]).astype(np.float32)
        T1w = T1w.reshape(
            1, T1w.shape[0], T1w.shape[1], T1w.shape[2]).astype(np.float32)
        T2w = T2w.reshape(
            1, T2w.shape[0], T2w.shape[1], T2w.shape[2]).astype(np.float32)
        T1ce=T1ce.reshape(
            1, T1ce.shape[0], T1ce.shape[1], T1ce.shape[2]).astype(np.float32)
        if 'onehot_label' in sample:
            return {'flair': torch.from_numpy(FLAIR), 't1':torch.from_numpy(T1w),'t2':torch.from_numpy(T2w),'seg': torch.from_numpy(sample['seg']).long(),'t1ce':torch.from_numpy(T1ce),
                    'onehot_label': torch.from_numpy(sample['onehot_label']).long()}
        else:
            return {'flair': torch.from_numpy(FLAIR), 't1':torch.from_numpy(T1w),'t2':torch.from_numpy(T2w),'seg': torch.from_numpy(sample['seg']).long(),'t1ce':torch.from_numpy(T1ce)}


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