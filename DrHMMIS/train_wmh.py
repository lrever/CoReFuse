import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
import random
import logging
import numpy as np
import nibabel as nib
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose
from tqdm import tqdm 
import SimpleITK as sitk
import scipy.spatial
# 引入 MONAI 组件
from monai.losses import FocalLoss
from monai.networks.nets import SwinUNETR
from monai.metrics import compute_hausdorff_distance
from networks.dual_encoder import Optimized_DynamicModal_Net
from monai.inferers import sliding_window_inference
from collections import defaultdict
import scipy.ndimage as ndimage
from networks.dual_enocder_unet import Ablation_NEncoder_Final_Net
from networks.context_lite import Ablation_ThreeEncoder_Final_Net
# ==============================================================================
# 1. 数据增强与预处理 (Transforms)
# ==============================================================================
def setup_logging(log_file='train_log.txt'):
    """
    配置日志：同时输出到控制台和文件
    """
    # 1. 定义格式
    log_format = "%(asctime)s - %(levelname)s - %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"
    
    # 2. 配置 Root Logger
    logging.basicConfig(
        level=logging.INFO, # 关键！必须设置为 INFO，否则默认只显示 WARNING
        format=log_format,
        datefmt=date_format,
        handlers=[
            logging.FileHandler(log_file),    # 输出到文件
            logging.StreamHandler(sys.stdout) # 输出到控制台
        ]
    )
    
    # 返回 logger 实例（虽然直接用 logging.info 也可以，但这样更规范）
    return logging.getLogger(__name__)



class IntensityNormalize(object):
    """ (保持不变) Z-Score 归一化 """
    def __init__(self, eps=1e-6):
        self.eps = eps

    def __call__(self, sample):
        for key in ['flair', 't1']:
            img = sample[key]
            mask = img > 0
            if mask.sum() == 0: continue
            mean = img[mask].mean()
            std = img[mask].std()
            img = (img - mean) / (std + self.eps)
            img[~mask] = 0
            sample[key] = img
        return sample


class RobustIntensityNormalize(object):
    """
    升级版归一化：
    1. 不依赖简单的 >0，而是使用分位数 (Percentile) 确定前景。
    2. 这样可以自动忽略背景噪声和颅骨的高亮干扰，使 Z-Score 更聚焦于脑实质。
    """
    def __init__(self, eps=1e-6, roi_percentile=10):
        self.eps = eps
        # 认为强度低于全图 10% 分位数的像素是背景/空气/噪声
        # 这比固定阈值 (30/70) 更能适应不同扫描仪
        self.roi_percentile = roi_percentile

    def __call__(self, sample):
        for key in ['flair', 't1']:
            img = sample[key]
            
            # --- 步骤 1: 确定更鲁棒的 Mask ---
            # 计算全图的低分位数作为阈值 (自适应阈值)
            # 作用：过滤掉背景中的低强度噪声，模拟文中提到的 "Thresholding"
            threshold = np.percentile(img, self.roi_percentile)
            
            # 生成掩码：只有大于阈值的区域才参与 Mean/Std 计算
            mask = img > threshold
            
            # 防御性检查
            if mask.sum() == 0:
                # 如果全黑，回退到全图
                mean = img.mean()
                std = img.std()
            else:
                # --- 步骤 2: 仅在 ROI 内计算统计量 ---
                # 这样颅骨（虽然高亮但面积小）和背景（面积大但值低）对统计量的影响被平衡了
                # 或者，如果你想更激进地排除颅骨，可以计算 img[mask] 的 [0, 95] 分位数的 mean/std
                pixels = img[mask]
                mean = pixels.mean()
                std = pixels.std()

            # --- 步骤 3: Z-Score ---
            img = (img - mean) / (std + self.eps)
            
            # --- 步骤 4: 背景置零 (可选) ---
            # 将低于阈值的区域强制置 0，模拟“软去颅/去背景”
            # 注意：归一化后背景变成了负数，这里我们用 mask 把背景切掉，让网络只看脑子
            # 这符合文中 "最大连通组件" 的思想：只保留脑子区域
            img[~mask] = 0 
            
            sample[key] = img
            
        return sample
    
class AdvancedModalityNormalize(object):
    """
    针对不同模态特性的高级归一化：
    FLAIR: 属于高信号病灶，保留高分位数信息，截断极值伪影。
    T1: 属于解剖结构，去除头皮脂肪高亮干扰，拉伸脑实质对比度。
    """
    def __init__(self, output_range=(0, 1)):
        self.output_range = output_range

    def __call__(self, sample):
        # 1. 处理 FLAIR
        if 'flair' in sample:
            img = sample['flair']
            mask = img > 0
            if mask.sum() > 0:
                # FLAIR 策略: 截断 99.9% (保留病灶，去噪点)
                # 下限取 1% (去背景噪点)
                v_min = np.percentile(img[mask], 1)
                v_max = np.percentile(img[mask], 99.9)
                
                # 截断
                img = np.clip(img, v_min, v_max)
                
                # Min-Max 归一化到 [0, 1]
                img = (img - v_min) / (v_max - v_min + 1e-8)
                
                # 背景复原为 0
                img[~mask] = 0
            sample['flair'] = img

        # 2. 处理 T1
        if 't1' in sample:
            img = sample['t1']
            mask = img > 0
            if mask.sum() > 0:
                # T1 策略: 截断 99.0% (T1的高亮通常是血管/脂肪，切掉有助于突出脑实质)
                v_min = np.percentile(img[mask], 1)
                v_max = np.percentile(img[mask], 99.0)
                
                img = np.clip(img, v_min, v_max)
                img = (img - v_min) / (v_max - v_min + 1e-8)
                img[~mask] = 0
            sample['t1'] = img
            
        return sample

class ModalitySpecificAugmentation(object):
    """
    针对模态特性的增强
    FLAIR: 重点增强 Gamma (模拟病灶亮度变化)
    T1: 重点增强 噪声 (模拟成像质量)
    """
    def __init__(self, prob=0.3):
        self.prob = prob

    def __call__(self, sample):
        # 1. Random Gamma only for FLAIR (改变病灶显著性)
        if np.random.rand() < self.prob and 'flair' in sample:
            gamma = np.random.uniform(0.7, 1.3) # 0.7变亮(病灶更显), 1.3变暗(病灶更隐蔽)
            img = sample['flair']
            # 确保在 [0,1] 范围内做 gamma
            sample['flair'] = np.power(img, gamma)

        # 2. Random Noise mainly for T1 (模拟结构不清)
        if np.random.rand() < self.prob and 't1' in sample:
            noise_std = np.random.uniform(0.0, 0.05)
            noise = np.random.normal(0, noise_std, sample['t1'].shape)
            sample['t1'] = sample['t1'] + noise
            # 加上噪声后可能会越界，Clip 一下
            sample['t1'] = np.clip(sample['t1'], 0, 1)
            
        return sample
class RandomRotFlip(object):
    """ 
    (更新) 随机旋转和翻转 
    关键修正: 如果旋转了图像，必须同步交换 Spacing 的 X 和 Y
    """
    def __call__(self, sample):
        FLAIR, T1w, label = sample['flair'], sample['t1'], sample['seg']
        spacing = sample['spacing'] # [sx, sy, sz]
        
        # 1. 随机旋转 90度 (XY平面)
        # 旋转 90度 或 270度 (k=1, k=3) 会导致长宽互换，Spacing X/Y 也要互换
        k = np.random.randint(0, 4)
        FLAIR = np.rot90(FLAIR, k).copy()
        T1w = np.rot90(T1w, k).copy()
        label = np.rot90(label, k).copy()
        
        if k % 2 != 0: # 旋转了 90 或 270 度
            spacing[0], spacing[1] = spacing[1], spacing[0]
        
        # 2. 随机翻转 (翻转不改变 spacing 大小，只改变方向，magnitude 不变)
        axis = np.random.randint(0, 2)
        if np.random.random() > 0.5:
            FLAIR = np.flip(FLAIR, axis=axis).copy()
            T1w = np.flip(T1w, axis=axis).copy()
            label = np.flip(label, axis=axis).copy()

        # 更新回去
        sample['flair'] = FLAIR
        sample['t1'] = T1w
        sample['seg'] = label
        sample['spacing'] = spacing
        return sample

class RandomGaussianNoise(object):
    """
    添加随机高斯噪声，模拟不同的 SNR (信噪比)。
    """
    def __init__(self, mean=0.0, std_range=(0.0, 0.1)):
        self.mean = mean
        self.std_range = std_range

    def __call__(self, sample):
        # 以一定的概率执行 (例如 15%)
        if np.random.uniform() > 0.15:
            return sample

        std = np.random.uniform(*self.std_range)
        
        for key in ['flair', 't1']:
            img = sample[key]
            mask = img > 0 # 只在脑内加噪声，或者全图加都行
            
            noise = np.random.normal(self.mean, std, img.shape)
            
            # 加噪声并保持原始数据范围大致不变
            img_noised = img + noise
            
            # 如果你有 mask 约束，可以只保留 mask 内的噪声
            # img[mask] = img_noised[mask] 
            # 但通常全图加更自然
            sample[key] = img_noised
            
        return sample

class RandomGamma(object):
    """
    非线性亮度调整: I_new = I_old ^ gamma
    gamma < 1: 图像变亮，低灰度区域对比度拉伸 (模拟低对比度扫描)
    gamma > 1: 图像变暗，高灰度区域对比度拉伸 (模拟高对比度扫描)
    """
    def __init__(self, gamma_range=(0.7, 1.5)):
        self.gamma_range = gamma_range

    def __call__(self, sample):
        if np.random.uniform() > 0.3: # 30% 概率执行
            return sample

        gamma = np.random.uniform(*self.gamma_range)
        
        for key in ['flair', 't1']:
            img = sample[key]
            
            # Gamma 变换通常在 [0, 1] 范围内进行效果最好
            # 先归一化到 [0, 1] (临时)
            v_min, v_max = img.min(), img.max()
            if v_max - v_min == 0: continue
            
            img_norm = (img - v_min) / (v_max - v_min)
            
            # 执行变换 (加上 epsilon 防止 log(0))
            img_gamma = np.power(img_norm + 1e-7, gamma)
            
            # 还原回原来的范围 (可选，或者直接让后续的 IntensityNormalize 处理)
            # 这里还原回去比较安全
            img = img_gamma * (v_max - v_min) + v_min
            
            sample[key] = img
            
        return sample
    
class RandomElasticDeformation(object):
    """
    弹性形变：生成平滑的随机位移场，对图像进行非刚体扭曲。
    这是医学图像分割中最强大的增强方法之一。
    """
    def __init__(self, alpha_range=(500, 800), sigma_range=(30, 50)):
        # alpha 控制形变强度，sigma 控制平滑程度
        self.alpha_range = alpha_range
        self.sigma_range = sigma_range

    def __call__(self, sample):
        if np.random.uniform() > 0.2: # 20% 概率
            return sample

        flair = sample['flair']
        # 获取 patch 的形状
        shape = flair.shape[1:] # (D, H, W) 因为你的 shape 是 (1, 128, 128, 48) ? 需确认维度
        # 如果 sample['flair'] 是 (1, W, H, D) 或者 (W, H, D)
        # 假设输入已经是 Patch 后的 numpy array (C, W, H, D) 或 (W, H, D)
        
        # 为了通用性，这里假设输入是 (C, W, H, D) 或 (W, H, D)
        # 我们只对空间维度生成位移场
        spatial_shape = flair.shape[-3:] 
        
        alpha = np.random.uniform(*self.alpha_range)
        sigma = np.random.uniform(*self.sigma_range)

        # 生成随机位移场
        dx = ndimage.gaussian_filter((np.random.rand(*spatial_shape) * 2 - 1), sigma) * alpha
        dy = ndimage.gaussian_filter((np.random.rand(*spatial_shape) * 2 - 1), sigma) * alpha
        dz = ndimage.gaussian_filter((np.random.rand(*spatial_shape) * 2 - 1), sigma) * alpha

        # 构建网格
        x, y, z = np.meshgrid(np.arange(spatial_shape[0]), 
                              np.arange(spatial_shape[1]), 
                              np.arange(spatial_shape[2]), indexing='ij')
        indices = np.reshape(x+dx, (-1, 1)), np.reshape(y+dy, (-1, 1)), np.reshape(z+dz, (-1, 1))

        # 应用形变
        # order=3 (双立方插值) 用于图像, order=0 (最近邻) 用于标签
        # 注意：这里需要对每个通道分别处理
        
        for key in ['flair', 't1']:
            # 假设 shape 是 (1, W, H, D)
            img = sample[key][0] 
            img_deformed = ndimage.map_coordinates(img, indices, order=3, mode='reflect').reshape(spatial_shape)
            sample[key][0] = img_deformed

        # 标签必须用 Nearest Neighbor (order=0)，防止引入不存在的类别小数
        seg = sample['seg'][0]
        seg_deformed = ndimage.map_coordinates(seg, indices, order=0, mode='reflect').reshape(spatial_shape)
        sample['seg'][0] = seg_deformed

        return sample

from monai.transforms import RandAffine
from monai.utils import GridSampleMode, GridSamplePadMode
from monai.transforms import RandAffined


class BrainRegionRandomCrop(object):
    """ (保持不变) 随机裁剪 """
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        FLAIR, T1w, label = sample['flair'], sample['t1'], sample['seg']
        # Spacing 在裁剪时不需要改变，因为像素间距没变
        
        # --- Padding ---
        for axis in range(3):
            if FLAIR.shape[axis] < self.output_size[axis]:
                diff = self.output_size[axis] - FLAIR.shape[axis]
                pad_l = diff // 2
                pad_r = diff - pad_l
                pads = [(0,0), (0,0), (0,0)]
                pads[axis] = (pad_l, pad_r)
                FLAIR = np.pad(FLAIR, pads, mode='constant', constant_values=0)
                T1w   = np.pad(T1w, pads, mode='constant', constant_values=0)
                label = np.pad(label, pads, mode='constant', constant_values=0)

        # --- Bounding Box ---
        coords = np.argwhere(FLAIR > 0)
        if len(coords) > 0:
            x_min, y_min, z_min = coords.min(axis=0)
            x_max, y_max, z_max = coords.max(axis=0)
        else:
            x_min, y_min, z_min = 0, 0, 0
            x_max, y_max, z_max = FLAIR.shape

        # --- Random Center ---
        w, h, d = FLAIR.shape
        ow, oh, od = self.output_size
        rand_x_min = max(0, x_min - ow // 2)
        rand_x_max = min(w - ow, x_max - ow // 2)
        rand_y_min = max(0, y_min - oh // 2)
        rand_y_max = min(h - oh, y_max - oh // 2)
        rand_z_min = max(0, z_min - od // 2)
        rand_z_max = min(d - od, z_max - od // 2)
        
        if rand_x_max <= rand_x_min: rand_x_min, rand_x_max = 0, w - ow
        if rand_y_max <= rand_y_min: rand_y_min, rand_y_max = 0, h - oh
        if rand_z_max <= rand_z_min: rand_z_min, rand_z_max = 0, d - od

        x1 = np.random.randint(rand_x_min, rand_x_max + 1)
        y1 = np.random.randint(rand_y_min, rand_y_max + 1)
        z1 = np.random.randint(rand_z_min, rand_z_max + 1)

        # --- Crop ---
        sample['flair'] = FLAIR[x1:x1+ow, y1:y1+oh, z1:z1+od]
        sample['t1']    = T1w[x1:x1+ow, y1:y1+oh, z1:z1+od]
        sample['seg']   = label[x1:x1+ow, y1:y1+oh, z1:z1+od]

        return sample

# ==========================================
# 2. Dataset (读取 Spacing)
# ==========================================

def load_nii_with_spacing(path):
    """ 同时读取数据和 spacing """
    img_obj = nib.load(path)
    data = img_obj.get_fdata().astype(np.float32)
    # 获取 voxel dimensions (sx, sy, sz)
    spacing = img_obj.header.get_zooms()[:3] 
    return data, np.array(spacing, dtype=np.float32)

class WMHDataset(Dataset):
    def __init__(self, root_dir, split='training', transform=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.samples = []
        
        split_dir = os.path.join(root_dir, split)
        if not os.path.exists(split_dir): raise ValueError(f"Path not found: {split_dir}")

        for center in sorted(os.listdir(split_dir)):
            center_dir = os.path.join(split_dir, center)
            if not os.path.isdir(center_dir): continue
            
            # 特殊处理 Amsterdam (它下面还有 Scanner 子文件夹)
            if center == 'Amsterdam':
                for scanner in os.listdir(center_dir):
                    s_dir = os.path.join(center_dir, scanner)
                    if os.path.isdir(s_dir): 
                        # 标记为 "Amsterdam_GE3T" 等
                        self._collect_cases(s_dir, domain_label=f"{center}_{scanner}")
            else:
                # 其他中心直接用名字，如 "Singapore", "Utrecht"
                self._collect_cases(center_dir, domain_label=center)
                
        print(f"[{split}] Loaded {len(self.samples)} samples.")

    def _collect_cases(self, base_dir, domain_label):
        for case in os.listdir(base_dir):
            case_dir = os.path.join(base_dir, case)
            if not os.path.isdir(case_dir): continue
            
            flair_p = os.path.join(case_dir, 'pre', 'FLAIR.nii.gz')
            t1_p = os.path.join(case_dir, 'pre', 'T1.nii.gz')
            seg_p = os.path.join(case_dir, 'wmh.nii.gz')

            if all(os.path.exists(p) for p in [flair_p, t1_p, seg_p]):
                self.samples.append({
                    'flair': flair_p, 
                    't1': t1_p, 
                    'seg': seg_p,
                    'center': domain_label # <--- 新增：记录来源
                })

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        flair, sp = load_nii_with_spacing(item['flair'])
        t1, _ = load_nii_with_spacing(item['t1'])
        seg, _ = load_nii_with_spacing(item['seg'])

        sample = {
            'flair': flair, 
            't1': t1, 
            'seg': seg, 
            'spacing': sp,
            'center': item['center'] # <--- 传递出去
        }
        
        if self.transform: sample = self.transform(sample)
        
        # Add channel dim
        sample['flair'] = sample['flair'][np.newaxis, ...]
        sample['t1'] = sample['t1'][np.newaxis, ...]
        sample['seg'] = sample['seg'][np.newaxis, ...]
        
        return sample
# ==============================================================================
# 3. 损失函数 (Loss Functions)
# ==============================================================================

class UserDiceLoss(nn.Module):
    """ 用户提供的 Dice Loss，针对 One-hot 编码 """
    def __init__(self, n_classes):
        super(UserDiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob)
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        return 1 - loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax: inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None: weight = [1] * self.n_classes
        
        loss = 0.0
        for i in range(self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            loss += dice * weight[i]
        return loss / self.n_classes

class TverskyLoss(nn.Module):
    def __init__(self, n_classes, alpha=0.3, beta=0.7, smooth=1e-5):
        """
        Tversky Loss: 
        alpha: FP (误检) 权重
        beta:  FN (漏检) 权重
        """
        super(TverskyLoss, self).__init__()
        self.n_classes = n_classes
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def _one_hot_encoder(self, input_tensor):
        """
        将整数标签 (B, D, H, W) 转换为 One-Hot (B, C, D, H, W)
        """
        tensor_list = []
        for i in range(self.n_classes):
            # 创建 mask: (B, D, H, W)
            temp = (input_tensor == i)
            # 增加 channel 维: (B, 1, D, H, W)
            tensor_list.append(temp.unsqueeze(1))
        
        # 拼接: (B, C, D, H, W)
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        
        # ====================================================
        # 🔧 修复核心：维度检查与 One-Hot 转换
        # ====================================================
        # 如果 target 是 (B, 1, D, H, W)，需要先 squeeze 掉 channel 1，再转 One-Hot
        if target.dim() == 5 and target.shape[1] == 1:
            target = target.squeeze(1).long()  # 变成 (B, D, H, W)
            target = self._one_hot_encoder(target) # 变成 (B, C, D, H, W)
        
        # 如果 target 是 (B, D, H, W)，直接转 One-Hot
        elif target.dim() == 4:
            target = target.long()
            target = self._one_hot_encoder(target)
            
        # 此时 target 应该是 (B, C, D, H, W)，其中 C = n_classes
        # ====================================================

        if weight is None:
            weight = [1] * self.n_classes
            
        loss = 0.0
        for i in range(self.n_classes):
            # 获取第 i 类的预测概率(p)和真实标签(t)
            p_i = inputs[:, i]
            t_i = target[:, i] # 现在这里不会报错了，因为 target 已经有 C 个通道了
            
            # 计算 TP, FP, FN
            tp = (p_i * t_i).sum()
            fp = (p_i * (1 - t_i)).sum()
            fn = ((1 - p_i) * t_i).sum()
            
            # Tversky 系数计算
            tversky_index = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
            
            # 累加加权 Loss
            loss += (1 - tversky_index) * weight[i]
            
        return loss / self.n_classes


class CombinedLoss(nn.Module):
    """ Dice + Focal 混合 Loss，支持 Label 屏蔽 """
    def __init__(self, n_classes=3, dice_weight=1.0, focal_weight=1.0):
        super(CombinedLoss, self).__init__()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.dice_func = UserDiceLoss(n_classes=n_classes)
        
        # Focal Loss 权重屏蔽 Label 2 (Bg:0.1, WMH:1.0, Other:0.0)
        # 修正 1: 在初始化时定义权重
        self.focal_weights = torch.tensor([0.1, 1.0, 0.0], dtype=torch.float32)
        
        # 修正 2: 在初始化 FocalLoss 时就传入 weight
        self.focal_func = FocalLoss(to_onehot_y=True, gamma=2.0, weight=self.focal_weights)

    def forward(self, inputs, targets, class_weights):
        # 1. Dice Loss (Inputs是Logits，需要Softmax)
        loss_dice = self.dice_func(inputs, targets, weight=class_weights, softmax=True)
        
        # 修正 3: 设备对齐 (Device Alignment)
        # 如果 FocalLoss 内部的 weight 还在 CPU，而 inputs 在 GPU，这里手动同步一下
        if self.focal_func.weight is not None:
             if self.focal_func.weight.device != inputs.device:
                self.focal_func.weight = self.focal_func.weight.to(inputs.device)

        # 2. Focal Loss (MONAI Focal 内部处理 Logits)
        # 修正 4: forward 函数不再传 weight 参数
        loss_focal = self.focal_func(inputs, targets)
        
        return self.dice_weight * loss_dice + self.focal_weight * loss_focal

class CombinedLoss_Optimized(nn.Module):
    """ 
    [升级版] Tversky + Focal 混合 Loss 
    优化目标：提升 WMH Recall，并开启 Label 2 辅助训练
    """
    def __init__(self, n_classes=3, tversky_weight=1.0, focal_weight=1.0):
        super(CombinedLoss_Optimized, self).__init__()
        self.tversky_weight = tversky_weight
        self.focal_weight = focal_weight
        
        # 1. 使用 Tversky Loss 替代 Dice Loss
        # 设置 beta=0.7, alpha=0.3 以强力提升 Recall
        self.tversky_func = TverskyLoss(n_classes=n_classes, alpha=0.45, beta=0.55)
        
        # 2. Focal Loss 权重策略调整 (开启 Label 2)
        # 背景(0): 0.1 (保持低关注)1q
        # WMH(1):  1.0 (核心关注)
        # Other(2): 0.5 (开启！让模型学会区分这是“其他病变”而不是背景，也不是WMH)
        self.focal_weights = torch.tensor([0.1, 1.0, 0.0], dtype=torch.float32)
        
        self.focal_func = FocalLoss(to_onehot_y=True, gamma=2.0, weight=self.focal_weights)

    def forward(self, inputs, targets, class_weights=None):
        """
        class_weights: 用于 Tversky Loss 的外部权重，建议设为 [0.1, 1.0, 0.5]
        """
        # 如果外部没传，就用默认的
        if class_weights is None:
            class_weights = [0.1, 1.0, 0.1]

        # 1. Tversky Loss (主要负责形状和 Recall)
        loss_tversky = self.tversky_func(inputs, targets, weight=class_weights, softmax=True)            
        
        # 设备对齐
        if self.focal_func.weight is not None:
             if self.focal_func.weight.device != inputs.device:
                self.focal_func.weight = self.focal_func.weight.to(inputs.device)

        # 2. Focal Loss (主要负责难易样本挖掘)
        loss_focal = self.focal_func(inputs, targets)
        
        return self.tversky_weight * loss_tversky + self.focal_weight * loss_focal

#####################绘图##########################
import matplotlib.pyplot as plt
import os

def plot_loss_curve(train_losses, val_losses, val_interval, save_path):
    """
    绘制并保存训练和验证 Loss 曲线
    """
    plt.figure(figsize=(10, 5))
    
    # 训练集 X 轴: [1, 2, 3, ...]
    epochs_train = range(1, len(train_losses) + 1)
    
    # 验证集 X 轴: [2, 4, 6, ...] (根据 interval 调整)
    epochs_val = [i * val_interval for i in range(1, len(val_losses) + 1)]
    
    # 绘制曲线
    plt.plot(epochs_train, train_losses, label='Train Loss', color='blue', alpha=0.6)
    plt.plot(epochs_val, val_losses, label='Val Loss', color='red', marker='.', linestyle='-')
    
    plt.title(f'Training vs Validation Loss (Val every {val_interval} epochs)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # 保存图片
    plt.savefig(save_path)
    plt.close() # 关闭画布，防止内存泄漏
########################################################

# ==============================================================================
# 4. 指标计算与日志 (Metrics & Logging)
# ==============================================================================

def setup_logger(save_dir):
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    logger = logging.getLogger("WMH_Train")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh = logging.FileHandler(os.path.join(save_dir, 'train_log.txt'))
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger

class MetricCalculator:
    """
    WMH Challenge 官方标准评价类 (支持 Spacing 物理坐标转换)
    """

    def _get_dsc_simple(self, pred, target):
        """
        快速像素级 Dice 计算 (用于常规 Epoch 监控)
        逻辑：(2 * 交集) / (预测面积 + 真实面积)
        与官方像素 Dice 逻辑完全对齐。
        """
        if torch.is_tensor(pred): pred = pred.detach().cpu().numpy()
        if torch.is_tensor(target): target = target.detach().cpu().numpy()
        
        # 展平为一维
        p = pred.flatten().astype(np.float32)
        t = target.flatten().astype(np.float32)
        
        intersection = np.sum(p * t)
        sum_total = np.sum(p) + np.sum(t)
        
        if sum_total == 0:
            return 1.0  # 官方逻辑：如果两者都为空，则认为完美匹配
        
        return (2.0 * intersection) / sum_total

    def calculate(self, preds, targets, spacings):
        """
        全量指标计算 (用于突破记录或周期性评估)
        涵盖：Dice, HD95, AVD, Recall, Lesion-F1
        Args:
            preds: (B, 1, D, H, W) Tensor, 且已经执行过 pred_wmh[targets==2]=0
            targets: (B, 1, D, H, W) Tensor, 且为 targets==1 的二值图
            spacings: (B, 3) Tensor/Array, 每个样本的物理间距 (sx, sy, sz)
        """
        if preds.dim() == 5: preds = preds.squeeze(1)
        if targets.dim() == 5: targets = targets.squeeze(1)
        
        preds_np = preds.detach().cpu().numpy().astype(np.uint8)
        targets_np = targets.detach().cpu().numpy().astype(np.uint8)
        spacings_np = spacings.detach().cpu().numpy()
        
        batch_size = preds_np.shape[0]
        # 注意：这里的键名需与 validate 函数中的调用保持一致
        batch_metrics = {"dice": [], "hd95": [], "avd": [], "recall": [], "lesion_f1": []}

        for i in range(batch_size):
            # 1. 维度转换：PyTorch (W,H,D) -> SimpleITK (D,H,W) 即 (Z,Y,X)
            p_arr = preds_np[i].transpose(2, 1, 0) 
            t_arr = targets_np[i].transpose(2, 1, 0)
            
            pred_img = sitk.GetImageFromArray(p_arr)
            target_img = sitk.GetImageFromArray(t_arr)
            
            # 2. 注入物理间距 (关键：确保 HD95 单位为 mm)
            current_spacing = tuple(float(x) for x in spacings_np[i])
            pred_img.SetSpacing(current_spacing)
            target_img.SetSpacing(current_spacing)
            
            # 3. 计算各个子指标
            batch_metrics["dice"].append(self._get_dsc(target_img, pred_img))
            batch_metrics["hd95"].append(self._get_hausdorff(target_img, pred_img))
            batch_metrics["avd"].append(self._get_avd(target_img, pred_img))
            
            rec, f1 = self._get_lesion_detection(target_img, pred_img)
            batch_metrics["recall"].append(rec)
            batch_metrics["lesion_f1"].append(f1)

        # 返回 Batch 平均值 (忽略 NaN)
        return {k: np.nanmean(v) for k, v in batch_metrics.items()}

    # ========================== 以下为官方算法复刻 ==========================

    def _get_dsc(self, testImage, resultImage):
        """官方 Dice 逻辑"""
        testArray = sitk.GetArrayFromImage(testImage).flatten()
        resultArray = sitk.GetArrayFromImage(resultImage).flatten()
        if np.sum(testArray) == 0 and np.sum(resultArray) == 0: return 1.0
        if np.sum(testArray) == 0 or np.sum(resultArray) == 0: return 0.0
        return 1.0 - scipy.spatial.distance.dice(testArray, resultArray)

    def _get_hausdorff(self, testImage, resultImage):
        """官方 95% 豪斯多夫距离 (mm)"""
        # 如果预测为空，HD 无意义
        res_stat = sitk.StatisticsImageFilter()
        res_stat.Execute(resultImage)
        if res_stat.GetSum() == 0: return float('nan')
        
        # 2D 腐蚀提取边缘 (针对每个切片)
        eTestImage = sitk.BinaryErode(testImage, (1, 1, 0))
        eResultImage = sitk.BinaryErode(resultImage, (1, 1, 0))
        
        hTestArray = sitk.GetArrayFromImage(sitk.Subtract(testImage, eTestImage))
        hResultArray = sitk.GetArrayFromImage(sitk.Subtract(resultImage, eResultImage))
        
        # 获取物理坐标点集
        test_idx = np.transpose(np.nonzero(hTestArray))
        res_idx = np.transpose(np.nonzero(hResultArray))
        
        # 坐标翻转 (z,y,x) -> (x,y,z) 并转为物理坐标
        test_coords = [testImage.TransformIndexToPhysicalPoint(idx[::-1].tolist()) for idx in test_idx]
        res_coords = [testImage.TransformIndexToPhysicalPoint(idx[::-1].tolist()) for idx in res_idx]

        if len(test_coords) == 0 or len(res_coords) == 0: return float('nan')

        # KDTree 计算 95th Percentile 距离
        def get_dist(a, b):
            tree = scipy.spatial.KDTree(a, leafsize=100)
            return tree.query(b, k=1, eps=0, p=2)[0]

        d_t_to_r = get_dist(test_coords, res_coords)
        d_r_to_t = get_dist(res_coords, test_coords)
        return max(np.percentile(d_t_to_r, 95), np.percentile(d_r_to_t, 95))

    def _get_avd(self, testImage, resultImage):
        """官方 AVD 逻辑"""
        t_stat = sitk.StatisticsImageFilter()
        r_stat = sitk.StatisticsImageFilter()
        t_stat.Execute(testImage)
        r_stat.Execute(resultImage)
        t_sum = t_stat.GetSum()
        r_sum = r_stat.GetSum()
        if t_sum == 0: return 0.0 if r_sum == 0 else 100.0
        return (abs(t_sum - r_sum) / float(t_sum)) * 100.0

    def _get_lesion_detection(self, testImage, resultImage):
        """官方病灶级 Recall 和 F1 逻辑"""
        ccFilter = sitk.ConnectedComponentImageFilter()
        ccFilter.SetFullyConnected(True)
        
        # 计算 Recall: GT病灶有多少被覆盖
        ccTest = ccFilter.Execute(testImage)
        lResult = sitk.Multiply(ccTest, sitk.Cast(resultImage, sitk.sitkUInt32))
        nWMH = len(np.unique(sitk.GetArrayFromImage(ccTest))) - 1
        recall = 1.0 if nWMH == 0 else (len(np.unique(sitk.GetArrayFromImage(lResult))) - 1) / float(nWMH)

        # 计算 Precision: 预测病灶有多少是有效的
        ccResult = ccFilter.Execute(resultImage)
        lTest = sitk.Multiply(ccResult, sitk.Cast(testImage, sitk.sitkUInt32))
        nDetections = len(np.unique(sitk.GetArrayFromImage(ccResult))) - 1
        precision = 1.0 if nDetections == 0 else (len(np.unique(sitk.GetArrayFromImage(lTest))) - 1) / float(nDetections)

        f1 = 0.0 if (precision + recall) == 0 else (2.0 * precision * recall) / (precision + recall)
        return recall, f1

# ==============================================================================
# 5. 简单的 3D U-Net 定义 (用于演示，可替换)
# ==============================================================================
class SimpleUNet3D(nn.Module):
    def __init__(self, in_channels=2, out_classes=3):
        super(SimpleUNet3D, self).__init__()
        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv3d(in_c, out_c, kernel_size=3, padding=1),
                nn.BatchNorm3d(out_c),
                nn.ReLU(inplace=True)
            )
        self.enc1 = conv_block(in_channels, 32)
        self.pool = nn.MaxPool3d(2)
        self.enc2 = conv_block(32, 64)
        self.enc3 = conv_block(64, 128)
        
        self.up2 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.dec2 = conv_block(128, 64) # 64+64
        self.up1 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2)
        self.dec1 = conv_block(64, 32)  # 32+32
        
        self.final = nn.Conv3d(32, out_classes, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        
        d2 = self.dec2(torch.cat([self.up2(e3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.final(d1)

# ==============================================================================
# 6. 训练与验证 Loop
# ==============================================================================

def train_epoch(model, loader, optimizer, criterion, device, epoch):
    model.train()
    running_loss = 0.0
    pbar = tqdm(loader, desc=f"Epoch {epoch} [Train]", unit="batch")
    
    for batch in pbar:
        # 输入拼接: (B, 2, D, H, W)
        flair = batch['flair'].to(device).float()
        t1 = batch['t1'].to(device).float()
        
        inputs = torch.cat([flair, t1], dim=1)
        
        targets = batch['seg'].to(device)
        
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs) # Output: (B, 3, D, H, W)
        
        # Loss: 屏蔽 Label 2 (Other) -> Weight=0.0
        loss = criterion(logits, targets, class_weights=[0.1, 1.0, 0.0])
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        running_loss += loss.item()
        pbar.set_postfix({'Loss': f"{loss.item():.4f}"})
        
    return running_loss / len(loader)



def validate_plot(model, loader, criterion, device, logger=None):
    """
    执行验证：
    1. 计算 Metrics (Dice, HD95...)，并按 Center 分组输出。
    2. 计算 Validation Loss。
    """
    model.eval()
    calc = MetricCalculator()
    
    # 数据结构: results["Center_Name"]["Metric_Name"] = [value1, value2...]
    results = defaultdict(lambda: defaultdict(list))
    
    # 用于计算平均 Val Loss
    total_val_loss = 0.0
    num_loss_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="[Validating]"):
            flair = batch['flair'].to(device).float()
            t1 = batch['t1'].to(device).float()
            inputs = torch.cat([flair, t1], dim=1)
            targets = batch['seg'].to(device)
            spacings = batch['spacing']
            
            # 获取中心名称 (如果没有则默认为 Global)
            center_name = batch['center'][0] if 'center' in batch else 'Unknown'
            
            # 1. 推理 (无 TTA)
            logits = sliding_window_inference(
                inputs, (128, 128, 48), 4, model, overlap=0.5, mode='gaussian'
            )
            

            loss = criterion(logits, targets,class_weights=[0.1, 1.0, 0.0])
            total_val_loss += loss.item()
            num_loss_batches += 1
        

            # 3. 计算指标
            preds = torch.argmax(logits, dim=1, keepdim=True)
            pred_wmh = (preds == 1).long() # 只看 Label 1
            
            metrics = calc.calculate(pred_wmh, targets == 1, spacings)
            
            # 4. 存入字典 (同时存入 Global 和 Specific Center)
            for k, v in metrics.items():
                if not np.isnan(v):
                    results['Global'][k].append(v)
                    results[center_name][k].append(v)

    # --- 汇总统计与打印 ---
    
    avg_val_loss = total_val_loss / num_loss_batches if num_loss_batches > 0 else 0.0
    final_metrics = {} # 返回给 Scheduler 的 Global 指标
    
    if logger:
        logger.info("-" * 95)
        logger.info(f"Val Loss: {avg_val_loss:.4f}")
        logger.info("-" * 95)
        logger.info(f"{'Center':<25} | {'Dice':<8} | {'HD95':<8} | {'Recall':<8} | {'F1':<8} | {'AVD':<8} | {'Count':<5}")
        logger.info("-" * 95)

    # 排序：Global 第一，其他按字母排
    centers = ['Global'] + sorted([k for k in results.keys() if k != 'Global'])
    
    for c in centers:
        # 计算该中心各项指标均值
        c_metrics = {k: np.mean(v) for k, v in results[c].items()}
        count = len(results[c]['dice'])
        
        # 记录 Global 指标用于返回值
        if c == 'Global':
            final_metrics = c_metrics
            
        if logger:
            logger.info(f"{c:<25} | {c_metrics['dice']:.4f}   | {c_metrics['hd95']:.4f}   | "
                        f"{c_metrics['recall']:.4f}   | {c_metrics['lesion_f1']:.4f}   | "
                        f"{c_metrics['avd']:.4f}   | {count:<5}")

    if logger: logger.info("-" * 95)
    
    # 返回 metrics 字典 和 loss 值
    return final_metrics, avg_val_loss

def validate(model, loader, criterion, device, full_metrics=False, logger=None):
    """
    对齐官方标准的验证函数 (含分中心输出支持)
    1. 屏蔽 Label 2 (non-WMH) 区域的预测干扰
    2. full_metrics=True 时计算 HD95, AVD (耗时)
    3. full_metrics=False 时仅计算 Dice (快速)
    """
    model.eval()
    calc = MetricCalculator()
    
    # 存储结构: results["Center"]["Metric"] = [val1, val2...]
    results = defaultdict(lambda: defaultdict(list))
    
    total_val_loss = 0.0
    num_loss_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="[Validating]"):
            flair = batch['flair'].to(device).float()
            t1 = batch['t1'].to(device).float()
            inputs = torch.cat([flair, t1], dim=1)
            targets = batch['seg'].to(device) # 0, 1, 2
            spacings = batch['spacing']
            
            # 获取中心名称，用于分组统计
            center_name = batch['center'][0] if 'center' in batch else 'Unknown'
            
            # 推理
            logits = sliding_window_inference(
                inputs, (128, 128, 48), 4, model, overlap=0.5, mode='gaussian'
            )
            
            # 计算 Loss (增加容错，防止 OOM 中断)
            try:
                loss = criterion(logits, targets, class_weights=[0.1, 1.0, 0.0])
                total_val_loss += loss.item()
                num_loss_batches += 1
            except RuntimeError:
                pass # 忽略显存不足导致的 Loss 计算失败
            
            # 后处理
            preds = torch.argmax(logits, dim=1, keepdim=True)
            
            # --- 对齐官方屏蔽逻辑 ---
            pred_wmh = (preds == 1).long()
            # 在 Label 2 区域，无论预测什么都抹除，不计入误报
            pred_wmh[targets == 2] = 0 
            target_wmh = (targets == 1).long()
            
            # 指标计算
            if full_metrics:
                metrics = calc.calculate(pred_wmh, target_wmh, spacings)
            else:
                # 快速计算像素级 Dice 用于监控
                # 假设 MetricCalculator 有 _get_dsc_simple，如果没有则用 calculate
                if hasattr(calc, '_get_dsc_simple'):
                    d_val = calc._get_dsc_simple(pred_wmh, target_wmh)
                    metrics = {'dice': d_val}
                else:
                    metrics = calc.calculate(pred_wmh, target_wmh, spacings)
            
            # 存入字典 (同时存入 Global 和 Specific Center)
            for k, v in metrics.items():
                if not np.isnan(v):
                    results['Global'][k].append(v)
                    results[center_name][k].append(v)

    # --- 汇总统计与日志输出 ---
    avg_val_loss = total_val_loss / num_loss_batches if num_loss_batches > 0 else 0.0
    final_metrics_global = {k: np.mean(v) for k, v in results['Global'].items()}
    
    if logger:
        logger.info("-" * 95)
        logger.info(f"Val Loss: {avg_val_loss:.4f} | Mode: {'FULL' if full_metrics else 'FAST'}")
        logger.info("-" * 95)
        
        # 动态调整表头
        if full_metrics:
            headers = f"{'Center':<25} | {'Dice':<8} | {'HD95':<8} | {'Recall':<8} | {'F1':<8} | {'AVD':<8} | {'Count':<5}"
        else:
            headers = f"{'Center':<25} | {'Dice':<8} | {'Count':<5}"
        logger.info(headers)
        logger.info("-" * 95)

        # 排序：Global 第一，其他按字母排
        centers = ['Global'] + sorted([k for k in results.keys() if k != 'Global'])
        
        for c in centers:
            c_metrics = {k: np.mean(v) for k, v in results[c].items()}
            count = len(results[c]['dice']) # 样本数量
            
            if full_metrics:
                logger.info(f"{c:<25} | {c_metrics.get('dice',0):.4f}   | {c_metrics.get('hd95',0):.4f}   | "
                            f"{c_metrics.get('recall',0):.4f}   | {c_metrics.get('lesion_f1',0):.4f}   | "
                            f"{c_metrics.get('avd',0):.4f}   | {count:<5}")
            else:
                logger.info(f"{c:<25} | {c_metrics.get('dice',0):.4f}   | {count:<5}")

        logger.info("-" * 95)
    
    return final_metrics_global, avg_val_loss



def run_training(model, train_loader, val_loader, device, save_dir, num_epochs=200):
    """
    训练主循环：回归 ReduceLROnPlateau 策略
    """
    # ---------------------------------------------------
    # 1. 内部初始化组件
    # ---------------------------------------------------
    # 使用之前优化过的 Tversky + Focal Loss (提升 Recall)
    criterion = CombinedLoss(n_classes=3).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    
    # === 修改点 A: 使用基于指标的调度器 ===
    # mode='max': 监控指标越大越好 (Dice)
    # factor=0.5: 触发时 LR 减半
    # patience=15: 指标 15 次验证没有提升才触发
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=15, verbose=True
    )
    
    # 日志配置
    log_path = os.path.join(save_dir, "train_log.txt")
    os.makedirs(save_dir, exist_ok=True)
    logger = setup_logging(log_path)
    
    logger.info(f"Start Training | Device: {device} | Epochs: {num_epochs}")
    logger.info("Config: Scheduler=ReduceLROnPlateau (Patience=15), Val Start=200")

    best_dice = 0.0
    full_eval_interval = 12 

    # ---------------------------------------------------
    # 2. 训练循环
    # ---------------------------------------------------
    for epoch in range(1, num_epochs + 1):
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")
        
        # === 训练步骤 ===
        for batch in pbar:
            flair = batch['flair'].to(device).float()
            t1 = batch['t1'].to(device).float()
            inputs = torch.cat([flair, t1], dim=1)
            targets = batch['seg'].to(device)
            
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            
            # Loss 计算 (注意：Label 2 权重设为 0.5 辅助训练)
            loss = criterion(logits, targets, class_weights=[0.1, 1.0, 0.0])
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        avg_train_loss = train_loss / len(train_loader)

        # === 验证逻辑 ===
        # 策略：前 200 轮不验证（专心拟合），200 轮后每 4 轮验证一次
        should_val = (epoch >= 100) and (epoch % 4 == 0)
        
        if should_val:
            # 判断是否计算全量指标 (HD95 等)
            is_full = (epoch % full_eval_interval == 0) or (epoch == num_epochs)
            
            # 执行验证
            val_metrics, val_loss = validate(
                model, val_loader, criterion, device, 
                full_metrics=is_full, logger=logger
            )
            
            cur_dice = val_metrics.get('dice', 0.0)
            
            # 获取当前 LR 用于打印
            current_lr = optimizer.param_groups[0]['lr']
            logger.info(f"[Epoch {epoch}] Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | Dice: {cur_dice:.4f} | LR: {current_lr:.6f}")
            
            # === 修改点 B: 调度器更新 ===
            # ReduceLROnPlateau 必须在拿到 Dice 后更新
            # 注意：这意味着前 200 轮 LR 恒定为 1e-3
            scheduler.step(cur_dice)
            
            # --- 保存最佳模型 ---
            if cur_dice > best_dice:
                best_dice = cur_dice
                logger.info(f"🌟 New Record! Best Dice: {best_dice:.4f}")
                
                # 如果破纪录时是快速模式，补测一次全量指标用于记录
                if not is_full:
                    full_val_metrics, _ = validate(model, val_loader, criterion, device, full_metrics=True)
                else:
                    full_val_metrics = val_metrics
                
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'metrics': full_val_metrics,
                    'best_dice': best_dice
                }, os.path.join(save_dir, "best_model.pth"))
                logger.info(f"Saved Best Model. Full Metrics: {full_val_metrics}")
        
        else:
            # 不验证时，只记录 Train Loss，LR 保持不变
            current_lr = optimizer.param_groups[0]['lr']
            logger.info(f"[Epoch {epoch}] Train Loss: {avg_train_loss:.4f} | (Val Skipped) | LR: {current_lr:.6f}")

    # ---------------------------------------------------
    # 3. 结束
    # ---------------------------------------------------
    torch.save(model.state_dict(), os.path.join(save_dir, "final_model.pth"))
    logger.info(f"Training Finished! Best Dice: {best_dice:.4f}")

def run_training_plot(model, train_loader, test_loader, device, save_dir, num_epochs=300):
    # ---------------------------------------------------
    # 1. 初始化设置
    # ---------------------------------------------------
    model = model.to(device)
    
    # 定义 Loss: 之前确定的 Dice + Focal，权重屏蔽 Label 2
    # 注意：确保 CombinedLoss 类内部默认或此处传入的权重是 [0.1, 1.0, 0.0]
    criterion = CombinedLoss(n_classes=3) 
    
    # 优化器
    optimizer = optim.AdamW(model.parameters(), lr=5e-3, weight_decay=1e-5)
    
    # 学习率调度器: 监控 Dice (max 模式)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=10, verbose=True
    )
    
    # 验证间隔 (每 2 个 Epoch 验一次)
    VAL_INTERVAL = 2
    
    # 历史记录列表
    train_loss_history = []
    val_loss_history = []
    best_dice = 0.0
    
    logging.info(f"Start training for {num_epochs} epochs. Val interval: {VAL_INTERVAL}")

    # ---------------------------------------------------
    # 2. 训练循环
    # ---------------------------------------------------
    for epoch in range(1, num_epochs + 1):
        
        # === 训练阶段 (每个 Epoch 都执行) ===
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, epoch)
        train_loss_history.append(train_loss)
        
        # === 验证阶段 (仅在特定 Epoch 执行) ===
        if epoch % VAL_INTERVAL == 0:
            
            # 执行验证 (返回 metrics 字典 和 平均 loss)
            # 注意: validate 函数需要接受 criterion 参数
            val_metrics, val_loss = validate_plot(model, test_loader, criterion, device, logging)
            
            # 记录验证 Loss
            val_loss_history.append(val_loss)
            
            # 获取当前 Dice (Global)
            current_dice = val_metrics.get('dice', 0.0)
            
            # --- A. 绘制 Loss 曲线 ---
            plot_loss_curve(
                train_loss_history, 
                val_loss_history, 
                VAL_INTERVAL, 
                os.path.join(save_dir, "loss_curve.png")
            )
            
            # --- B. 保存最佳模型 (特殊标记) ---
            if current_dice > best_dice:
                best_dice = current_dice
                torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pth"))
                # 打印特殊标记日志
                logging.info(f"🌟 [New Record] Saved best model at epoch {epoch} (Dice: {best_dice:.4f})")
            
            # --- C. 学习率调度 ---
            # ReduceLROnPlateau 需要传入当前的指标
            scheduler.step(current_dice)
                
            # 打印本轮汇总
            logging.info(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Dice: {current_dice:.4f}")
            
        else:
            # 跳过验证的轮次，只打印训练 Loss
            logging.info(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | (Validation Skipped)")
    
    # ---------------------------------------------------
    # 3. 训练结束
    # ---------------------------------------------------
    
    # 保存最后一个 Epoch 的模型
    torch.save(model.state_dict(), os.path.join(save_dir, "final_model.pth"))
    logging.info(f"Training Finished! Best Dice: {best_dice:.4f}")


# ==============================================================================
# 7. 主程序入口
# ==============================================================================
if __name__ == '__main__':
    # 配置区f
    ROOT_PATH = '/data4T/WHM'       # 修改为你的数据路径
    SAVE_DIR = './logs_experiment_UNET'
    BATCH_SIZE = 4                  # 根据显存调整
    NUM_WORKERS = 4
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR, exist_ok=True)
        print(f"Created directory: {SAVE_DIR}")
    # 1. Transforms (Patch: 128x128x48)
    # train_tf = Compose([
    #     IntensityNormalize(),
    #     RandomRotFlip(),
    #     BrainRegionRandomCrop(output_size=(128, 128, 48)) 
    # ])

    train_tf = Compose([
    # 1. 第一步：先切小块，极大提升速度
    BrainRegionRandomCrop(output_size=(128, 128, 48)),
    

    # 2. 第二步：空间几何增强 (只动位置)
    RandomRotFlip(),
    RandomGaussianNoise(std_range=(0, 0.05)),
    RandomGamma(gamma_range=(0.8, 1.2)),
    #RandomElasticDeformation(), # 如果算力够强可以加上这个

    # 4. 第四步：最后兜底，统一分布
    RobustIntensityNormalize(),
])

    # train_tf = Compose([
    #     # 1. 裁剪 (保持不变)
    #     BrainRegionRandomCrop(output_size=(128, 128, 48)),
        
    #     # 2. 空间增强 (保持不变)
    #     RandomRotFlip(),
        
    #     # 3. 归一化 (核心修改！！)
    #     # 先归一化到 [0, 1]，这样后面的 Gamma 变换才准确
    #     AdvancedModalityNormalize(), 
        
    #     # 4. 模态特异性增强 (核心修改！！)
    #     ModalitySpecificAugmentation(prob=0.3),
        
    #     # 5. 弹性形变 (可选)
    #     #
    #     ])
    val_tf = Compose([
        RobustIntensityNormalize(),

    ])
    
    # 2. Datasets & Loaders
    def worker_init(worker_id): random.seed(1337 + worker_id)
    logger = setup_logging(os.path.join(SAVE_DIR, "train_log.txt"))
    
    logger.info(f"Using device: {DEVICE}")
    logger.info(f"Saving results to: {SAVE_DIR}")

    train_ds = WMHDataset(ROOT_PATH, split='training', transform=train_tf)
    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, 
                            num_workers=4, pin_memory=True, worker_init_fn=worker_init)

    # 验证 Loader (【关键】Batch Size 必须为 1)
    # 因为不同病人的全图尺寸是不一样的 (有的Z=48，有的Z=83)，不能堆叠成 Batch
    test_ds = WMHDataset(ROOT_PATH, split='test', transform=val_tf)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, 
                            num_workers=1, pin_memory=True)
    
    # 3. Model
    # 输入通道=2 (FLAIR+T1), 输出通道=3 (Bg, WMH, Other)
    #model = SimpleUNet3D(in_channels=2, out_classes=3).to(DEVICE)
    #model=Optimized_DynamicModal_Net(n_classes=3, base_c=32, num_modalities=2).to(DEVICE)
    # model = SwinUNETR(
    #             spatial_dims=3,      # 3D
    #             in_channels=2,
    #             out_channels=3,
    #             feature_size=48,
    #         ).cuda()
    model = Ablation_NEncoder_Final_Net(
        n_classes=3, 
        num_modalities=2, 
        base_c=16, 
        deep_sup=False
    ).to(DEVICE)
    #model = UNet3D(in_channels=2,n_classes=3, base_c=32).cuda()
    # 4. Start
    run_training(model, train_loader, test_loader, DEVICE, SAVE_DIR, num_epochs=200)