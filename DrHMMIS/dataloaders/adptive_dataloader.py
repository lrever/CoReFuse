import os
import numpy as np
import h5py
import torch
from torch.utils.data import Dataset
from collections import defaultdict
import torch.nn.functional as F
# -------------------- 数据增强类 -------------------- #
class RandomRegionCrop:
    """基于 mask 区域权重随机裁剪 patch + 动态权重更新"""
    def __init__(self, patch_size=(64,64,64), temperature=0.05, momentum=0.9):
        self.patch_size = np.array(patch_size)
        self.temperature = temperature
        self.momentum = momentum

        # 动态权重统计
        self.region_weights = dict()
        self.region_loss_tracker = dict()
        self.region_total_loss = defaultdict(float)
        self.region_total_voxels = defaultdict(int)
        self.all_region_ids = set()

    def __call__(self, sample):
        mask = sample['mask']

        # 初始化均匀权重
        if not self.region_weights:
            unique_regions = np.unique(mask)
            self.region_weights = {r: 1/len(unique_regions) for r in unique_regions}

        # 根据权重选择区域
        regions = list(self.region_weights.keys())
        weights = np.array([self.region_weights[r] for r in regions])
        weights /= weights.sum()
        chosen_region = np.random.choice(regions, p=weights)
        # 找到该区域所有 voxel
        idxs = np.argwhere(mask==chosen_region)
        center_idx = idxs[np.random.randint(len(idxs))]

        # 计算 patch 范围
        w,h,d = mask.shape
        start = np.maximum(center_idx - self.patch_size//2, 0)
        end = np.minimum(start + self.patch_size, [w,h,d])
        start = end - self.patch_size  # 修正边界

        patch = {k: sample[k][start[0]:end[0], start[1]:end[1], start[2]:end[2]] for k in sample}
        patch['chosen_region'] = chosen_region
        return patch

    # ----------------- loss 累积 ----------------- #
    def accumulate_patch_loss(self, patch_loss_map, patch_mask):
        """
        patch_loss_map: [B,D,H,W] 或 [B,1,D,H,W] 或 [B,2,D,H,W]
        patch_mask: [B,D,H,W]
        """
        import numpy as np

        # 转为 numpy
        if isinstance(patch_loss_map, torch.Tensor):
            patch_loss_map = patch_loss_map.detach().cpu().numpy()
        if isinstance(patch_mask, torch.Tensor):
            patch_mask = patch_mask.detach().cpu().numpy()

        # 增加 batch 维度（如果缺少）
        if patch_loss_map.ndim == 3:  # [D,H,W]
            patch_loss_map = patch_loss_map[np.newaxis, ...]
            patch_mask = patch_mask[np.newaxis, ...]

        B = patch_loss_map.shape[0]

        for b in range(B):
            loss_b = patch_loss_map[b]
            mask_b = patch_mask[b]

            # ---------------- 处理 channel ---------------- #
            # [1,D,H,W] -> [D,H,W]
            if loss_b.ndim == 4 and loss_b.shape[0] == 1:
                loss_b = loss_b.squeeze(0)
            # [2,D,H,W] -> 取前景 channel=1
            elif loss_b.ndim == 4 and loss_b.shape[0] == 2:
                loss_b = loss_b[1]

            # ---------------- 累积 ---------------- #
            for r in np.unique(mask_b):
                mask_r = (mask_b == r)
                if np.any(mask_r):
                    self.region_total_loss[r] += loss_b[mask_r].sum()
                    self.region_total_voxels[r] += mask_r.sum()
                    self.all_region_ids.add(r)

    # ----------------- 权重更新 ----------------- #
    # def update_region_weights(self):
    #     if not self.all_region_ids.issubset(self.region_total_voxels.keys()):
    #         return False

    #     # 平均 loss
    #     region_avg_loss = {r: self.region_total_loss[r] / max(self.region_total_voxels[r], 1)
    #                        for r in self.all_region_ids}

    #     # 指数移动平均
    #     for r, avg in region_avg_loss.items():
    #         old_val = self.region_loss_tracker.get(r, 1.0)
    #         self.region_loss_tracker[r] = self.momentum*old_val + (1-self.momentum)*avg

    #     # -----------------------------------------------------
    #     # !! 在这里添加你想要的日志输出 !!
    #     # -----------------------------------------------------
    #     print("\n" + "="*30)
    #     print(f"区域权重更新 (Temperature: {self.temperature})")
    #     print("="*30)
        
    #     sorted_ids = sorted(self.all_region_ids) # 确保顺序一致
        
    #     print("ID | 本 Epoch 平均 Loss | EMA 跟踪 Loss")
    #     print("---|--------------------|----------------")
    #     for r in sorted_ids:
    #         print(f"{r:<2d} | {region_avg_loss.get(r, 0):<18.6f} | {self.region_loss_tracker.get(r, 0):<15.6f}")

    #     # softmax 归一化得到新权重
    #     losses = np.array([self.region_loss_tracker[r] for r in sorted(self.all_region_ids)])
    #     exp_losses = np.exp(losses / self.temperature)
    #     softmax_weights = exp_losses / exp_losses.sum()
    #     self.region_weights = {r: w for r, w in zip(sorted(self.all_region_ids), softmax_weights)}

    #     return True
    def update_region_weights(self):
        """
        在 epoch 结束后调用，计算并更新所有区域的采样权重。
        使用归一化（Normalization）代替 Softmax。
        """
        if not self.all_region_ids:
            print("警告：没有收集到任何区域的损失信息，跳过权重更新。")
            return False

        # 1. 计算本 Epoch 每个区域的平均 loss
        region_avg_loss = {}
        total_avg_loss = 0.0
        sorted_ids = sorted(list(self.all_region_ids))

        for r in sorted_ids:
            if r in self.region_total_voxels and self.region_total_voxels[r] > 0:
                avg_loss = self.region_total_loss[r] / self.region_total_voxels[r]
            else:
                # 如果某个区域没有被采样到（例如区域0），给一个极小的损失值
                # 这样它仍然有很小的概率被选中（避免除以零）
                avg_loss = 1e-9  
            
            region_avg_loss[r] = avg_loss
            total_avg_loss += avg_loss

        # 2. (可选) 使用 EMA 平滑损失值
        # 这可以防止权重在不同 epoch 之间剧烈波动
        for r in sorted_ids:
            if r not in self.region_loss_tracker:
                self.region_loss_tracker[r] = region_avg_loss[r]  # 初始化
            else:
                old_val = self.region_loss_tracker[r]
                self.region_loss_tracker[r] = (self.momentum * old_val) + (1.0 - self.momentum) * region_avg_loss[r]

        # 3. 计算归一化权重（使用平滑后的损失）
        # 我们使用 self.region_loss_tracker 中的值进行归一化
        smoothed_losses = np.array([self.region_loss_tracker.get(r, 1e-9) for r in sorted_ids])
        
        # 确保所有损失都是非负的（Focal Loss 应该是，但以防万一）
        smoothed_losses = np.maximum(smoothed_losses, 0)
        
        total_smoothed_loss = np.sum(smoothed_losses)

        print("\n" + "="*30)
        print(f"区域权重更新 (使用归一化, Momentum: {self.momentum})")
        print("="*30)
        print(f"{'ID':<3} | {'Epoch Avg Loss':<18} | {'EMA-Smoothed Loss':<20}")
        print("----|--------------------|---------------------")
        
        final_weights = {}
        if total_smoothed_loss > 0:
            for i, r in enumerate(sorted_ids):
                weight = smoothed_losses[i] / total_smoothed_loss
                final_weights[r] = weight
                
                # 打印日志
                print(f"{r:<3} | {region_avg_loss.get(r, 0):<18.6f} | {self.region_loss_tracker.get(r, 0):<20.6f}")
        else:
            # 如果总损失为0（例如所有都是背景），则均匀分配
            print("所有区域损失均为0，使用均匀权重。")
            num_regions = len(sorted_ids)
            for r in sorted_ids:
                final_weights[r] = 1.0 / num_regions

        # 4. 更新权重
        self.region_weights = final_weights

        print("\n--- 更新后的采样权重 ---")
        weights_log = [f"ID {r}: {self.region_weights.get(r, 0):.4f}" for r in sorted_ids]
        print(", ".join(weights_log))
        print("="*30 + "\n")

        # 5. 重置当前 epoch 的累加器
        self.region_total_loss.clear()
        self.region_total_voxels.clear()
        self.all_region_ids.clear()

        return True
    

class AdaptiveRegionSampler:
    """
    根据模型预测的不确定性（熵）自适应地对图像区域进行采样。
    
    工作流程：
    1.  训练时，在每个 batch 之后，调用 `self.accumulate_uncertainty()`
    2.  在每个 epoch 结束时，调用 `self.update_sampling_weights()`
    3.  `DataLoader` 在 `__getitem__` 中调用 `self()` 来获取一个 patch。
    """
    
    def __init__(self, patch_size, num_classes, temperature=1.0, momentum=0.9):
        self.patch_size = np.array(patch_size)
        self.num_classes = num_classes
        self.temperature = temperature # 控制采样分布的 "尖锐度"
        self.momentum = momentum     # 用于平滑（EMA）不确定性得分
        
        # 存储每个区域的采样权重 (概率)
        self.region_weights = {} 
        
        # 跟踪每个区域的累积不确定性（熵）和体素数量
        self.region_total_uncertainty = defaultdict(float)
        self.region_total_voxels = defaultdict(int)
        
        # 跟踪每个区域的平滑后的不确定性 (EMA)
        self.region_uncertainty_tracker = {}
        
        self.all_region_ids = set() # 跟踪此 epoch 中遇到的所有区域ID

    def __call__(self, sample):
        """
        从完整的图像/标签/掩码 'sample' 中采样一个 patch。
        """
        mask = sample['mask']
        image_shape = mask.shape

        # 1. 如果权重字典为空，初始化为均匀分布
        if not self.region_weights:
            unique_regions = np.unique(mask)
            # 过滤掉背景（假设背景标签为0，如果它存在）
            unique_regions = [r for r in unique_regions if r != 0]
            if not unique_regions: # 如果只有背景
                unique_regions = [0]
            
            num_regions = len(unique_regions)
            self.region_weights = {r: 1.0 / num_regions for r in unique_regions}
            if 0 not in self.region_weights:
                self.region_weights[0] = 0.0 # 默认不从背景采样，除非它是唯一选项
            print(f"Initialized region weights: {self.region_weights}")

        # 2. 根据权重选择一个区域
        regions = list(self.region_weights.keys())
        weights = np.array([self.region_weights[r] for r in regions])
        
        # 确保权重总和为 1
        weights_sum = weights.sum()
        if weights_sum == 0:  # 避免除以零，如果所有权重都是0
            weights = np.ones_like(weights) / len(weights)
        else:
            weights /= weights_sum
            
        chosen_region = np.random.choice(regions, p=weights)

        # 3. 在所选区域内随机选择一个中心点
        # 如果选择背景（0），则从整个图像中随机采样
        if chosen_region == 0:
             indices = np.argwhere(np.ones(image_shape))
        else:
            indices = np.argwhere(mask == chosen_region)
            
        if len(indices) == 0: # 如果选择的区域为空（不应该发生，但作为保险）
            print(f"Warning: Region {chosen_region} has no voxels. Defaulting to center crop.")
            center_idx = np.array(image_shape) // 2
        else:
            center_idx = indices[np.random.randint(0, len(indices))]

        # 4. 计算 patch 的边界 (处理边缘情况)
        start = np.maximum(center_idx - self.patch_size // 2, 0)
        end = start + self.patch_size
        
        # 检查是否超出边界并调整
        shift = np.minimum(image_shape - end, 0)
        start += shift
        end += shift

        # 5. 提取 patch
        s_x, s_y, s_z = start
        e_x, e_y, e_z = end
        
        patch = {}
        for key, val in sample.items():
            if val is not None and hasattr(val, 'shape') and len(val.shape) >= 3:
                patch[key] = val[s_x:e_x, s_y:e_y, s_z:e_z]
            else:
                patch[key] = val # 复制元数据，如 'chosen_region' (虽然在这里设置，但以防万一)

        patch['chosen_region'] = chosen_region # 存储选择的区域，以便后续使用
        return patch

    def accumulate_uncertainty(self, logits, mask):
        """
        计算并累积每个区域的平均不确定性（熵）。
        
        Args:
            logits (torch.Tensor): 模型的原始输出 (B, C, D, H, W)
            mask (torch.Tensor): 区域掩码 (B, D, H, W)
        """
        # 1. 计算 Softmax 概率
        # 使用 log_softmax 和 softmax 组合来提高数值稳定性
        log_probs = F.log_softmax(logits, dim=1)  # (B, C, D, H, W)
        probs = torch.exp(log_probs)             # (B, C, D, H, W)

        # 2. 计算熵图: H = -sum(p * log_p)
        # (B, C, D, H, W) * (B, C, D, H, W) -> (B, C, D, H, W)
        entropy_map = -torch.sum(probs * log_probs, dim=1) # (B, D, H, W)
        
        # 3. 移至 CPU 和 NumPy 以便聚合
        entropy_map_np = entropy_map.detach().cpu().numpy()
        mask_np = mask.cpu().numpy()

        B = entropy_map_np.shape[0]
        for b in range(B):
            b_entropy = entropy_map_np[b]
            b_mask = mask_np[b]
            
            unique_regions_in_batch = np.unique(b_mask)
            for r in unique_regions_in_batch:
                # 只关心我们定义的区域，特别是排除背景（通常为0）
                if r == 0: # 假设 0 是背景
                    continue
                
                self.all_region_ids.add(r)
                region_mask = (b_mask == r)
                
                # 累加这个区域的总熵和体素数
                self.region_total_uncertainty[r] += b_entropy[region_mask].sum()
                self.region_total_voxels[r] += region_mask.sum()

    def update_sampling_weights(self):
        """
        在 epoch 结束时调用，计算并更新所有区域的采样权重。
        """
        if not self.all_region_ids:
            print("警告：未收集到任何区域的不确定性信息，跳过权重更新。")
            return False

        # 1. 计算本 Epoch 每个区域的平均不确定性（熵）
        region_avg_uncertainty = {}
        for r in self.all_region_ids:
            if r in self.region_total_voxels and self.region_total_voxels[r] > 0:
                avg_uncertainty = self.region_total_uncertainty[r] / self.region_total_voxels[r]
            else:
                avg_uncertainty = 0.0
            
            region_avg_uncertainty[r] = avg_uncertainty

        # 2. 使用指数移动平均 (EMA) 平滑不确定性得分
        for r, avg in region_avg_uncertainty.items():
            if r not in self.region_uncertainty_tracker:
                self.region_uncertainty_tracker[r] = avg  # 第一次出现，直接赋值
            else:
                old_val = self.region_uncertainty_tracker[r]
                self.region_uncertainty_tracker[r] = (self.momentum * old_val) + (1.0 - self.momentum) * avg

        # 3. 使用 Softmax 转换平滑后的不确定性，得到新的采样权重
        sorted_ids = sorted(list(self.all_region_ids))
        uncertainties = np.array([self.region_uncertainty_tracker.get(r, 0) for r in sorted_ids])
        
        # 调试日志
        print("\n" + "="*30)
        print(f"区域权重更新 (T={self.temperature}, M={self.momentum})")
        print("="*30)
        print(f"{'ID':<3} | {'Epoch Avg Entropy':<18} | {'EMA-Smoothed Entropy':<20}")
        print("----|--------------------|---------------------")
        for r in sorted_ids:
            print(f"{r:<3} | {region_avg_uncertainty.get(r, 0):<18.6f} | {self.region_uncertainty_tracker.get(r, 0):<20.6f}")

        # 使用 Temperature Scaling 的 Softmax
        exp_uncertainties = np.exp(uncertainties / self.temperature)
        sum_exp = np.sum(exp_uncertainties)
        
        if sum_exp == 0:
            # 如果所有熵都为0，退回到均匀采样
            softmax_weights = np.ones_like(uncertainties) / len(uncertainties)
        else:
            softmax_weights = exp_uncertainties / sum_exp
            
        # 4. 更新权重字典（包括区域0）
        new_weights = {r: w for r, w in zip(sorted_ids, softmax_weights)}
        new_weights[0] = 0.0 # 确保背景区域的权重为0，我们不想主动采样背景
        
        # 归一化，使总和为1
        total_weight = sum(new_weights.values())
        if total_weight > 0:
            for r in new_weights:
                new_weights[r] /= total_weight
        else: # 如果所有权重都为0（例如只有背景），则均匀分配
             num_regions = len(self.region_weights) if self.region_weights else 1
             new_weights = {r: 1.0/num_regions for r in self.region_weights.keys()}
             if not new_weights: # 处理边缘情况
                 new_weights = {0: 1.0}

        self.region_weights = new_weights
        
        print("\n--- 更新后的采样权重 ---")
        weights_log = [f"ID {r}: {self.region_weights.get(r, 0):.4f}" for r in sorted(self.region_weights.keys()) if r != 0]
        print(", ".join(weights_log))
        print("="*30 + "\n")

        # 5. 重置当前 epoch 的累加器
        self.region_total_uncertainty.clear()
        self.region_total_voxels.clear()
        self.all_region_ids.clear()

        return True



class RandomRotFlip:
    """随机旋转 + 翻转"""
    def __call__(self, sample):
        FLAIR, T1, T2, seg, mask = sample['flair'], sample['t1'], sample['t2'], sample['seg'], sample['mask']
        k = np.random.randint(0, 4)
        FLAIR = np.rot90(FLAIR, k, axes=(0,1))
        T1 = np.rot90(T1, k, axes=(0,1))
        T2 = np.rot90(T2, k, axes=(0,1))
        seg = np.rot90(seg, k, axes=(0,1))
        mask = np.rot90(mask, k, axes=(0,1))
        axis = np.random.randint(0,2)
        FLAIR = np.flip(FLAIR, axis=axis).copy()
        T1 = np.flip(T1, axis=axis).copy()
        T2 = np.flip(T2, axis=axis).copy()
        seg = np.flip(seg, axis=axis).copy()
        mask = np.flip(mask, axis=axis).copy()
        return {'flair':FLAIR,'t1':T1,'t2':T2,'seg':seg,'mask':mask}

class ToTensor:
    """转为 torch.Tensor"""
    def __call__(self, sample):
        FLAIR = torch.from_numpy(sample['flair'][np.newaxis].astype(np.float32))
        T1 = torch.from_numpy(sample['t1'][np.newaxis].astype(np.float32))
        T2 = torch.from_numpy(sample['t2'][np.newaxis].astype(np.float32))
        seg = torch.from_numpy(sample['seg'].astype(np.int64))
        mask = torch.from_numpy(sample['mask'].astype(np.int64))
        return {'flair':FLAIR,'t1':T1,'t2':T2,'seg':seg,'mask':mask}

# -------------------- Compose -------------------- #
class Compose:
    """类似 torchvision.transforms.Compose"""
    def __init__(self, transforms):
        self.transforms = transforms
        # 查找 RandomRegionCrop 用于外部访问累积 loss
        self.region_crop = None
        for t in transforms:
            if isinstance(t, (RandomRegionCrop, AdaptiveRegionSampler)):
                self.region_crop = t

    def __call__(self, sample):
        for t in self.transforms:
            sample = t(sample)
        return sample

# -------------------- Dataset -------------------- #
class AdaptiveEPVSDataset(Dataset):
    """返回 sample 已是 patch，训练循环无需额外采样"""
    def __init__(self, base_dir, split='train', transform=None):
        self._base_dir = base_dir
        self.transform = transform
        txt_file = f"{base_dir}/train1.txt" if split=='train' else f"{base_dir}/val0.txt"
        with open(txt_file,'r') as f:
            self.image_list = [line.strip().split(',')[0] for line in f]

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_name = self.image_list[idx]
        h5f = h5py.File(f"{self._base_dir}/{image_name}",'r')
        sample = {
            'flair': h5f['flair'][:],
            't1': h5f['t1'][:],
            't2': h5f['t2'][:],
            'seg': h5f['seg'][:].astype(np.uint8),
            'mask': h5f['mask'][:].astype(np.uint8)
        }
        if self.transform:
            sample = self.transform(sample)  # patch 采样 + 数据增强 + ToTensor
        return sample, image_name
