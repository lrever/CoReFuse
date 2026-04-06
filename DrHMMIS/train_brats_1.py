import argparse
import logging
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import random
import sys
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tensorboardX import SummaryWriter
from tqdm import tqdm
import h5py
from collections import defaultdict

# 引入项目路径
sys.path.insert(0, '/home/dell/hxy/SSL4MIS/project/workspace') 

# ==============================================================================
# MONAI & Metrics Imports
# ==============================================================================
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    RandFlipd,
    RandRotate90d,
    ToTensord,
    MapTransform,
)
from monai.losses import FocalLoss
from monai.inferers import sliding_window_inference
from medpy import metric

# 自定义网络
from networks.dual_enocder_unet import Ablation_NEncoder_Final_Net
from utils import losses as utils_losses

# ==============================================================================
# 参数配置
# ==============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='/data4T/brats20_h5', help='BraTS H5 Data Root')
parser.add_argument('--list_dir', type=str, default='/home/dell/hxy/SSL4MIS/project/workspace/hxy/splits/train_splits', help='Txt split files')
parser.add_argument('--exp', type=str, default='brats2021_anatomy_mask_corrected', help='experiment_name')
parser.add_argument('--model', type=str, default='no_encoder', help='model_name')
parser.add_argument('--max_iterations', type=int, default=10000, help='maximum iteration number to train')
parser.add_argument('--batch_size', type=int, default=2, help='batch_size per gpu')
parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.001, help='segmentation network learning rate')
parser.add_argument('--patch_size', type=int, nargs='+', default=[96, 96, 96], help='patch size of network input')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--num_classes', type=int,  default=4, help='output channel of network')
parser.add_argument('--val_every', type=int,  default=400, help='validation frequency')

args = parser.parse_args()

# ==============================================================================
# 1. 解剖区域 Loss 管理器
# ==============================================================================
class AnatomyLossManager:
    def __init__(self, momentum=0.7, temperature=0.1):
        self.momentum = momentum
        self.temperature = temperature
        
        self.ema_losses = {} 
        self.region_probs = {}
        
        self.epoch_loss_sum = defaultdict(float)
        self.epoch_voxel_count = defaultdict(int)
        
        logging.info(f"AnatomyLossManager Init: Momentum={momentum}, Temp={temperature}")

    def update(self, pixel_loss_map, anatomy_mask):
        """ 
        pixel_loss_map: [B, D, H, W] (模型对 label 预测的 Loss)
        anatomy_mask:   [B, D, H, W] (脑区解剖 Mask，用于统计分区域 Loss)
        """
        if isinstance(pixel_loss_map, torch.Tensor):
            pixel_loss_map = pixel_loss_map.detach().cpu().numpy()
        if isinstance(anatomy_mask, torch.Tensor):
            anatomy_mask = anatomy_mask.detach().cpu().numpy()
            
        batch_size = pixel_loss_map.shape[0]
        
        for b in range(batch_size):
            loss_b = pixel_loss_map[b]
            mask_b = anatomy_mask[b]
            
            # 统计 Mask 中存在的脑区 ID
            present_regions = np.unique(mask_b)
            
            for r in present_regions:
                r = int(r)
                # 忽略背景区域 (假设 ID 0 是背景，如果脑区包含 0 则去掉此判断)
                if r == 0: continue 

                region_pixels = (mask_b == r)
                count = np.sum(region_pixels)
                
                if count > 0:
                    avg_val = loss_b[region_pixels].mean()
                    
                    self.epoch_loss_sum[r] += loss_b[region_pixels].sum()
                    self.epoch_voxel_count[r] += count
                    
                    if r not in self.ema_losses:
                        self.ema_losses[r] = avg_val
                    else:
                        self.ema_losses[r] = self.momentum * self.ema_losses[r] + \
                                             (1 - self.momentum) * avg_val
        
        self._normalize_probs()

    def _normalize_probs(self):
        if not self.ema_losses:
            return
        regions = list(self.ema_losses.keys())
        losses = np.array([self.ema_losses[r] for r in regions])
        
        scaled_losses = losses / self.temperature
        scaled_losses -= np.max(scaled_losses)
        exp_vals = np.exp(scaled_losses)
        probs = exp_vals / np.sum(exp_vals)
        
        for r, p in zip(regions, probs):
            self.region_probs[r] = p

    def get_probs(self):
        return self.region_probs if self.region_probs else None

    def print_epoch_stats(self, epoch):
        logging.info("\n" + "="*80)
        logging.info(f"Epoch {epoch} Anatomy Loss Stats (Momentum={self.momentum}, Temp={self.temperature})")
        logging.info("="*80)
        logging.info(f"{'Region ID':<10} | {'Avg Loss':<18} | {'EMA Loss':<15} | {'Next Prob':<15}")
        logging.info("-" * 80)
        
        sorted_ids = sorted(self.ema_losses.keys())
        for r in sorted_ids:
            if self.epoch_voxel_count[r] > 0:
                epoch_avg = self.epoch_loss_sum[r] / self.epoch_voxel_count[r]
            else:
                epoch_avg = 0.0
            
            ema_val = self.ema_losses.get(r, 0.0)
            prob_val = self.region_probs.get(r, 0.0)
            logging.info(f"{r:<10d} | {epoch_avg:<18.6f} | {ema_val:<15.6f} | {prob_val:<15.4f}")
        logging.info("="*80 + "\n")
        
        self.epoch_loss_sum.clear()
        self.epoch_voxel_count.clear()

# ==============================================================================
# 2. 自定义 Transform (针对 Mask 进行采样)
# ==============================================================================
class AnatomyAwareCrop(MapTransform):
    def __init__(self, keys, manager, patch_size, mask_key='mask'):
        super().__init__(keys)
        self.manager = manager
        self.patch_size = patch_size
        self.mask_key = mask_key  # 这里指定使用的是 'mask' 字段

    def __call__(self, data):
        d = dict(data)
        
        # 1. 获取脑区 Mask (不是 label)
        anatomy_mask = d[self.mask_key]
        if isinstance(anatomy_mask, torch.Tensor):
            anatomy_mask = anatomy_mask.numpy()
        
        # 兼容 Channel First [C, D, H, W] 或 [D, H, W]
        if anatomy_mask.ndim == 4:
            spatial_mask = anatomy_mask[0]
        else:
            spatial_mask = anatomy_mask

        # 2. 确定采样区域 ID
        present_regions = np.unique(spatial_mask).astype(int)
        # 过滤掉背景 0 (如果不需要在背景采样)
        present_regions = present_regions[present_regions != 0]

        if len(present_regions) == 0:
            # 如果全是背景，随机采样
            center = np.array(spatial_mask.shape) // 2
        else:
            global_probs = self.manager.get_probs()
            
            if global_probs is None:
                # 冷启动：均匀分布
                target_probs = np.ones(len(present_regions)) / len(present_regions)
            else:
                # 根据 Loss 权重采样
                probs_arr = np.array([global_probs.get(r, 1e-6) for r in present_regions])
                target_probs = probs_arr / (probs_arr.sum() + 1e-9)
                
            target_id = np.random.choice(present_regions, p=target_probs)
            
            # 3. 在选定区域内随机选一个中心点
            candidates = np.argwhere(spatial_mask == target_id)
            if len(candidates) > 0:
                center = candidates[np.random.randint(len(candidates))]
            else:
                center = np.array(spatial_mask.shape) // 2
            
        # 4. 计算 Patch 范围
        crop_slices = []
        for c, p, dim in zip(center, self.patch_size, spatial_mask.shape):
            start = max(0, min(c - p // 2, dim - p))
            crop_slices.append(slice(start, start + p))
        slices_tuple = tuple(crop_slices)
        
        # 5. 对所有 Keys (image, label, mask) 执行裁剪
        for key in self.keys:
            if key in d:
                obj = d[key]
                if obj.ndim == 4: # [C, D, H, W]
                    d[key] = obj[:, slices_tuple[0], slices_tuple[1], slices_tuple[2]]
                elif obj.ndim == 3: # [D, H, W]
                    d[key] = obj[slices_tuple[0], slices_tuple[1], slices_tuple[2]]
        
        return d

def get_transforms(patch_size, manager):
    # 关键：这三个 Key 都要一起做变换，保持对齐
    train_keys = ["image", "label", "mask"]
    return Compose([
        EnsureChannelFirstd(keys=["label", "mask"], channel_dim="no_channel"),
        # 使用 AnatomyAwareCrop，基于 "mask" 进行采样
        AnatomyAwareCrop(keys=train_keys, manager=manager, patch_size=patch_size, mask_key="mask"),
        # 空间增强必须同时作用于 image, label 和 mask
        RandFlipd(keys=train_keys, prob=0.5, spatial_axis=0),
        RandFlipd(keys=train_keys, prob=0.5, spatial_axis=1),
        RandFlipd(keys=train_keys, prob=0.5, spatial_axis=2),
        RandRotate90d(keys=train_keys, prob=0.5, max_k=3),
        ToTensord(keys=train_keys),
    ])

# ==============================================================================
# 3. Dataset (读取 Mask)
# ==============================================================================
class BraTS_Final_Dataset(Dataset):
    def __init__(self, base_dir, list_dir, split, transform=None):
        self.transform = transform
        self.base_dir = base_dir
        self.sample_list = []
        
        file_name = 'train1.txt' if split == 'train' else 'val0.txt'
        list_path_h5 = os.path.join(self.base_dir, file_name)
        if os.path.exists(list_path_h5):
            list_path = list_path_h5
        else:
            list_path = os.path.join(list_dir, split + '.txt')

        with open(list_path, 'r') as f:
            self.sample_list = [line.strip() for line in f.readlines()]
        logging.info(f"Loaded {len(self.sample_list)} samples for split: {split}")

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case_name = self.sample_list[idx]
        h5_path = os.path.join(self.base_dir, case_name) if case_name.endswith('.h5') else os.path.join(self.base_dir, case_name, f"{case_name}.h5")

        try:
            with h5py.File(h5_path, 'r') as hf:
                # 1. Image
                t1 = np.array(hf['t1']).astype(np.float32)
                t2 = np.array(hf['t2']).astype(np.float32)
                flair = np.array(hf['flair']).astype(np.float32)
                image = np.stack([t1, t2, flair], axis=0) # [3, D, H, W]

                # 2. Label (Seg)
                label = np.array(hf['seg']).astype(np.uint8)
                label[label == 4] = 3 
                
                # 3. Mask (Anatomy Mask)
                if 'mask' in hf.keys():
                    mask = np.array(hf['mask']).astype(np.uint8)
                else:
                    # 如果没有 mask，暂时用前景代替，或者报错
                    # logging.warning(f"No 'mask' found in {case_name}, using foreground.")
                    mask = (np.sum(image, axis=0) > 0).astype(np.uint8)
                    
        except Exception as e:
            logging.error(f"Error loading {h5_path}: {e}")
            return self.__getitem__(np.random.randint(len(self.sample_list)))
            
        # 返回 image, label, mask
        sample = {'image': image, 'label': label, 'mask': mask}
        if self.transform:
            sample = self.transform(sample)
        return sample

# ==============================================================================
# 4. 验证函数 (Recall & Precision & TC/ET)
# ==============================================================================
def get_brats_regions(label):
    TC = ((label == 1) | (label == 3)).astype(np.uint8)
    ET = (label == 3).astype(np.uint8)
    return None, TC, ET

def cal_metric(pred, gt):
    if gt.sum() == 0:
        return np.nan, np.nan, np.nan, np.nan
    if pred.sum() == 0:
        return 0, 0, 0, 0
    
    dice = metric.binary.dc(pred, gt)
    hd95 = metric.binary.hd95(pred, gt)
    recall = metric.binary.sensitivity(pred, gt)
    precision = metric.binary.precision(pred, gt)
    return dice, hd95, recall, precision

def validate_on_brats_tc_et(model, valloader, patch_size):
    model.eval()
    dice_all = {"TC": [], "ET": []}
    hd95_all = {"TC": [], "ET": []}
    recall_all = {"TC": [], "ET": []}
    precision_all = {"TC": [], "ET": []}

    with torch.no_grad():
        for batch_data in tqdm(valloader, desc="Validating", ncols=70):
            images = batch_data["image"].cuda()           
            labels = batch_data["label"].squeeze(1).cpu().numpy()
            
            # 注意：验证时通常不需要 Anatomy Mask，除非你想评估特定脑区的性能
            # 这里我们只用 image 和 label 计算病灶指标

            def model_inference_wrapper(inputs):
                x_t1 = inputs[:, 0:1, ...]
                x_t2 = inputs[:, 1:2, ...]
                x_flair = inputs[:, 2:3, ...]
                return model(x_t1, x_t2, x_flair)

            val_logits = sliding_window_inference(
                inputs=images, 
                roi_size=patch_size, 
                sw_batch_size=4, 
                predictor=model_inference_wrapper,
                overlap=0.6 
            )
            
            val_preds = torch.argmax(val_logits, dim=1).cpu().numpy()

            for i in range(val_preds.shape[0]):
                pred = val_preds[i]
                gt = labels[i]
                _, pred_tc, pred_et = get_brats_regions(pred)
                _, gt_tc, gt_et = get_brats_regions(gt)

                for name, p_bin, g_bin in zip(["TC", "ET"], [pred_tc, pred_et], [gt_tc, gt_et]):
                    dice, hd95, recall, precision = cal_metric(p_bin, g_bin)
                    dice_all[name].append(dice)
                    hd95_all[name].append(hd95)
                    recall_all[name].append(recall)
                    precision_all[name].append(precision)

    avg_results = {}
    for region in ["TC", "ET"]:
        avg_results[region] = {
            "dice": np.nanmean(dice_all[region]), 
            "hd95": np.nanmean(hd95_all[region]),
            "recall": np.nanmean(recall_all[region]),
            "precision": np.nanmean(precision_all[region])
        }
    return avg_results

# ==============================================================================
# 5. 训练主流程
# ==============================================================================
def train(args, snapshot_path):
    anatomy_manager = AnatomyLossManager(momentum=0.7, temperature=0.05)
    
    # 传入 transforms
    train_transforms = get_transforms(args.patch_size, anatomy_manager)
    
    db_train = BraTS_Final_Dataset(args.root_path, args.list_dir, 'train', transform=train_transforms)
    trainloader = DataLoader(db_train, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    
    db_val = BraTS_Final_Dataset(args.root_path, args.list_dir, 'val', transform=None)
    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=2)

    model = model = Ablation_NEncoder_Final_Net(
        n_classes=3, 
        num_modalities=3, 
        base_c=32, 
        deep_sup=False
    ).cuda() 
    optimizer = optim.SGD(model.parameters(), lr=args.base_lr, momentum=0.9, weight_decay=1e-4)
    
    train_focal = utils_losses.FocalLoss() 
    train_dice = utils_losses.DiceLoss(n_classes=args.num_classes)
    feedback_loss_func = FocalLoss(include_background=True, to_onehot_y=True, gamma=2.0, reduction="none")

    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_iterations // len(trainloader) + 1
    best_performance = 0.0
    
    logging.info(f"{len(trainloader)} iterations per epoch. Total {max_epoch} epochs.")

    for epoch_num in range(max_epoch):
        model.train()
        pbar = tqdm(trainloader, ncols=80, desc=f"Epoch {epoch_num}")
        
        for sampled_batch in pbar:
            if sampled_batch is None: continue
            
            vol_batch = sampled_batch['image'].cuda()   # [B, 3, D, H, W]
            label_batch = sampled_batch['label'].cuda() # [B, 1, D, H, W]
            # 获取 Transform 后的 Mask (与 image/label 对应的 patch)
            mask_batch = sampled_batch['mask'].cuda()   # [B, 1, D, H, W]

            x_t1 = vol_batch[:, 0:1, ...]
            x_t2 = vol_batch[:, 1:2, ...]
            x_flair = vol_batch[:, 2:3, ...]

            outputs, ds1, ds2 = model(x_t1, x_t2, x_flair)

            # --- Label Loss (病灶分割) ---
            target_focal = label_batch.squeeze(1).long()
            loss_main = 0.4 * train_focal(outputs, target_focal) + 0.6 * train_dice(outputs, label_batch)
            
            target_shape = label_batch.shape[2:]
            ds1_up = F.interpolate(ds1, size=target_shape, mode='trilinear', align_corners=True)
            loss_ds1 = 0.4 * train_focal(ds1_up, target_focal) + 0.6 * train_dice(ds1_up, label_batch)
            
            ds2_up = F.interpolate(ds2, size=target_shape, mode='trilinear', align_corners=True)
            loss_ds2 = 0.4 * train_focal(ds2_up, target_focal) + 0.6 * train_dice(ds2_up, label_batch)
            
            total_loss = loss_main + 0.5 * loss_ds1 + 0.25 * loss_ds2

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # --- Anatomy Feedback (使用 Mask 更新权重) ---
            with torch.no_grad():
                # 计算病灶预测的 Loss Map
                monai_loss_out = feedback_loss_func(outputs, label_batch)
                spatial_loss_map = torch.sum(monai_loss_out, dim=1) 
                
                # 传入对应的脑区 Mask，统计每个脑区的 Loss
                mask_cpu = mask_batch.detach().cpu().squeeze(1)
                anatomy_manager.update(spatial_loss_map, mask_cpu)

            # LR Schedule
            lr_ = args.base_lr * (1.0 - iter_num / args.max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_
            
            iter_num += 1
            writer.add_scalar('info/total_loss', total_loss.item(), iter_num)
            
            # --- 验证逻辑 ---
            if iter_num > 0 and iter_num % args.val_every == 0:
                logging.info(f"Iteration {iter_num}: Starting Validation...")
                avg_results = validate_on_brats_tc_et(model, valloader, args.patch_size)
                model.train() 

                val_mean_dice = (avg_results['TC']['dice'] + avg_results['ET']['dice']) / 2
                
                logging.info(
                    f"Iter {iter_num} | Mean Dice: {val_mean_dice:.4f} | "
                    f"TC [Dice: {avg_results['TC']['dice']:.4f}, Rec: {avg_results['TC']['recall']:.4f}, Prec: {avg_results['TC']['precision']:.4f}] | "
                    f"ET [Dice: {avg_results['ET']['dice']:.4f}, Rec: {avg_results['ET']['recall']:.4f}, Prec: {avg_results['ET']['precision']:.4f}]"
                )
                
                writer.add_scalar('val/TC_dice', avg_results['TC']['dice'], iter_num)
                writer.add_scalar('val/ET_dice', avg_results['ET']['dice'], iter_num)
                writer.add_scalar('val/Mean_dice', val_mean_dice, iter_num)
                
                writer.add_scalar('val/TC_recall', avg_results['TC']['recall'], iter_num)
                writer.add_scalar('val/ET_recall', avg_results['ET']['recall'], iter_num)
                writer.add_scalar('val/TC_precision', avg_results['TC']['precision'], iter_num)
                writer.add_scalar('val/ET_precision', avg_results['ET']['precision'], iter_num)
                
                if val_mean_dice > best_performance:
                    best_performance = val_mean_dice
                    save_best = os.path.join(snapshot_path, 'best_model.pth')
                    torch.save(model.state_dict(), save_best)
                    logging.info(f"Saved Best Model: {best_performance:.4f}")
                    
                save_latest = os.path.join(snapshot_path, 'latest_model.pth')
                torch.save(model.state_dict(), save_latest)

            if iter_num >= args.max_iterations: break
        
        # Epoch 结束：打印解剖区域权重统计
        anatomy_manager.print_epoch_stats(epoch_num)

        if iter_num >= args.max_iterations: break

    writer.close()
    return "Finished"

if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    snapshot_path = "./self-brats20_optimized/model/{}/{}".format(args.exp, args.model)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    
    train(args, snapshot_path)