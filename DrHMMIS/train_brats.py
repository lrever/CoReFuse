import argparse
import logging
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import random
import sys
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tensorboardX import SummaryWriter
from tqdm import tqdm
import h5py
import torch.nn.functional as F
from networks.mednext_mymodel import ClinPAFNet_MedNeXt_Classic,ClinPAFNet_Feedback
from networks.epvs_brats_model import ClinPAFNet_General
from networks.final_network import Optimized_MultiModal_Net_1
# 项目路径
sys.path.insert(0, '/home/dell/hxy/SSL4MIS/project/workspace')
print(f"系统路径已更新: {sys.path[0]}")

# MONAI Transforms
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    RandSpatialCropd,
    RandFlipd,
    RandRotate90d,
    ToTensord,
)
from monai.inferers import sliding_window_inference

# 自定义模块
from utils import losses 
from medpy import metric
from networks.context_lite import Ablation_ThreeEncoder_Final_Net
from networks.gate_cnn_transformer import ThreeEncoderNaiveFusionUNet1 # 三输入模型
from networks.context import ThreeEncoder_SelectiveAxial_Net
# ============================= 参数配置 =============================
parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='/data4T/brats20_h5')
parser.add_argument('--list_dir', type=str, default='/home/dell/hxy/SSL4MIS/project/workspace/hxy/splits/train_splits')
parser.add_argument('--exp', type=str, default='brats2021_resune')
parser.add_argument('--model', type=str, default='my_model_bad_final1')
parser.add_argument('--max_iterations', type=int, default=10000)
parser.add_argument('--batch_size', type=int, default=2)
parser.add_argument('--deterministic', type=int, default=1)
parser.add_argument('--base_lr', type=float, default=0.05)
parser.add_argument('--patch_size', type=list, default=[96, 96, 96])
parser.add_argument('--seed', type=int, default=1337)
parser.add_argument('--num_classes', type=int, default=4)
parser.add_argument('--val_every', type=int, default=400)

args = parser.parse_args()

# ============================= 数据集 =============================
class BraTS_Final_Dataset(Dataset):
    def __init__(self, base_dir, list_dir, split, transform=None):
        self.transform = transform
        self.base_dir = base_dir
        file_name = 'train1.txt' if split == 'train' else 'val0.txt'
        list_path = os.path.join(self.base_dir, file_name) \
            if os.path.exists(os.path.join(self.base_dir, file_name)) \
            else os.path.join(list_dir, split + '.txt')

        with open(list_path, 'r') as f:
            self.sample_list = [line.strip() for line in f.readlines()]
        logging.info(f"Loaded {len(self.sample_list)} samples for split: {split}")

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case_name = self.sample_list[idx]
        h5_path = os.path.join(self.base_dir, case_name, f"{case_name}.h5") \
            if not case_name.endswith('.h5') else os.path.join(self.base_dir, case_name)

        with h5py.File(h5_path, 'r') as hf:
            t1 = np.array(hf['t1']).astype(np.float32)
            t2 = np.array(hf['t2']).astype(np.float32)
            flair = np.array(hf['flair']).astype(np.float32)
            image = np.stack([t1, t2, flair], axis=0)
            label = np.array(hf['seg']).astype(np.uint8)
            label[label == 4] = 3

        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        return sample

class BraTS_Final_Dataset_1(Dataset):
    def __init__(self, base_dir, list_dir, split, transform=None):
        self.transform = transform
        self.base_dir = base_dir

        file_name = 'train1.txt' if split == 'train' else 'val0.txt'
        list_path = os.path.join(self.base_dir, file_name) \
            if os.path.exists(os.path.join(self.base_dir, file_name)) \
            else os.path.join(list_dir, split + '.txt')

        with open(list_path, 'r') as f:
            self.sample_list = [line.strip() for line in f.readlines()]

        logging.info(f"Loaded {len(self.sample_list)} samples for split={split}")

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case_name = self.sample_list[idx]

        # ===== h5 路径 =====
        if case_name.endswith('.h5'):
            h5_path = os.path.join(self.base_dir, case_name)
        else:
            h5_path = os.path.join(self.base_dir, case_name, f"{case_name}.h5") 

        with h5py.File(h5_path, 'r') as hf:
            # image: [4, D, H, W]
            image_all = np.array(hf['image'], dtype=np.float32)

            # 原顺序: [T1, T1ce, FLAIR, T2]
            # 去掉 T1，仅保留 [T1ce, FLAIR, T2]
            image = image_all[1:4, ...]   # [3, D, H, W]

            # label: [D, H, W]
            label = np.array(hf['seg'], dtype=np.uint8)

            # BraTS 标签规范化
            label[label == 4] = 3

        sample = {
            'image': image,   # [3, D, H, W]
            'label': label    # [D, H, W]
        }

        if self.transform is not None:
            sample = self.transform(sample)

        return sample



def get_segmentation_transforms(patch_size):
    return Compose([
        EnsureChannelFirstd(keys=["label"], channel_dim="no_channel"),
        RandSpatialCropd(keys=["image","label"], roi_size=patch_size, random_size=False),
        RandFlipd(keys=["image","label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image","label"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image","label"], prob=0.5, spatial_axis=2),
        RandRotate90d(keys=["image","label"], prob=0.5, max_k=3),
        ToTensord(keys=["image","label"]),
    ])

# ============================= 验证指标 =============================
def get_brats_regions(label):
    WT = (label > 0).astype(np.uint8)
    TC = ((label == 1) | (label == 3)).astype(np.uint8)
    ET = (label == 3).astype(np.uint8)
    return WT, TC, ET

def cal_metric(pred, gt):
    if gt.sum() == 0:
        return np.nan, np.nan, np.nan, np.nan
    if pred.sum() == 0:
        return 0,0,0,0
    dice = metric.binary.dc(pred, gt)
    hd95 = metric.binary.hd95(pred, gt)
    recall = metric.binary.sensitivity(pred, gt)
    precision = metric.binary.precision(pred, gt)
    return dice, hd95, recall, precision

def validate_on_brats_tc_et(model, valloader, patch_size):
    model.eval()
    metrics = {k: {'dice': [], 'hd95': [], 'recall': [], 'precision': []} for k in ['WT','TC','ET']}

    with torch.no_grad():
        for batch_data in tqdm(valloader, desc="Validating", ncols=70):
            images = batch_data["image"].cuda()
            labels = batch_data["label"].squeeze(1).cpu().numpy()

            def model_inference_wrapper(inputs):
                x_t1 = inputs[:,0:1,...]
                x_t2 = inputs[:,1:2,...]
                x_flair = inputs[:,2:3,...]
                return model(x_t1, x_t2, x_flair)

            val_logits = sliding_window_inference(
                inputs=images,
                roi_size=patch_size,
                sw_batch_size=4,
                predictor=model_inference_wrapper,
                overlap=0.5
            )
            val_preds = torch.argmax(val_logits, dim=1).cpu().numpy()

            for i in range(val_preds.shape[0]):
                pred = val_preds[i]
                gt = labels[i]
                pred_wt, pred_tc, pred_et = get_brats_regions(pred)
                gt_wt, gt_tc, gt_et = get_brats_regions(gt)

                for name,p,g in zip(['WT','TC','ET'],
                                    [pred_wt,pred_tc,pred_et],
                                    [gt_wt,gt_tc,gt_et]):
                    dice, hd95, recall, precision = cal_metric(p,g)
                    metrics[name]['dice'].append(dice)
                    metrics[name]['hd95'].append(hd95)
                    metrics[name]['recall'].append(recall)
                    metrics[name]['precision'].append(precision)

    avg_results = {k:{m: np.nanmean(metrics[k][m]) for m in metrics[k]} for k in metrics}
    model.train()
    return avg_results

# ============================= 训练 =============================
def train(args, snapshot_path):
    model = Ablation_ThreeEncoder_Final_Net(
        base_c=16,
        n_classes=args.num_classes,
        opt_encoder=False,
        opt_fusion_shallow=False,
        opt_fusion_deep=False,
        deep_sup=False
    ).cuda()
    #model=ClinPAFNet_GeneralV2(n_classes=args.num_classes,base_c=16).cuda()
    # model=ClinPAFNet_General(n_classes=4,base_c=16).cuda()
    # model = ThreeEncoder_SelectiveAxial_Net(n_classes=4,
    #     base_c=16,
    #     n_levels=4).cuda()
    deep_sup=False
    optimizer = optim.SGD(model.parameters(), lr=args.base_lr, momentum=0.9, weight_decay=1e-4)
    dice_loss = losses.DiceLoss(args.num_classes)
    focal_loss = losses.FocalLoss()

    train_ds = BraTS_Final_Dataset(args.root_path, args.list_dir, 'train',
                                   get_segmentation_transforms(args.patch_size))
    val_ds = BraTS_Final_Dataset(args.root_path, args.list_dir, 'val')

    trainloader = DataLoader(train_ds, args.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    valloader = DataLoader(val_ds, 1, shuffle=False, num_workers=2)

    writer = SummaryWriter(snapshot_path+'/log')
    iter_num = 0
    best_performance = 0.0

    max_epoch = args.max_iterations // len(trainloader) + 1
    for epoch_num in tqdm(range(max_epoch), desc="Epochs", ncols=70):
        model.train()
        for batch in trainloader:
            if batch is None: continue

            volume_batch = batch['image'].cuda()
            label_batch = batch['label'].cuda()

            x_t1 = volume_batch[:,0:1,...]
            x_t2 = volume_batch[:,1:2,...]
            x_flair = volume_batch[:,2:3,...]
            if deep_sup:
                outputs, ds1, ds2 = model(x_t1, x_t2, x_flair)
                label_squeezed_long = label_batch.squeeze(1).long()
                label_with_channel_long = label_batch.long()

                loss_main = 0.4*focal_loss(outputs,label_squeezed_long) + 0.6*dice_loss(outputs,label_with_channel_long)
                ds1_up = F.interpolate(ds1, size=label_batch.shape[2:], mode='trilinear', align_corners=True)
                ds2_up = F.interpolate(ds2, size=label_batch.shape[2:], mode='trilinear', align_corners=True)
                loss_ds1 = 0.4*focal_loss(ds1_up,label_squeezed_long)+0.6*dice_loss(ds1_up,label_with_channel_long)
                loss_ds2 = 0.4*focal_loss(ds2_up,label_squeezed_long)+0.6*dice_loss(ds2_up,label_with_channel_long)
                total_loss = loss_main + 0.5*loss_ds1 + 0.25*loss_ds2
            else:
                outputs = model(x_t1, x_t2, x_flair)
                label_squeezed_long = label_batch.squeeze(1).long()
                label_with_channel_long = label_batch.long()
                total_loss = 0.4*focal_loss(outputs,label_squeezed_long) + 0.6*dice_loss(outputs,label_with_channel_long)
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            lr = args.base_lr*(1-iter_num/args.max_iterations)**0.9
            for pg in optimizer.param_groups:
                pg['lr'] = lr

            iter_num += 1
            writer.add_scalar('train/loss', total_loss.item(), iter_num)
            writer.add_scalar('train/lr', lr, iter_num)

            if iter_num % 20 == 0:
                if deep_sup:
                    logging.info(f"Iter {iter_num}: loss={total_loss.item():.4f}, main={loss_main.item():.4f}, ds1={loss_ds1.item():.4f}, ds2={loss_ds2.item():.4f}")
                else:
                    logging.info(f"Iter {iter_num}: loss={total_loss.item():.4f}")

            if iter_num % args.val_every == 0 or iter_num >= args.max_iterations:
                logging.info("Starting Validation...")
                avg_results = validate_on_brats_tc_et(model, valloader, args.patch_size)
                val_mean_dice = (avg_results['TC']['dice'] + avg_results['ET']['dice'])/2

                logging.info(
                    f"Val @ iter {iter_num} | Mean TC+ET Dice: {val_mean_dice:.4f}\n"
                    f"WT | Dice: {avg_results['WT']['dice']:.4f}, HD95: {avg_results['WT']['hd95']:.2f}, "
                    f"Recall: {avg_results['WT']['recall']:.4f}, Precision: {avg_results['WT']['precision']:.4f}\n"
                    f"TC | Dice: {avg_results['TC']['dice']:.4f}, HD95: {avg_results['TC']['hd95']:.2f}, "
                    f"Recall: {avg_results['TC']['recall']:.4f}, Precision: {avg_results['TC']['precision']:.4f}\n"
                    f"ET | Dice: {avg_results['ET']['dice']:.4f}, HD95: {avg_results['ET']['hd95']:.2f}, "
                    f"Recall: {avg_results['ET']['recall']:.4f}, Precision: {avg_results['ET']['precision']:.4f}"
                )

                for k in avg_results:
                    for m in avg_results[k]:
                        writer.add_scalar(f'val/{k}_{m}', avg_results[k][m], iter_num)

                if val_mean_dice > best_performance:
                    best_performance = val_mean_dice
                    torch.save(model.state_dict(), os.path.join(snapshot_path,'best_model.pth'))
                    logging.info(f"Saved Best Model (Mean TC+ET Dice={best_performance:.4f})")

            if iter_num >= args.max_iterations:
                break

        if iter_num >= args.max_iterations:
            break

    writer.close()
    logging.info("Training Finished!")

# ============================= 主函数 =============================
if __name__=="__main__":
    cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    snapshot_path = f"./self-brats20_mednext_mymodel_401/model/{args.exp}/{args.model}"
    os.makedirs(snapshot_path, exist_ok=True)

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    train(args, snapshot_path)
