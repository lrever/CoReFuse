import argparse
import logging
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import random
import shutil
import sys
import time
sys.path.insert(0, '/home/dell/hxy/SSL4MIS/project/workspace') 
print(f"系统路径已更新: {sys.path}")
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn import BCEWithLogitsLoss
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm import tqdm
from utils.losses import interaction_losses_triplet
from dataloaders import utils
from dataloaders.epvs import (epvs_001, CenterCrop, RandomCrop,
                                   RandomRotFlip, ToTensor,
                                   TwoStreamBatchSampler)
from dataloaders.adptive_dataloader import AdaptiveEPVSDataset,Compose,ToTensor,RandomRotFlip,RandomRegionCrop,AdaptiveRegionSampler 
from utils import losses, metrics, ramps
from val_3D import test_all_case_epvs,test_all_case_epvs_check
from monai.losses import DiceLoss, FocalLoss # 推荐使用MONAI                                                                                                                                                                             
import os
from networks.dual_encoder import Optimized_DynamicModal_Net

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='/data4T/epvs_self-dataset/h5', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='mednext', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='mednext_dynamic', help='model_name')
parser.add_argument('--max_iterations', type=int,
                    default=20000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=2,
                    help='batch_size per gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.001,
                    help='segmentation network learning rate')
parser.add_argument('--patch_size', type=list,  default=[96, 96, 96],
                    help='patch size of network input')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--labeled_num', type=int, default=25,
                    help='labeled data')

args = parser.parse_args()

def train(args, snapshot_path):
    base_lr = args.base_lr
    train_data_path = args.root_path
    batch_size = args.batch_size
    max_iterations = args.max_iterations
    num_classes = 2
    NUM_REGIONS=11
    opt_encoder=True
    opt_fusion_deep=True 
    opt_fusion_shallow=True
    deep_sup=False
    patch_check=True
    #model = net_factory_3d(net_type=args.model, in_chns=3, class_num=num_classes,opt_encoder=opt_encoder,opt_fusion_deep=opt_fusion_deep,opt_fusion_shallow=opt_fusion_shallow,deep_sup=deep_sup)
    # model=ClinPAFNet_Ablation(
    #     n_classes=2, base_c=16
    # ).cuda()
    # model=Optimized_DynamicModal_Net(n_classes=2, base_c=16, num_modalities=3).cuda()
    model = Ablation_ThreeEncoder_Final_Net(n_classes=num_classes, base_c=16, n_levels=4, 
                 opt_encoder=True,        # True=Hybrid(Medium), False=Standard(SimpleStem)
                 opt_fusion_shallow=True, # True=UltraLite(DS), False=Naive(Concat)
                 opt_fusion_deep=True,    # True=Calibration, False=Naive(Concat)
                 deep_sup=deep_sup).cuda()
    # model = UNet3D(in_channels=3,n_classes=2, base_c=32).cuda()

    if patch_check==False:
        db_train = epvs_001(base_dir=train_data_path,
                            split='train',
                            transform=transforms.Compose([
                                RandomRotFlip(),
                                RandomCrop(args.patch_size),
                                ToTensor(),
                            ]))
        
    else:
        # 统计分布
        region_cropper = RandomRegionCrop(patch_size=(96,96,96))
        # region_sampler = AdaptiveRegionSampler(patch_size=(96,96,96), num_classes=2, temperature=0.1, momentum=0.9)
        transform = Compose([region_cropper, RandomRotFlip(), ToTensor()])

        db_train = AdaptiveEPVSDataset(base_dir=train_data_path, split='train', transform=transform)

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)  
    #region_cropper = db_train.transform.region_crop
    if patch_check:
        if not region_cropper:
            # 如果你注释掉了上面的 region_crop 行，就会发生这种情况
            logging.warning("在 transforms 中未找到 RandomRegionCrop。自适应采样已禁用。")
    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True,
                             num_workers=16, pin_memory=True, worker_init_fn=worker_init_fn)
    #sampler_instance = db_train.transform.region_crop
    # if not isinstance(sampler_instance, AdaptiveRegionSampler):
    #      logging.warning("未能在 transforms 中找到 AdaptiveRegionSampler。自适应采样将无法工作。")
    #      sampler_instance = None
    model.train()
    # optimizer = optim.SGD(model.parameters(), lr=base_lr,
    #                       momentum=0.9, weight_decay=0.0001)
    optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=0.0001)
    #ce_loss = losses.BinaryFocalLoss(alpha=0.75, gamma=2.0)
    ce_loss = FocalLoss(to_onehot_y=True, gamma=2.0, use_softmax=True)
    ce_loss_voxel = FocalLoss(
    to_onehot_y=True,
    gamma=2.0,
    use_softmax=True,
    reduction='none'  # <-- 返回每个 voxel 的损失
)
    dice_loss = losses.DiceLoss(2)
    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        if patch_check:
            if region_cropper.region_weights:
                logging.info(f"Epoch {epoch_num} 采样权重: {region_cropper.region_weights}")
        for i_batch, sample in enumerate(trainloader):
            sampled_batch,image_name=sample[0],sample[1]
            input_batch=[]
            flair_batch, t1_batch, t2_batch, label_batch, mask_batch = (
                sampled_batch['flair'].cuda(),
                sampled_batch['t1'].cuda(),
                sampled_batch['t2'].cuda(),
                sampled_batch['seg'].cuda(),
                sampled_batch['mask']
            )
            input_batch=torch.cat([flair_batch,t1_batch,t2_batch],dim=1)
            atlas_one_hot = F.one_hot(mask_batch,num_classes=11)
            atlas_input = atlas_one_hot.permute(0, 4, 1, 2, 3).float()
            # input_batch.append(flair_batch)
            # input_batch.append(t1_batch)
            # input_batch.append(t2_batch)
            #prediction, w_spatial = model(input_batch,atlas_input)

            ################################
            # # outputs = model(input_batch)
            # outputs,_ = model(t1_batch,t2_batch,flair_batch)
            # # outputs=model(t2_batch)
            # outputs_soft = torch.softmax(outputs, dim=1)
            # loss_ce = ce_loss(outputs, label_batch.unsqueeze(1))
            # loss_dice = dice_loss(outputs_soft, label_batch.unsqueeze(1))
            # loss =  loss_dice + loss_ce
            ################################
            if deep_sup==False:
                #outputs = model(t1_batch,t2_batch,flair_batch)
                outputs=model(input_batch)
                outputs_soft = torch.softmax(outputs, dim=1)
                loss_ce = ce_loss(outputs, label_batch.unsqueeze(1))
                loss_dice = dice_loss(outputs_soft, label_batch.unsqueeze(1))
                focal_map = ce_loss_voxel(outputs, label_batch.unsqueeze(1))  # [B,1,D,H,W]
                
                focal_map = focal_map.squeeze(1)  # [B,D,H,W]
                loss =  loss_dice + loss_ce
            else:
                outputs, ds1, ds2 = model(t1_batch, t2_batch, flair_batch)
                # 2. 准备 Ground Truth
                # 假设 label_batch shape 为 [B, D, H, W]
                target = label_batch.unsqueeze(1) # [B, 1, D, H, W]
                target_size = target.shape[2:]    # (D, H, W)

                # 3. 对辅助输出进行上采样 (Upsample to GT size)
                ds1_up = F.interpolate(ds1, size=target_size, mode='trilinear', align_corners=True)
                ds2_up = F.interpolate(ds2, size=target_size, mode='trilinear', align_corners=True)

                # -----------------------------------------------------------
                # 4. 计算 Main Loss (权重 1.0)
                # -----------------------------------------------------------
                outputs_soft = torch.softmax(outputs, dim=1)
                loss_ce_main = ce_loss(outputs, target)
                loss_dice_main = dice_loss(outputs_soft, target)
                loss_main = loss_ce_main + loss_dice_main
                focal_map = ce_loss_voxel(outputs, label_batch.unsqueeze(1)) 
                loss_ce=loss_ce_main
                loss_dice=loss_dice_main
                # -----------------------------------------------------------
                # 5. 计算 Deep Supervision Loss (权重通常为 0.5, 0.25)
                # -----------------------------------------------------------
                # DS1 Loss
                ds1_soft = torch.softmax(ds1_up, dim=1)
                loss_ce_ds1 = ce_loss(ds1_up, target)
                loss_dice_ds1 = dice_loss(ds1_soft, target)
                loss_ds1 = loss_ce_ds1 + loss_dice_ds1

                # DS2 Loss
                ds2_soft = torch.softmax(ds2_up, dim=1)
                loss_ce_ds2 = ce_loss(ds2_up, target)
                loss_dice_ds2 = dice_loss(ds2_soft, target)
                loss_ds2 = loss_ce_ds2 + loss_dice_ds2

                # -----------------------------------------------------------
                # 6. 总 Loss 加权求和
                # -----------------------------------------------------------
                # 推荐权重: Main=1.0, DS1=0.5, DS2=0.25
                loss = loss_main + 0.5 * loss_ds1 + 0.25 * loss_ds2




            # 累积区域权重用 voxel-wise loss
            if patch_check:
                if region_cropper:
                    # 使用 .detach() 来确保不为此图存储梯度，节省内存
                    region_cropper.accumulate_patch_loss(focal_map.detach(), mask_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)
            writer.add_scalar('info/loss_dice', loss_dice, iter_num)
            # writer.add_scalar('info/loss_dice_region', loss_region, iter_num)
            # logging.info(
            #     'iteration %d : loss : %f' %
            #     (iter_num, loss.item()))
            logging.info(
                'iteration %d : loss : %f, loss_ce: %f, loss_dice: %f' %
                (iter_num, loss.item(), loss_ce.item(), loss_dice.item()))

            writer.add_scalar('loss/loss', loss, iter_num)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        
            if iter_num % 20 == 0:
                print(f"DEBUG Iter {iter_num}: Label sum = {label_batch.sum().item()}, Label mean = {label_batch.float().mean().item()}")
            if iter_num > 0 and iter_num % 200 == 0:
                model.eval()
                avg_metric = test_all_case_epvs_check(
                    model, args.root_path, test_list="val1.txt", num_classes=2, patch_size=args.patch_size,
                    stride_xy=64, stride_z=64)
                
                val_dice = avg_metric[:, 0].mean()
                val_hd95 = avg_metric[:, 1].mean()
                
                # [新增] 获取 Recall 和 Precision
                # 请确保你在 val_3D.py 中已经把它们放在了第 2 和 第 3 列
                val_recall = avg_metric[:, 2].mean()
                val_precision = avg_metric[:, 3].mean()
                if avg_metric[:, 0].mean() > best_performance:
                    best_performance = avg_metric[:, 0].mean()
                    save_mode_path = os.path.join(snapshot_path,
                                                  'iter_{}_dice_{}.pth'.format(
                                                      iter_num, round(best_performance, 4)))
                    save_best = os.path.join(snapshot_path,
                                             '{}_best_model.pth'.format(args.model))
                    torch.save(model.state_dict(), save_mode_path)
                    torch.save(model.state_dict(), save_best)
                writer.add_scalar('info/val_dice_score', val_dice, iter_num)
                writer.add_scalar('info/val_hd95', val_hd95, iter_num)
                writer.add_scalar('info/val_recall', val_recall, iter_num)       # <--- 新增
                writer.add_scalar('info/val_precision', val_precision, iter_num) # <--- 新增

                # [修改] 日志打印新增指标
                logging.info(
                    'iteration %d : dice_score : %f hd95 : %f recall : %f precision : %f' % 
                    (iter_num, val_dice, val_hd95, val_recall, val_precision)
                )

                
                
                model.train()

            if iter_num % 3000 == 0:
                save_mode_path = os.path.join(
                    snapshot_path, 'iter_' + str(iter_num) + '.pth')                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     
                torch.save(model.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))

            if iter_num >= max_iterations:
                break
        if patch_check:
            if region_cropper:
                logging.info(f"Epoch {epoch_num} 结束. 正在更新区域权重...")
                update_success = region_cropper.update_region_weights()
                
                if update_success:
                    logging.info(f"--- 下一个 Epoch 的新权重 ---")
                    logging.info(region_cropper.region_weights)
                else:
                    logging.warning("权重更新失败 (某些区域可能未被采样到).")
        if iter_num >= max_iterations:
            iterator.close()
            break

    writer.close()
    return "Training Finished!"


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

    snapshot_path = "./self-epvs/model/{}/{}".format(args.exp, args.model)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    train(args, snapshot_path)