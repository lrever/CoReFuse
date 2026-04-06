import os
import torch
import numpy as np
import SimpleITK as sitk
from torch.utils.data import DataLoader
from tqdm import tqdm
from collections import defaultdict
from torchvision.transforms import Compose
from train_wmh import WMHDataset,RobustIntensityNormalize
import scipy
from networks.dual_encoder import Optimized_DynamicModal_Net
from monai.inferers import sliding_window_inference
from monai.networks.nets import SwinUNETR
from networks.dual_enocder_unet import Ablation_NEncoder_Final_Net
# ==========================================
# 1. 核心评估指标计算类 (对齐官方标准)
# ==========================================
class OfficialMetricCalculator:
    def __init__(self):
        self.cc_filter = sitk.ConnectedComponentImageFilter()
        self.cc_filter.SetFullyConnected(True)

    def calculate_subject_metrics(self, pred_mask, gt_mask, spacing):
        """计算单个受试者的 5 项核心指标 + 分层 Recall"""
        # 转为 SimpleITK 图像并设置 Spacing
        p_img = sitk.GetImageFromArray(pred_mask.transpose(2, 1, 0))
        g_img = sitk.GetImageFromArray(gt_mask.transpose(2, 1, 0))
        p_img.SetSpacing(tuple(float(s) for s in spacing))
        g_img.SetSpacing(tuple(float(s) for s in spacing))

        # 1. Dice (DSC)
        dice = self._get_dice(p_img, g_img)
        
        # 2. H95 (mm)
        h95 = self._get_h95(p_img, g_img)
        
        # 3. lAVD (Absolute log-transformed volume difference) 
        lavd = self._get_lavd(p_img, g_img)
        
        # 4 & 5. Lesion-wise Recall & F1 [cite: 195, 196]
        recall, f1, r_small, r_large = self._get_lesion_metrics(p_img, g_img)

        return {
            "dice": dice, "h95": h95, "lavd": lavd, 
            "recall": recall, "f1": f1,
            "r_small": r_small, "r_large": r_large
        }

    def _get_dice(self, p_img, g_img):
        p_arr = sitk.GetArrayFromImage(p_img).flatten()
        g_arr = sitk.GetArrayFromImage(g_img).flatten()
        if np.sum(p_arr) + np.sum(g_arr) == 0: return 1.0
        return 2. * np.sum(p_arr * g_arr) / (np.sum(p_arr) + np.sum(g_arr))

    def _get_h95(self, p_img, g_img):
        # 官方 2D 腐蚀边缘提取逻辑
        e_p = sitk.BinaryErode(p_img, (1, 1, 0))
        e_g = sitk.BinaryErode(g_img, (1, 1, 0))
        h_p = sitk.GetArrayFromImage(sitk.Subtract(p_img, e_p))
        h_g = sitk.GetArrayFromImage(sitk.Subtract(g_img, e_g))
        
        pts_p = [p_img.TransformIndexToPhysicalPoint(idx[::-1].tolist()) for idx in np.transpose(np.nonzero(h_p))]
        pts_g = [g_img.TransformIndexToPhysicalPoint(idx[::-1].tolist()) for idx in np.transpose(np.nonzero(h_g))]
        
        if not pts_p or not pts_g: return 100.0 # 默认最大惩罚
        
        tree_p = scipy.spatial.KDTree(pts_p)
        tree_g = scipy.spatial.KDTree(pts_g)
        d_p_g = tree_p.query(pts_g, k=1)[0]
        d_g_p = tree_g.query(pts_p, k=1)[0]
        return max(np.percentile(d_p_g, 95), np.percentile(d_g_p, 95))

    def _get_lavd(self, p_img, g_img):
        # 官方 lAVD 公式: |log(SegVol / TrueVol)| [cite: 216, 217]
        v_p = np.sum(sitk.GetArrayFromImage(p_img)) + 1e-6
        v_g = np.sum(sitk.GetArrayFromImage(g_img)) + 1e-6
        return np.abs(np.log(v_p / v_g))

    def _get_lesion_metrics(self, p_img, g_img):
        cc_g = self.cc_filter.Execute(g_img)
        stats = sitk.LabelShapeStatisticsImageFilter()
        stats.Execute(cc_g)
        labels = stats.GetLabels()
        
        if not labels: return 1.0, 1.0, 1.0, 1.0

        # 计算病灶体积中位数用于分层 
        sizes = [stats.GetPhysicalSize(l) for l in labels]
        median_v = np.median(sizes)
        
        # 命中判断
        hit_map = sitk.GetArrayFromImage(sitk.Multiply(cc_g, sitk.Cast(p_img, sitk.sitkUInt32)))
        hits = np.unique(hit_map)
        
        s_tot, s_hit = 0, 0
        l_tot, l_hit = 0, 0
        for label, size in zip(labels, sizes):
            is_hit = 1 if label in hits else 0
            if size <= median_v:
                s_tot += 1; s_hit += is_hit
            else:
                l_tot += 1; l_hit += is_hit
        
        recall = (s_hit + l_hit) / (s_tot + l_tot)
        r_small = s_hit / s_tot if s_tot > 0 else 1.0
        r_large = l_hit / l_tot if l_tot > 0 else 1.0
        
        # F1 计算 (需要反向计算 Precision)
        cc_p = self.cc_filter.Execute(p_img)
        p_hit_map = sitk.GetArrayFromImage(sitk.Multiply(cc_p, sitk.Cast(g_img, sitk.sitkUInt32)))
        precision = (len(np.unique(p_hit_map)) - 1) / (len(np.unique(sitk.GetArrayFromImage(cc_p))) - 1 + 1e-6)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
        
        return recall, f1, r_small, r_large

# ==========================================
# 2. 自助法评估主程序
# ==========================================
def run_official_evaluation(model_path, test_loader, device):
    # 加载模型
    #model = Optimized_DynamicModal_Net(n_classes=3, base_c=32, num_modalities=2).to(device)
    # model = MedNeXt(
    #         in_channels = 2, 
    #         n_channels = 32,
    #         n_classes = 3, 
    #         exp_r=2                 ,         # Expansion ratio as in Swin Transformers
    #         kernel_size=3,                     # Can test kernel_size
    #         deep_supervision=False,             # Can be used to test deep supervision
    #         do_res=False,                      # Can be used to individually test residual connection
    #         do_res_up_down = True,
    #         block_counts = [2,2,2,2,2,2,2,2,2]
    #         #dim='2d'
    #     ).to(device)
    #model = UNet3D(in_channels=2,n_classes=3, base_c=32).cuda()
    # model = SwinUNETR(
    #             spatial_dims=3,      # 3D
    #             in_channels=2,
    #             out_channels=3,
    #             feature_size=48,
    #         ).cuda()
    model = Ablation_NEncoder_Final_Net(
        n_classes=3, 
        num_modalities=2, 
        base_c=32, 
        deep_sup=False
    ).to(device)
    # 2. 安全加载权重 (针对 PyTorch 2.6+ 修复)
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    except TypeError:
        # 兼容旧版本 PyTorch (不支持 weights_only 参数)
        checkpoint = torch.load(model_path, map_location=device)
    # 3. 提取权重状态字典
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    
    calc = OfficialMetricCalculator()
    all_subject_results = []

    # 第一步：遍历测试集获取原始数据
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Inference"):
            inputs = torch.cat([batch['flair'], batch['t1']], dim=1).to(device).float()
            targets = batch['seg'].to(device)
            spacing = batch['spacing'][0].cpu().numpy()
            
            logits = sliding_window_inference(inputs, (128, 128, 64), 4, model, overlap=0.5)
            preds = torch.argmax(logits, dim=1, keepdim=True)
            
            # 官方 Mask Label 2 逻辑 [cite: 120]
            pred_wmh = (preds == 1).cpu().numpy()[0, 0].astype(np.uint8) # 增加 .astype(np.uint8)
            target_wmh = (targets == 1).cpu().numpy()[0, 0].astype(np.uint8) # 增加 .astype(np.uint8)
            target_all = targets.cpu().numpy()[0, 0]
            pred_wmh[target_all == 2] = 0 
            
            # 计算该病例指标
            metrics = calc.calculate_subject_metrics(pred_wmh, target_wmh, spacing)
            all_subject_results.append(metrics)

    # 第二步：自助法（Bootstrapping）计算置信区间 [cite: 227, 228]
    n_bootstrap = 2000
    metrics_keys = ["dice", "h95", "lavd", "recall", "f1", "r_small", "r_large"]
    final_report = {}

    print(f"\nRunning {n_bootstrap} Bootstraps...")
    for key in metrics_keys:
        scores = [res[key] for res in all_subject_results]
        boot_means = []
        for _ in range(n_bootstrap):
            resample = np.random.choice(scores, size=len(scores), replace=True) 
            boot_means.append(np.mean(resample))
        
        final_report[key] = {
            "mean": np.mean(scores),
            "ci_low": np.percentile(boot_means, 2.5),
            "ci_high": np.percentile(boot_means, 97.5)
        }

    # 第三步：打印对齐官方 Table II 格式的报告 [cite: 306, 312]
    print("\n" + "="*60)
    print(f"{'Metric':<15} | {'Mean':<10} | {'95% Confidence Interval':<25}")
    print("-" * 60)
    for k, v in final_report.items():
        print(f"{k:<15} | {v['mean']:.4f}     | ({v['ci_low']:.4f} - {v['ci_high']:.4f})")
    print("="*60)

if __name__ == '__main__':
    # 路径配置
    TEST_MODEL = '/home/dell/hxy/logs_experiment_UNET/best_model.pth'
    ROOT_PATH = '/data4T/WHM'
    
    # 加载验证集 (Batch Size 必须为 1 以处理不同尺寸)
    val_tf = Compose([RobustIntensityNormalize()])
    test_ds = WMHDataset(ROOT_PATH, split='test', transform=val_tf)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=2)
    
    run_official_evaluation(TEST_MODEL, test_loader, torch.device("cuda"))