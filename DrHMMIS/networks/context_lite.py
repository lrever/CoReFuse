import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.insert(0, '/home/dell/hxy/SSL4MIS/project/workspace') 
print(f"系统路径已更新: {sys.path}")
from networks.gate_cnn_transformer import TransformerBlock, Light_CrossAttention_Block_Symmetric
from networks.gate_cnn_transformer import ThreeEncoderNaiveFusionUNet,ThreeEncoderNaiveFusionUNet1
from networks.context import ThreeEncoder_SelectiveAxial_Net,ThreeEncoder_Calibration_Net,CrossModalCalibrationFusion,SelectiveAxialEncoder3D
import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.gate_cnn_transformer import Light_CrossAttention_Block_Symmetric

import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.gate_cnn_transformer import TransformerBlock, Light_CrossAttention_Block_Symmetric
from networks.classic_networks import UNet3D,ResUNet3D
# ===============================================================================
#   基础工具
# ===============================================================================
class NaiveFusion(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv3d(in_channels*3, in_channels, kernel_size=3, padding=1, bias=False), nn.BatchNorm3d(in_channels), nn.ReLU(True))
    def forward(self, x1, x2, x3): return self.conv(torch.cat([x1, x2, x3], dim=1)), {}
class DSConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
        super().__init__()
        self.dw_conv = nn.Conv3d(in_channels, in_channels, kernel_size=kernel_size, 
                                 stride=stride, padding=padding, groups=in_channels, bias=bias)
        self.pw_conv = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias)

    def forward(self, x):
        return self.pw_conv(self.dw_conv(x))

class Up(nn.Module):
    """ 原始 Up: 用于最后一层高精度恢复 """
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels), nn.ReLU(True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels), nn.ReLU(True)
        )
    def forward(self, x, skip):
        x = self.up(x)
        diffZ = skip.size()[2] - x.size()[2]; diffY = skip.size()[3] - x.size()[3]; diffX = skip.size()[4] - x.size()[4]
        x = F.pad(x, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2, diffZ // 2, diffZ - diffZ // 2])
        return self.conv(torch.cat([x, skip], dim=1))

class DSConv_Up(nn.Module):
    """ DSConv Up: 用于深层，节省算力 """
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.conv = nn.Sequential(
            DSConv3d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels), nn.ReLU(True),
            DSConv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels), nn.ReLU(True)
        )
    def forward(self, x, skip):
        x = self.up(x)
        diffZ = skip.size()[2] - x.size()[2]; diffY = skip.size()[3] - x.size()[3]; diffX = skip.size()[4] - x.size()[4]
        x = F.pad(x, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2, diffZ // 2, diffZ - diffZ // 2])
        return self.conv(torch.cat([x, skip], dim=1))

# ===============================================================================
#   Encoder: SelectiveAxialEncoder3D_Lite
# ===============================================================================





class MixedScaleContextAggregation(nn.Module):
    """
    [Generalization Module]
    同时集成 Strip Pooling (针对细长/管状结构) 和 Global Pooling (针对团块/语义结构)。
    通过 Channel Attention 动态选择上下文类型。
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # 内部降维以减少计算量
        mid_channels = out_channels // 2
        
        # ------------------------------------------------------------------
        # Path 1: Strip Pooling (捕捉各向异性长程依赖 - EPVS/Vessels)
        # ------------------------------------------------------------------
        self.pool_d = nn.AdaptiveAvgPool3d((None, 1, 1))
        self.pool_h = nn.AdaptiveAvgPool3d((1, None, 1))
        self.pool_w = nn.AdaptiveAvgPool3d((1, 1, None))
        
        self.conv_d = nn.Conv3d(in_channels, mid_channels, 3, padding=1, bias=False)
        self.conv_h = nn.Conv3d(in_channels, mid_channels, 3, padding=1, bias=False)
        self.conv_w = nn.Conv3d(in_channels, mid_channels, 3, padding=1, bias=False)
        
        # ------------------------------------------------------------------
        # Path 2: Global Pooling (捕捉各向同性全局语义 - Tumor/Organs)
        # ------------------------------------------------------------------
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.conv_global = nn.Conv3d(in_channels, mid_channels, 1, bias=False)
        
        # ------------------------------------------------------------------
        # Feature Aggregation & Attention
        # ------------------------------------------------------------------
        # 输入通道 = (Strip_D + Strip_H + Strip_W + Global) = 4 * mid
        self.fusion_conv = nn.Sequential(
            nn.Conv3d(mid_channels * 4, out_channels, 1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(True)
        )
        
        # Gating Mechanism: 决定上下文特征的注入强度
        self.gamma_gate = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(out_channels, out_channels // 4, 1),
            nn.ReLU(True),
            nn.Conv3d(out_channels // 4, out_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, d, h, w = x.shape
        
        # 1. Strip Path Forward
        # 沿 D/H/W 轴池化 -> 卷积提取特征 -> 插值回原尺寸
        p_d = F.interpolate(self.conv_d(self.pool_d(x)), size=(d,h,w), mode='trilinear', align_corners=False)
        p_h = F.interpolate(self.conv_h(self.pool_h(x)), size=(d,h,w), mode='trilinear', align_corners=False)
        p_w = F.interpolate(self.conv_w(self.pool_w(x)), size=(d,h,w), mode='trilinear', align_corners=False)
        
        # 2. Global Path Forward
        p_g = F.interpolate(self.conv_global(self.global_pool(x)), size=(d,h,w), mode='trilinear', align_corners=False)
        
        # 3. Aggregation
        context = self.fusion_conv(torch.cat([p_d, p_h, p_w, p_g], dim=1))
        
        # 4. Adaptive Selection (Residual Injection)
        # 类似于 SE-Block 的操作，但这决定的是"上下文特征"的权重
        gate = self.gamma_gate(context)
        
        return x + context * gate
    



class SelectiveFusion(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super().__init__()
        mid_channels = max(in_channels // reduction, 16)
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, mid_channels, bias=False), nn.ReLU(True),
            nn.Linear(mid_channels, in_channels * 2, bias=False),
        )
    def forward(self, x_local, x_axial):
        b, c, d, h, w = x_local.size()
        s = self.avg_pool(x_local + x_axial).view(b, c)
        attn = F.softmax(self.fc(s).view(b, 2, c), dim=1)
        return x_local * attn[:, 0, :].view(b, c, 1, 1, 1) + x_axial * attn[:, 1, :].view(b, c, 1, 1, 1)

class AxialTubularExcitation(nn.Module):
    def __init__(self, dim, kernel_size=7):
        super().__init__()
        pad = (kernel_size - 1) // 2
        self.conv_d = nn.Conv3d(dim, dim, kernel_size=(kernel_size, 1, 1), padding=(pad, 0, 0), groups=dim)
        self.conv_h = nn.Conv3d(dim, dim, kernel_size=(1, kernel_size, 1), padding=(0, pad, 0), groups=dim)
        self.conv_w = nn.Conv3d(dim, dim, kernel_size=(1, 1, kernel_size), padding=(0, 0, pad), groups=dim)
        self.act = nn.GELU()
        self.fusion = nn.Conv3d(dim * 3, dim, kernel_size=1)
    def forward(self, x):
        d = self.conv_d(x); h = self.conv_h(x); w = self.conv_w(x)
        return self.act(self.fusion(torch.cat([d, h, w], dim=1)))

class SelectiveAxialResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels); self.relu = nn.ReLU(True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.axial_branch = AxialTubularExcitation(out_channels, kernel_size=7)
        self.bn_axial = nn.BatchNorm3d(out_channels)
        self.selection = SelectiveFusion(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(nn.Conv3d(in_channels, out_channels, 1, stride=stride, bias=False), nn.BatchNorm3d(out_channels))
    def forward(self, x):
        res = self.shortcut(x)
        x = self.relu(self.bn1(self.conv1(x)))
        feat_local = self.bn2(self.conv2(x))
        feat_axial = self.bn_axial(self.axial_branch(x))
        return self.relu(self.selection(feat_local, feat_axial) + res)

class SelectiveAxialEncoder3D_Lite(nn.Module):
    def __init__(self, in_ch, channels):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv3d(in_ch, channels[0] // 2, 3, 1, 1, bias=False),
            nn.BatchNorm3d(channels[0] // 2), nn.ReLU(True),
            DSConv3d(channels[0] // 2, channels[0] // 2, 3, 1, 1, bias=False),
            nn.BatchNorm3d(channels[0] // 2), nn.ReLU(True),
            DSConv3d(channels[0] // 2, channels[0], 3, 1, 1, bias=False),
            nn.BatchNorm3d(channels[0]), nn.ReLU(True)
        )
        self.layer1 = SelectiveAxialResBlock(channels[0], channels[0], stride=1)
        self.layer2 = SelectiveAxialResBlock(channels[0], channels[1], stride=2)
        self.layer3 = SelectiveAxialResBlock(channels[1], channels[2], stride=2)
        self.layer4 = SelectiveAxialResBlock(channels[2], channels[3], stride=2)
    def forward(self, x):
        x0 = self.stem(x)
        x1 = self.layer1(x0); x2 = self.layer2(x1); x3 = self.layer3(x2); x4 = self.layer4(x3)
        return [x1, x2, x3, x4]

# ===============================================================================
#   Fusion: Concat 逻辑 (Balacned)
# ===============================================================================
class FrequencyAwareFusion_Lite(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.low_pass = nn.AvgPool3d(3, 1, 1)
        
        # 优化1: Group Conv 提取权重
        self.competitor = nn.Sequential(
            nn.Conv3d(in_channels*3, in_channels*3, 3, padding=1, groups=3, bias=False), 
            nn.ReLU(True), 
            nn.Conv3d(in_channels*3, 3, 1, bias=True), 
            nn.Softmax(dim=1)
        )
        # 优化2: 共享门控 DSConv
        self.shared_gate = nn.Sequential(
            DSConv3d(in_channels*3, in_channels//2, 3, 1, 1, bias=False),
            nn.ReLU(True), nn.Conv3d(in_channels//2, 3, 1, bias=True), nn.Sigmoid()
        )
        # 优化3: 最终融合使用 Concat + DSConv (修复了 Additive 导致的性能下降)
        self.final = nn.Sequential(
            DSConv3d(in_channels*2, in_channels, 3, 1, 1, bias=False), 
            nn.BatchNorm3d(in_channels), nn.ReLU(True),
            DSConv3d(in_channels, in_channels, 3, 1, 1, bias=False), 
            nn.BatchNorm3d(in_channels), nn.ReLU(True)
        )

    def forward(self, x_t1, x_t2, x_flair):
        l_t1 = self.low_pass(x_t1); h_t1 = x_t1 - l_t1
        l_t2 = self.low_pass(x_t2); h_t2 = x_t2 - l_t2
        l_flair = self.low_pass(x_flair); h_flair = x_flair - l_flair
        
        # Low Freq
        low_cat = torch.cat([l_t1, l_t2, l_flair], dim=1)
        low_w = self.competitor(low_cat)
        w1, w2, w3 = torch.split(low_w, 1, dim=1)
        l_fused = l_t1*w1 + l_t2*w2 + l_flair*w3
        
        # High Freq
        high_cat = torch.cat([h_t1, h_t2, h_flair], dim=1)
        gates = self.shared_gate(high_cat)
        g1, g2, g3 = torch.split(gates, 1, dim=1)
        h_fused = h_t1*g1 + h_t2*g2 + h_flair*g3
        
        # Concat Fusion
        return self.final(torch.cat([l_fused, h_fused], dim=1)), {}

class CrossModalCalibrationFusion_Lite(nn.Module):
    def __init__(self, in_channels, num_heads=4, reduction_ratio=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.calibration_fc = nn.Sequential(
            nn.Conv3d(in_channels*3, (in_channels*3)//reduction_ratio, 1, bias=False), nn.ReLU(True),
            nn.Conv3d((in_channels*3)//reduction_ratio, in_channels*3, 1, bias=False), nn.Sigmoid()
        )
        self.spatial_gate_shared = nn.Sequential(
            nn.Conv3d(in_channels*3, (in_channels*3)//4, 1, groups=3, bias=False), nn.ReLU(True),
            nn.Conv3d((in_channels*3)//4, 3, 1, groups=3, bias=True), nn.Sigmoid()
        )
        self.query_gen = nn.Sequential(nn.Conv3d(in_channels*3, in_channels, 1, bias=False), nn.BatchNorm3d(in_channels), nn.ReLU(True))
        self.attention_block = Light_CrossAttention_Block_Symmetric(in_channels=in_channels, num_heads=num_heads, reduction_ratio=reduction_ratio)

    def forward(self, t1, t2, flair):
        cat_feat = torch.cat([t1, t2, flair], dim=1)
        weights = self.calibration_fc(self.avg_pool(cat_feat))
        cat_calibrated = cat_feat * weights
        
        masks = self.spatial_gate_shared(cat_calibrated)
        c = t1.size(1)
        t1 = cat_calibrated[:, :c] * masks[:, 0:1]
        t2 = cat_calibrated[:, c:2*c] * masks[:, 1:2]
        flair = cat_calibrated[:, 2*c:] * masks[:, 2:3]
        all_feat = torch.cat([t1, t2, flair], dim=1)
        return self.attention_block(x_q=self.query_gen(all_feat), x_kv_concat=all_feat), {}
class SelectiveAxialEncoder3D_Step1(nn.Module):
    def __init__(self, in_ch, channels):
        super().__init__()
        # [优化点]: Hybrid DS-Stem
        self.stem = nn.Sequential(
            # 第一层保持普通卷积 (Raw Input -> Feature，保住最原始信息)
            nn.Conv3d(in_ch, channels[0] // 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(channels[0] // 2), nn.ReLU(True),
            
            # 后续两层使用 DSConv (在特征空间增加深度，但降算力)
            DSConv3d(channels[0] // 2, channels[0] // 2, kernel_size=3, padding=1),
            nn.BatchNorm3d(channels[0] // 2), nn.ReLU(True),
            
            DSConv3d(channels[0] // 2, channels[0], kernel_size=3, padding=1),
            nn.BatchNorm3d(channels[0]), nn.ReLU(True)
        )
        self.layer1 = SelectiveAxialResBlock(channels[0], channels[0], stride=1)
        self.layer2 = SelectiveAxialResBlock(channels[0], channels[1], stride=2)
        self.layer3 = SelectiveAxialResBlock(channels[1], channels[2], stride=2)
        self.layer4 = SelectiveAxialResBlock(channels[2], channels[3], stride=2)

    def forward(self, x):
        x0 = self.stem(x)
        x1 = self.layer1(x0); x2 = self.layer2(x1); x3 = self.layer3(x2); x4 = self.layer4(x3)
        return [x1, x2, x3, x4]
# ===============================================================================
#   Bottleneck
# ===============================================================================
class StripPooling3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        mid_channels = out_channels // 2
        self.pool_d = nn.AdaptiveAvgPool3d((None, 1, 1))
        self.pool_h = nn.AdaptiveAvgPool3d((1, None, 1))
        self.pool_w = nn.AdaptiveAvgPool3d((1, 1, None))
        self.conv_d = nn.Conv3d(in_channels, mid_channels, 3, padding=1, bias=False)
        self.conv_h = nn.Conv3d(in_channels, mid_channels, 3, padding=1, bias=False)
        self.conv_w = nn.Conv3d(in_channels, mid_channels, 3, padding=1, bias=False)
        self.fusion = nn.Sequential(nn.Conv3d(mid_channels * 3, out_channels, 1, bias=False), nn.BatchNorm3d(out_channels), nn.ReLU(True))
    def forward(self, x):
        d,h,w = x.shape[2:]
        p_d = F.interpolate(self.conv_d(self.pool_d(x)), size=(d,h,w), mode='trilinear')
        p_h = F.interpolate(self.conv_h(self.pool_h(x)), size=(d,h,w), mode='trilinear')
        p_w = F.interpolate(self.conv_w(self.pool_w(x)), size=(d,h,w), mode='trilinear')
        return x + self.fusion(torch.cat([p_d, p_h, p_w], dim=1))


class StripPooling3D_1(nn.Module):
    """
    [最终优化版] 3D 条纹池化 (Asymmetric Convolution + Attention)
    
    改进：
    1. 使用非对称卷积 (e.g., 3x1x1) 替代 1x1x1 或 3x3x3。
       优势：既保留了沿轴向的上下文感知 (比1x1强)，又大幅减少了参数量 (比3x3少9倍)。
    2. 保持 Attention 机制。
    """
    def __init__(self, in_channels, out_channels=None):
        super().__init__()
        if out_channels is None:
            out_channels = in_channels
            
        # 通道调整
        self.channel_adjust = nn.Identity()
        if in_channels != out_channels:
            self.channel_adjust = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm3d(out_channels),
                nn.ReLU(True)
            )
        
        mid_channels = out_channels // 2
        
        # 1. 池化
        self.pool_d = nn.AdaptiveAvgPool3d((None, 1, 1))
        self.pool_h = nn.AdaptiveAvgPool3d((1, None, 1))
        self.pool_w = nn.AdaptiveAvgPool3d((1, 1, None))
        
        # 2. 非对称卷积 (关键修改)
        # 专门提取沿轴向的局部上下文，防止信息断裂
        self.conv_d = nn.Conv3d(out_channels, mid_channels, kernel_size=(3, 1, 1), padding=(1, 0, 0), bias=False)
        self.conv_h = nn.Conv3d(out_channels, mid_channels, kernel_size=(1, 3, 1), padding=(0, 1, 0), bias=False)
        self.conv_w = nn.Conv3d(out_channels, mid_channels, kernel_size=(1, 1, 3), padding=(0, 0, 1), bias=False)
        
        # 3. 融合
        self.fusion = nn.Sequential(
            nn.Conv3d(mid_channels * 3, out_channels, 1, bias=False),
            nn.Sigmoid() 
        )

    def forward(self, x):
        x = self.channel_adjust(x)
        d, h, w = x.shape[2:]
        
        # 1. Strip Pooling + Asymmetric Conv
        p_d = self.conv_d(self.pool_d(x))
        p_h = self.conv_h(self.pool_h(x))
        p_w = self.conv_w(self.pool_w(x))
        
        # 2. Upsample
        p_d = F.interpolate(p_d, size=(d,h,w), mode='trilinear', align_corners=False)
        p_h = F.interpolate(p_h, size=(d,h,w), mode='trilinear', align_corners=False)
        p_w = F.interpolate(p_w, size=(d,h,w), mode='trilinear', align_corners=False)
        
        # 3. Attention
        attention_map = self.fusion(torch.cat([p_d, p_h, p_w], dim=1))
        
        return x + x * attention_map
        


        
class FrequencyAwareFusion_Lite_Concat(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.low_pass = nn.AvgPool3d(3, 1, 1)
        
        # 优化1: Group Conv 提取权重 (Input 3C -> Output 3)
        # T1, T2, FLAIR 先独立计算，最后 1x1 融合
        self.competitor = nn.Sequential(
            nn.Conv3d(in_channels*3, in_channels*3, 3, padding=1, groups=3, bias=False), 
            nn.ReLU(True), 
            nn.Conv3d(in_channels*3, 3, 1, bias=True), 
            nn.Softmax(dim=1)
        )
        # 优化2: 共享门控 DSConv (处理 High Freq)
        self.shared_gate = nn.Sequential(
            DSConv3d(in_channels*3, in_channels//2, 3, 1, 1, bias=False),
            nn.ReLU(True), nn.Conv3d(in_channels//2, 3, 1, bias=True), nn.Sigmoid()
        )
        # 优化3: 最终融合使用 DSConv
        # [关键]: 输入是 2C (low + high)，我们用 DSConv 来消化这个宽输入
        self.final = nn.Sequential(
            DSConv3d(in_channels*2, in_channels, 3, 1, 1, bias=False), 
            nn.BatchNorm3d(in_channels), nn.ReLU(True),
            DSConv3d(in_channels, in_channels, 3, 1, 1, bias=False), 
            nn.BatchNorm3d(in_channels), nn.ReLU(True)
        )

    def forward(self, x_t1, x_t2, x_flair):
        l_t1 = self.low_pass(x_t1); h_t1 = x_t1 - l_t1
        l_t2 = self.low_pass(x_t2); h_t2 = x_t2 - l_t2
        l_flair = self.low_pass(x_flair); h_flair = x_flair - l_flair
        
        # Low Freq
        low_cat = torch.cat([l_t1, l_t2, l_flair], dim=1)
        low_w = self.competitor(low_cat)
        w1, w2, w3 = torch.split(low_w, 1, dim=1)
        l_fused = l_t1*w1 + l_t2*w2 + l_flair*w3
        
        # High Freq
        high_cat = torch.cat([h_t1, h_t2, h_flair], dim=1)
        gates = self.shared_gate(high_cat)
        g1, g2, g3 = torch.split(gates, 1, dim=1)
        h_fused = h_t1*g1 + h_t2*g2 + h_flair*g3
        
        # Concat Fusion (确保信息无损)
        return self.final(torch.cat([l_fused, h_fused], dim=1)), {}


class FrequencyAwareFusion_UltraLite(nn.Module):
    """
    [极致轻量化融合]
    针对 FLOPs 分析结果进行定点清除：
    1. Competitor: Group Conv (Std) -> Group DSConv
    2. Final: 2层 DSConv -> 1层 DSConv
    """
    def __init__(self, in_channels):
        super().__init__()
        self.low_pass = nn.AvgPool3d(3, 1, 1)
        
        # [优化点 1] Competitor: 从标准 Group Conv 改为 DSConv
        # 即使是生成权重，也不需要全通道的空间卷积
        # 结构: Depthwise (提取空间) -> Pointwise (通道混合) -> 1x1 (降维到3)
        self.competitor = nn.Sequential(
            # Depthwise: 96 -> 96 (计算量极低)
            nn.Conv3d(in_channels*3, in_channels*3, kernel_size=3, padding=1, groups=in_channels*3, bias=False),
            # Pointwise: 96 -> 96 (groups=3, 保持模态独立)
            nn.Conv3d(in_channels*3, in_channels*3, kernel_size=1, groups=3, bias=False),
            nn.ReLU(True),
            # Reduce: 96 -> 3
            nn.Conv3d(in_channels*3, 3, kernel_size=1, bias=True),
            nn.Softmax(dim=1)
        )
        
        # [优化点 2] Shared Gate: 保持 DSConv (已经很省了)
        self.shared_gate = nn.Sequential(
            DSConv3d(in_channels*3, in_channels//2, 3, 1, 1, bias=False),
            nn.ReLU(True),
            nn.Conv3d(in_channels//2, 3, 1, bias=True),
            nn.Sigmoid()
        )
        
        # [优化点 3] Final: 砍掉一层，只保留一层 DSConv
        # 输入 2C -> 输出 C
        self.final = nn.Sequential(
            DSConv3d(in_channels*2, in_channels, 3, 1, 1, bias=False),
            nn.BatchNorm3d(in_channels), 
            nn.ReLU(True)
        )

    def forward(self, x_t1, x_t2, x_flair):
        l_t1 = self.low_pass(x_t1); h_t1 = x_t1 - l_t1
        l_t2 = self.low_pass(x_t2); h_t2 = x_t2 - l_t2
        l_flair = self.low_pass(x_flair); h_flair = x_flair - l_flair
        
        # Low Freq
        low_cat = torch.cat([l_t1, l_t2, l_flair], dim=1)
        low_w = self.competitor(low_cat)
        w1, w2, w3 = torch.split(low_w, 1, dim=1)
        l_fused = l_t1*w1 + l_t2*w2 + l_flair*w3
        
        # High Freq
        high_cat = torch.cat([h_t1, h_t2, h_flair], dim=1)
        gates = self.shared_gate(high_cat)
        g1, g2, g3 = torch.split(gates, 1, dim=1)
        h_fused = h_t1*g1 + h_t2*g2 + h_flair*g3
        
        # Concat Fusion
        return self.final(torch.cat([l_fused, h_fused], dim=1)), {}
# ===============================================================================
#   主网络: ThreeEncoder_Balanced_NoDS_Net
# ===============================================================================
class ThreeEncoder_Step1_L1L2_Lite_Net(nn.Module):
    def __init__(self, n_classes, base_c=32, n_levels=4):
        super().__init__()
        modal_channels = [base_c * (2**i) for i in range(n_levels)]
        
        # 1. Encoder (优化 Stem)
        self.encoder_t1 = SelectiveAxialEncoder3D_Step1(in_ch=1, channels=modal_channels)
        self.encoder_t2 = SelectiveAxialEncoder3D_Step1(in_ch=1, channels=modal_channels)
        self.encoder_flair = SelectiveAxialEncoder3D_Step1(in_ch=1, channels=modal_channels)

        # 2. Fusion (L1/L2 使用 Lite_Concat, L3/L4 使用原版)
        self.fuse1 = FrequencyAwareFusion_Lite_Concat(modal_channels[0]) 
        self.fuse2 = FrequencyAwareFusion_Lite_Concat(modal_channels[1]) 
        self.fuse3 = CrossModalCalibrationFusion(modal_channels[2])
        self.fuse4 = CrossModalCalibrationFusion(modal_channels[3])
        
        # 3. Bottleneck (StripPooling)
        self.bottleneck = StripPooling3D(modal_channels[3], modal_channels[3])
        
        # 4. Decoder (原始 Up)
        self.decoders = nn.ModuleList()
        # L3 -> L2
        self.decoders.append(Up(modal_channels[3], modal_channels[2], modal_channels[2]))
        # L2 -> L1
        self.decoders.append(Up(modal_channels[2], modal_channels[1], modal_channels[1]))
        # L1 -> L0
        self.decoders.append(Up(modal_channels[1], modal_channels[0], modal_channels[0]))

        # Output Heads (无 Deep Supervision，仅输出最终结果)
        self.outc = nn.Conv3d(modal_channels[0], n_classes, 1)

    def forward(self, x_t1, x_t2, x_flair):
        t1 = self.encoder_t1(x_t1); t2 = self.encoder_t2(x_t2); flair = self.encoder_flair(x_flair)

        fused_skips = []
        # L1 Fusion (Lite)
        x1, _ = self.fuse1(t1[0], t2[0], flair[0]); fused_skips.append(x1)
        # L2 Fusion (Lite)
        x2, _ = self.fuse2(t1[1], t2[1], flair[1]); fused_skips.append(x2)
        # L3 Fusion (Original)
        x3, _ = self.fuse3(t1[2], t2[2], flair[2]); fused_skips.append(x3)
        # L4 Fusion (Original)
        x4, _ = self.fuse4(t1[3], t2[3], flair[3])
        
        x = self.bottleneck(x4)
        
        skip3 = fused_skips.pop(); x = self.decoders[0](x, skip3)
        skip2 = fused_skips.pop(); x = self.decoders[1](x, skip2)
        skip1 = fused_skips.pop(); x = self.decoders[2](x, skip1)
        
        return self.outc(x)
    
class ThreeEncoder_Step1_FusionLite_NoDS_Net(nn.Module):
    def __init__(self, n_classes, base_c=32, n_levels=4):
        super().__init__()
        modal_channels = [base_c * (2**i) for i in range(n_levels)]
        
        # 1. Encoder (Heavy - SOTA)
        self.encoder_t1 = SelectiveAxialEncoder3D(in_ch=1, channels=modal_channels)
        self.encoder_t2 = SelectiveAxialEncoder3D(in_ch=1, channels=modal_channels)
        self.encoder_flair = SelectiveAxialEncoder3D(in_ch=1, channels=modal_channels)

        # 2. Fusion (Modified: L1/L2 use Lite version)
        self.fuse1 = FrequencyAwareFusion_UltraLite(modal_channels[0]) # Lite
        self.fuse2 = FrequencyAwareFusion_UltraLite(modal_channels[1]) # Lite
        
        # L3/L4 保持原版 (深层)
        self.fuse3 = CrossModalCalibrationFusion(modal_channels[2])
        self.fuse4 = CrossModalCalibrationFusion(modal_channels[3])
        
        # 3. Bottleneck (Original)
        self.transformer_block = StripPooling3D(modal_channels[3], modal_channels[3])
        
        # 4. Decoder (Original Up)
        self.decoders = nn.ModuleList()
        self.decoders.append(Up(modal_channels[3], modal_channels[2], modal_channels[2]))
        self.decoders.append(Up(modal_channels[2], modal_channels[1], modal_channels[1]))
        self.decoders.append(Up(modal_channels[1], modal_channels[0], modal_channels[0]))

        # Output Heads (无 Deep Supervision，单输出)
        self.outc = nn.Conv3d(modal_channels[0], n_classes, 1)

    def forward(self, x_t1, x_t2, x_flair):
        t1 = self.encoder_t1(x_t1); t2 = self.encoder_t2(x_t2); flair = self.encoder_flair(x_flair)

        fused_skips = []
        x1, _ = self.fuse1(t1[0], t2[0], flair[0]); fused_skips.append(x1)
        x2, _ = self.fuse2(t1[1], t2[1], flair[1]); fused_skips.append(x2)
        x3, _ = self.fuse3(t1[2], t2[2], flair[2]); fused_skips.append(x3)
        x4, _ = self.fuse4(t1[3], t2[3], flair[3])
        
        x = self.transformer_block(x4)
        
        skip3 = fused_skips.pop(); x = self.decoders[0](x, skip3)
        skip2 = fused_skips.pop(); x = self.decoders[1](x, skip2)
        skip1 = fused_skips.pop(); x = self.decoders[2](x, skip1)
        
        return self.outc(x)

class CoordinateAttention3D(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.pool_d = nn.AdaptiveAvgPool3d((None, 1, 1))
        self.pool_h = nn.AdaptiveAvgPool3d((1, None, 1))
        self.pool_w = nn.AdaptiveAvgPool3d((1, 1, None))
        
        mip = max(8, in_channels // reduction)
        self.conv1 = nn.Conv3d(in_channels, mip, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(mip)
        self.act = nn.ReLU(True)
        
        self.conv_d = nn.Conv3d(mip, in_channels, kernel_size=1, bias=False)
        self.conv_h = nn.Conv3d(mip, in_channels, kernel_size=1, bias=False)
        self.conv_w = nn.Conv3d(mip, in_channels, kernel_size=1, bias=False)

    def forward(self, x):
        identity = x
        n, c, d, h, w = x.size()
        x_d = self.pool_d(x)
        x_h = self.pool_h(x)
        x_w = self.pool_w(x)
        
        # Shared transformation
        y_d = self.act(self.bn1(self.conv1(x_d)))
        y_h = self.act(self.bn1(self.conv1(x_h)))
        y_w = self.act(self.bn1(self.conv1(x_w)))
        
        a_d = torch.sigmoid(self.conv_d(y_d))
        a_h = torch.sigmoid(self.conv_h(y_h))
        a_w = torch.sigmoid(self.conv_w(y_w))
        
        return identity * a_d * a_h * a_w

# ===============================================================================
#   [Decoder] CoordAtt_Up
#   替代原始 Up，加入去噪能力
# ===============================================================================
class CoordAtt_Up(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels), nn.ReLU(True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels), nn.ReLU(True)
        )
        self.ca = CoordinateAttention3D(out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        diffZ = skip.size()[2] - x.size()[2]; diffY = skip.size()[3] - x.size()[3]; diffX = skip.size()[4] - x.size()[4]
        x = F.pad(x, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2, diffZ // 2, diffZ - diffZ // 2])
        
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        x = self.ca(x) # [关键] 去噪
        return x

# ===============================================================================
#   主网络: ThreeEncoder_Step2_Coord_Net
# ===============================================================================
# class FrequencyAwareFusion_UltraLite(nn.Module):
#     def __init__(self, in_channels):
#         super().__init__()
#         self.low_pass = nn.AvgPool3d(3, 1, 1)
        
#         # [UltraLite] Competitor: 完全 Depthwise
#         # 1. Spatial (DW): 96->96, groups=96. 极低计算量
#         # 2. Channel (PW): 96->3.
#         self.competitor = nn.Sequential(
#             nn.Conv3d(in_channels*3, in_channels*3, kernel_size=3, padding=1, groups=in_channels*3, bias=False),
#             nn.Conv3d(in_channels*3, 3, kernel_size=1, bias=True),
#             nn.Softmax(dim=1)
#         )
        
#         # [UltraLite] Shared Gate: DSConv
#         self.shared_gate = nn.Sequential(
#             DSConv3d(in_channels*3, in_channels//2, 3, 1, 1, bias=False),
#             nn.ReLU(True),
#             nn.Conv3d(in_channels//2, 3, 1, bias=True),
#             nn.Sigmoid()
#         )
        
#         # [UltraLite] Final: 1层 DSConv
#         self.final = nn.Sequential(
#             DSConv3d(in_channels*2, in_channels, 3, 1, 1, bias=False),
#             nn.BatchNorm3d(in_channels), 
#             nn.ReLU(True)
#         )

#     def forward(self, x_t1, x_t2, x_flair):
#         l_t1 = self.low_pass(x_t1); h_t1 = x_t1 - l_t1
#         l_t2 = self.low_pass(x_t2); h_t2 = x_t2 - l_t2
#         l_flair = self.low_pass(x_flair); h_flair = x_flair - l_flair
        
#         # Low Freq
#         low_cat = torch.cat([l_t1, l_t2, l_flair], dim=1)
#         low_w = self.competitor(low_cat)
#         w1, w2, w3 = torch.split(low_w, 1, dim=1)
#         l_fused = l_t1*w1 + l_t2*w2 + l_flair*w3
        
#         # High Freq
#         high_cat = torch.cat([h_t1, h_t2, h_flair], dim=1)
#         gates = self.shared_gate(high_cat)
#         g1, g2, g3 = torch.split(gates, 1, dim=1)
#         h_fused = h_t1*g1 + h_t2*g2 + h_flair*g3
        
#         # Concat Fusion
#         return self.final(torch.cat([l_fused, h_fused], dim=1)), {}

class ThreeEncoder_Final_SOTA_Net(nn.Module):
    def __init__(self, n_classes, base_c=32, n_levels=4):
        super().__init__()
        modal_channels = [base_c * (2**i) for i in range(n_levels)]
        
        # 1. Encoder: Heavy (SOTA)
        self.encoder_t1 = SelectiveAxialEncoder3D(in_ch=1, channels=modal_channels)
        self.encoder_t2 = SelectiveAxialEncoder3D(in_ch=1, channels=modal_channels)
        self.encoder_flair = SelectiveAxialEncoder3D(in_ch=1, channels=modal_channels)

        # 2. Fusion: UltraLite (DSConv)
        self.fuse1 = FrequencyAwareFusion_UltraLite(modal_channels[0]) 
        self.fuse2 = FrequencyAwareFusion_UltraLite(modal_channels[1]) 
        
        self.fuse3 = CrossModalCalibrationFusion(modal_channels[2])
        self.fuse4 = CrossModalCalibrationFusion(modal_channels[3])
        
        # 3. Bottleneck
        self.bottleneck = StripPooling3D(modal_channels[3], modal_channels[3])
        
        # 4. Decoder: Original Up (Most Stable)
        self.decoders = nn.ModuleList()
        self.decoders.append(Up(modal_channels[3], modal_channels[2], modal_channels[2]))
        self.decoders.append(Up(modal_channels[2], modal_channels[1], modal_channels[1]))
        self.decoders.append(Up(modal_channels[1], modal_channels[0], modal_channels[0]))

        # 5. Output Heads (Deep Supervision Enabled)
        self.ds2 = nn.Conv3d(modal_channels[2], n_classes, 1)
        self.ds1 = nn.Conv3d(modal_channels[1], n_classes, 1)
        self.outc = nn.Conv3d(modal_channels[0], n_classes, 1)

    def forward(self, x_t1, x_t2, x_flair):
        t1 = self.encoder_t1(x_t1); t2 = self.encoder_t2(x_t2); flair = self.encoder_flair(x_flair)

        fused_skips = []
        x1, _ = self.fuse1(t1[0], t2[0], flair[0]); fused_skips.append(x1)
        x2, _ = self.fuse2(t1[1], t2[1], flair[1]); fused_skips.append(x2)
        x3, _ = self.fuse3(t1[2], t2[2], flair[2]); fused_skips.append(x3)
        x4, _ = self.fuse4(t1[3], t2[3], flair[3])
        
        x = self.bottleneck(x4)
        
        skip3 = fused_skips.pop(); x = self.decoders[0](x, skip3); ds2 = self.ds2(x)
        skip2 = fused_skips.pop(); x = self.decoders[1](x, skip2); ds1 = self.ds1(x)
        skip1 = fused_skips.pop(); x = self.decoders[2](x, skip1); final = self.outc(x)

        if self.training:
            return final, ds1, ds2
        return final

class AxialTubularExcitation(nn.Module):
    def __init__(self, dim, kernel_size=7):
        super().__init__()
        pad = (kernel_size - 1) // 2
        self.conv_d = nn.Conv3d(dim, dim, kernel_size=(kernel_size, 1, 1), padding=(pad, 0, 0), groups=dim)
        self.conv_h = nn.Conv3d(dim, dim, kernel_size=(1, kernel_size, 1), padding=(0, pad, 0), groups=dim)
        self.conv_w = nn.Conv3d(dim, dim, kernel_size=(1, 1, kernel_size), padding=(0, 0, pad), groups=dim)
        self.act = nn.GELU(); self.fusion = nn.Conv3d(dim * 3, dim, kernel_size=1)
    def forward(self, x):
        d = self.conv_d(x); h = self.conv_h(x); w = self.conv_w(x)
        return self.act(self.fusion(torch.cat([d, h, w], dim=1)))

class OptimizedAxialResBlock(nn.Module):
    """ [块内优化] Conv1用标准卷积，Conv2用DSConv """
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        # Conv1: 标准卷积 (任务重)
        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels); self.relu = nn.ReLU(True)
        # Conv2: DSConv (任务轻)
        self.conv2 = DSConv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.axial_branch = AxialTubularExcitation(out_channels, kernel_size=7)
        self.bn_axial = nn.BatchNorm3d(out_channels)
        self.selection = SelectiveFusion(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(nn.Conv3d(in_channels, out_channels, 1, stride=stride, bias=False), nn.BatchNorm3d(out_channels))
    def forward(self, x):
        res = self.shortcut(x)
        x = self.relu(self.bn1(self.conv1(x)))
        feat_local = self.bn2(self.conv2(x))
        feat_axial = self.bn_axial(self.axial_branch(x))
        return self.relu(self.selection(feat_local, feat_axial) + res)

class SelectiveAxialEncoder3D_Medium(nn.Module):
    def __init__(self, in_ch, channels):
        super().__init__()
        # Stem: 2 Std + 1 DS
        self.stem = nn.Sequential(
            nn.Conv3d(in_ch, channels[0] // 2, 3, 1, 1, bias=False), nn.BatchNorm3d(channels[0] // 2), nn.ReLU(True),
            nn.Conv3d(channels[0] // 2, channels[0] // 2, 3, 1, 1, bias=False), nn.BatchNorm3d(channels[0] // 2), nn.ReLU(True),
            DSConv3d(channels[0] // 2, channels[0], 3, 1, 1, bias=False), nn.BatchNorm3d(channels[0]), nn.ReLU(True)
        )
        self.layer1 = OptimizedAxialResBlock(channels[0], channels[0], stride=1)
        self.layer2 = OptimizedAxialResBlock(channels[0], channels[1], stride=2)
        self.layer3 = OptimizedAxialResBlock(channels[1], channels[2], stride=2)
        self.layer4 = OptimizedAxialResBlock(channels[2], channels[3], stride=2)
    def forward(self, x):
        x0 = self.stem(x); x1 = self.layer1(x0); x2 = self.layer2(x1); x3 = self.layer3(x2); x4 = self.layer4(x3)
        return [x1, x2, x3, x4]


class ThreeEncoder_EncoderOpt_Net(nn.Module):
    def __init__(self, n_classes, base_c=32, n_levels=4):
        super().__init__()
        modal_channels = [base_c * (2**i) for i in range(n_levels)]
        
        # 1. Encoder (Medium: Hybrid Stem + Hybrid Block)
        self.encoder_t1 = SelectiveAxialEncoder3D_Medium(in_ch=1, channels=modal_channels)
        self.encoder_t2 = SelectiveAxialEncoder3D_Medium(in_ch=1, channels=modal_channels)
        self.encoder_flair = SelectiveAxialEncoder3D_Medium(in_ch=1, channels=modal_channels)

        # 2. Fusion (UltraLite)
        self.fuse1 = FrequencyAwareFusion_UltraLite(modal_channels[0]) 
        self.fuse2 = FrequencyAwareFusion_UltraLite(modal_channels[1]) 
        self.fuse3 = CrossModalCalibrationFusion(modal_channels[2])
        self.fuse4 = CrossModalCalibrationFusion(modal_channels[3])
        
        # 3. Bottleneck
        self.bottleneck = StripPooling3D(modal_channels[3], modal_channels[3])
        
        # 4. Decoder (Original Up)
        self.decoders = nn.ModuleList()
        self.decoders.append(Up(modal_channels[3], modal_channels[2], modal_channels[2]))
        self.decoders.append(Up(modal_channels[2], modal_channels[1], modal_channels[1]))
        self.decoders.append(Up(modal_channels[1], modal_channels[0], modal_channels[0]))

        # 5. Deep Supervision
        self.ds2 = nn.Conv3d(modal_channels[2], n_classes, 1)
        self.ds1 = nn.Conv3d(modal_channels[1], n_classes, 1)
        self.outc = nn.Conv3d(modal_channels[0], n_classes, 1)

    def forward(self, x_t1, x_t2, x_flair):
        t1 = self.encoder_t1(x_t1); t2 = self.encoder_t2(x_t2); flair = self.encoder_flair(x_flair)

        fused_skips = []
        x1, _ = self.fuse1(t1[0], t2[0], flair[0]); fused_skips.append(x1)
        x2, _ = self.fuse2(t1[1], t2[1], flair[1]); fused_skips.append(x2)
        x3, _ = self.fuse3(t1[2], t2[2], flair[2]); fused_skips.append(x3)
        x4, _ = self.fuse4(t1[3], t2[3], flair[3])
        
        x = self.bottleneck(x4)
        
        skip3 = fused_skips.pop(); x = self.decoders[0](x, skip3); ds2 = self.ds2(x)
        skip2 = fused_skips.pop(); x = self.decoders[1](x, skip2); ds1 = self.ds1(x)
        skip1 = fused_skips.pop(); x = self.decoders[2](x, skip1); final = self.outc(x)

        if self.training:
            return final, ds1, ds2
        return final
    


class DoubleConv(nn.Module):
    """ [Baseline] 原始 U-Net 的基础模块: (Conv-BN-ReLU) * 2 """
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm3d(out_channels), nn.ReLU(True),
            nn.Conv3d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels), nn.ReLU(True)
        )
    def forward(self, x):
        return self.conv(x) # 无残差
class SelectiveFusion(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super().__init__()
        mid_channels = max(in_channels // reduction, 16)
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(nn.Linear(in_channels, mid_channels, bias=False), nn.ReLU(True), nn.Linear(mid_channels, in_channels * 2, bias=False))
    def forward(self, x_local, x_axial):
        b, c, d, h, w = x_local.size()
        s = self.avg_pool(x_local + x_axial).view(b, c)
        attn = F.softmax(self.fc(s).view(b, 2, c), dim=1)
        return x_local * attn[:, 0, :].view(b, c, 1, 1, 1) + x_axial * attn[:, 1, :].view(b, c, 1, 1, 1)

class AxialTubularExcitation(nn.Module):
    def __init__(self, dim, kernel_size=7):
        super().__init__()
        pad = (kernel_size - 1) // 2
        self.conv_d = nn.Conv3d(dim, dim, kernel_size=(kernel_size, 1, 1), padding=(pad, 0, 0), groups=dim)
        self.conv_h = nn.Conv3d(dim, dim, kernel_size=(1, kernel_size, 1), padding=(0, pad, 0), groups=dim)
        self.conv_w = nn.Conv3d(dim, dim, kernel_size=(1, 1, kernel_size), padding=(0, 0, pad), groups=dim)
        self.act = nn.GELU(); self.fusion = nn.Conv3d(dim * 3, dim, kernel_size=1)
    def forward(self, x):
        d = self.conv_d(x); h = self.conv_h(x); w = self.conv_w(x)
        return self.act(self.fusion(torch.cat([d, h, w], dim=1)))

class  HybridResBlock(nn.Module):
    """ [Ours] 混合残差块: Std + DS + Axial + Residual """
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels); self.relu = nn.ReLU(True)
        # DSConv 降计算量
        self.conv2 = DSConv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.axial_branch = AxialTubularExcitation(out_channels, kernel_size=7)
        self.bn_axial = nn.BatchNorm3d(out_channels)
        self.selection = SelectiveFusion(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(nn.Conv3d(in_channels, out_channels, 1, stride=stride, bias=False), nn.BatchNorm3d(out_channels))

    def forward(self, x):
        res = self.shortcut(x)
        x = self.relu(self.bn1(self.conv1(x)))
        feat_local = self.bn2(self.conv2(x))
        feat_axial = self.bn_axial(self.axial_branch(x))
        return self.relu(self.selection(feat_local, feat_axial) + res) # 有残差
class DynamicEncoder(nn.Module):
    def __init__(self, in_ch, channels, opt_encoder=False):
        super().__init__()
        self.opt_encoder = opt_encoder
        
        if opt_encoder: 
            # [Ours] Hybrid Deep Stem (2 Std + 1 DS)
            self.stem = nn.Sequential(
                nn.Conv3d(in_ch, channels[0] // 2, 3, 1, 1, bias=False), nn.BatchNorm3d(channels[0] // 2), nn.ReLU(True),
                nn.Conv3d(channels[0] // 2, channels[0] // 2, 3, 1, 1, bias=False), nn.BatchNorm3d(channels[0] // 2), nn.ReLU(True),
                DSConv3d(channels[0] // 2, channels[0], 3, 1, 1, bias=False), nn.BatchNorm3d(channels[0]), nn.ReLU(True)
            )
            Block = HybridResBlock
        else: 
            # [Baseline] Standard U-Net Stem (Double Conv)
            self.stem = DoubleConv(in_ch, channels[0], stride=1)
            Block = DoubleConv # 普通 DoubleConv，无残差
        
        self.layer1 = Block(channels[0], channels[0], stride=1)
        self.layer2 = Block(channels[0], channels[1], stride=2)
        self.layer3 = Block(channels[1], channels[2], stride=2)
        self.layer4 = Block(channels[2], channels[3], stride=2)

    def forward(self, x):
        x0 = self.stem(x); x1 = self.layer1(x0); x2 = self.layer2(x1); x3 = self.layer3(x2); x4 = self.layer4(x3)
        return [x1, x2, x3, x4]
class StandardBottleneck(nn.Module):
    """
    [Baseline] 原始 U-Net 的 Bottleneck
    结构: (Conv3x3 -> BN -> ReLU) * 2
    没有残差连接，没有 1x1 投影，最纯粹的卷积堆叠。
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # 在 U-Net 中，Bottleneck 通常保持通道数不变，或者翻倍。
        # 这里为了适配接口，第一层负责将 in_channels 映射到 out_channels
        
        self.conv = nn.Sequential(
            # 第一层卷积：调整通道数
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            
            # 第二层卷积：特征提炼
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)
class FrequencyAwareFusion_Lite1(nn.Module):
    """
    [最优平衡版] 
    策略：
    1. Competitor: Group=3 (保稳定，避免 Recall 崩盘)
    2. Gate & Final: DSConv (降计算量)
    预期 Fusion FLOPs: ~13G (原 Lite 版是 46G)
    """
    def __init__(self, in_channels):
        super().__init__()
        self.low_pass = nn.AvgPool3d(3, 1, 1)
        
        # 计算中间通道 (确保能被3整除)
        mid_channels = 3 * (in_channels // 2)
        
        # 1. Competitor: 保持 Group=3 (为了稳定性，这就也是 FLOPs 的主要来源，约 9G)
        self.competitor = nn.Sequential(
            nn.Conv3d(in_channels*3, mid_channels, kernel_size=3, padding=1, 
                      groups=3, bias=False), 
            nn.ReLU(True),
            nn.Conv3d(mid_channels, 3, kernel_size=1, bias=True),
            nn.Softmax(dim=1)
        )
        
        # 2. Shared Gate: 改回 DSConv (降算力，约 1G)
        self.shared_gate = nn.Sequential(
            # Depthwise
            nn.Conv3d(in_channels*3, in_channels*3, kernel_size=3, padding=1, 
                      groups=in_channels*3, bias=False),
            # Pointwise
            nn.Conv3d(in_channels*3, 3, kernel_size=1, bias=True),
            nn.Sigmoid()
        )
        
        # 3. Final: 改回 DSConv (降算力，约 1G)
        # 输入 2C (Low+High), 输出 C
        self.final = nn.Sequential(
            # Depthwise
            nn.Conv3d(in_channels*2, in_channels*2, kernel_size=3, padding=1, 
                      groups=in_channels*2, bias=False),
            # Pointwise
            nn.Conv3d(in_channels*2, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm3d(in_channels), 
            nn.ReLU(True)
        )

    def forward(self, x_t1, x_t2, x_flair):
        l_t1 = self.low_pass(x_t1); h_t1 = x_t1 - l_t1
        l_t2 = self.low_pass(x_t2); h_t2 = x_t2 - l_t2
        l_flair = self.low_pass(x_flair); h_flair = x_flair - l_flair
        
        # Low Freq
        low_cat = torch.cat([l_t1, l_t2, l_flair], dim=1)
        low_w = self.competitor(low_cat)
        w1, w2, w3 = torch.split(low_w, 1, dim=1)
        l_fused = l_t1*w1 + l_t2*w2 + l_flair*w3
        
        # High Freq
        high_cat = torch.cat([h_t1, h_t2, h_flair], dim=1)
        gates = self.shared_gate(high_cat)
        g1, g2, g3 = torch.split(gates, 1, dim=1)
        h_fused = h_t1*g1 + h_t2*g2 + h_flair*g3
        
        return self.final(torch.cat([l_fused, h_fused], dim=1)), {}
class Ablation_ThreeEncoder_Final_Net(nn.Module):
    def __init__(self, n_classes, base_c=32, n_levels=4, 
                 opt_encoder=True,        # True=Hybrid(Medium), False=Standard(SimpleStem)
                 opt_fusion_shallow=True, # True=UltraLite(DS), False=Naive(Concat)
                 opt_fusion_deep=True,    # True=Calibration, False=Naive(Concat)
                 deep_sup=False):          # True=3 Outputs, False=1 Output
        super().__init__()
        self.deep_sup = deep_sup
        modal_channels = [base_c * (2**i) for i in range(n_levels)]
        
        # 1. Encoders
        self.encoder_t1 = DynamicEncoder(1, modal_channels, opt_encoder)
        self.encoder_t2 = DynamicEncoder(1, modal_channels, opt_encoder)
        self.encoder_flair = DynamicEncoder(1, modal_channels, opt_encoder)

        # 2. Fusion
        # Shallow L1/L2
        if opt_fusion_shallow:
            self.fuse1 = FrequencyAwareFusion_UltraLite(modal_channels[0])
            self.fuse2 = FrequencyAwareFusion_UltraLite(modal_channels[1])
        else:
            self.fuse1 = NaiveFusion(modal_channels[0])
            self.fuse2 = NaiveFusion(modal_channels[1])
            
        # Deep L3/L4
        if opt_fusion_deep:
            self.fuse3 = CrossModalCalibrationFusion(modal_channels[2])
            self.fuse4 = CrossModalCalibrationFusion(modal_channels[3])
        else:
            self.fuse3 = NaiveFusion(modal_channels[2])
            self.fuse4 = NaiveFusion(modal_channels[3])
        if opt_encoder:
            # 3. Bottleneck
            self.bottleneck = StripPooling3D(modal_channels[3], modal_channels[3])
        else:
            self.bottleneck = StandardBottleneck(modal_channels[3], modal_channels[3])
            
        # 4. Decoder (Fixed to Original Up)
        self.decoders = nn.ModuleList()
        self.decoders.append(Up(modal_channels[3], modal_channels[2], modal_channels[2]))
        self.decoders.append(Up(modal_channels[2], modal_channels[1], modal_channels[1]))
        self.decoders.append(Up(modal_channels[1], modal_channels[0], modal_channels[0]))

        # 5. Output Heads
        self.outc = nn.Conv3d(modal_channels[0], n_classes, 1)
        if self.deep_sup:
            self.ds2 = nn.Conv3d(modal_channels[2], n_classes, 1)
            self.ds1 = nn.Conv3d(modal_channels[1], n_classes, 1)

    def forward(self, x_t1, x_t2, x_flair):
        t1 = self.encoder_t1(x_t1); t2 = self.encoder_t2(x_t2); flair = self.encoder_flair(x_flair)

        fused_skips = []
        x1, _ = self.fuse1(t1[0], t2[0], flair[0]); fused_skips.append(x1)
        x2, _ = self.fuse2(t1[1], t2[1], flair[1]); fused_skips.append(x2)
        x3, _ = self.fuse3(t1[2], t2[2], flair[2]); fused_skips.append(x3)
        x4, _ = self.fuse4(t1[3], t2[3], flair[3])
        
        x = self.bottleneck(x4)
        
        skip3 = fused_skips.pop(); x = self.decoders[0](x, skip3); ds2 = self.ds2(x) if self.deep_sup else None
        skip2 = fused_skips.pop(); x = self.decoders[1](x, skip2); ds1 = self.ds1(x) if self.deep_sup else None
        skip1 = fused_skips.pop(); x = self.decoders[2](x, skip1); final = self.outc(x)

        if self.training and self.deep_sup:
            return final, ds1, ds2
        return final
    

class Ablation_ThreeEncoder_Final_Net_1(nn.Module):
    def __init__(self, n_classes, base_c=32, n_levels=4, 
                 opt_encoder=True,        # True=Hybrid(Medium), False=Standard(SimpleStem)
                 opt_fusion_shallow=True, # True=UltraLite(DS), False=Naive(Concat)
                 opt_fusion_deep=True,    # True=Calibration, False=Naive(Concat)
                 deep_sup=True):          # True=3 Outputs, False=1 Output
        super().__init__()
        self.deep_sup = deep_sup
        modal_channels = [base_c * (2**i) for i in range(n_levels)]
        # 例如 base_c=16, n=4 -> [16, 32, 64, 128]
        # modal_channels[3] = 128
        
        # 1. Encoders (保持不变)
        self.encoder_t1 = DynamicEncoder(1, modal_channels, opt_encoder)
        self.encoder_t2 = DynamicEncoder(1, modal_channels, opt_encoder)
        self.encoder_flair = DynamicEncoder(1, modal_channels, opt_encoder)

        # 2. Fusion (保持不变)
        # Shallow L1/L2
        if opt_fusion_shallow:
            self.fuse1 = FrequencyAwareFusion_UltraLite(modal_channels[0])
            self.fuse2 = FrequencyAwareFusion_UltraLite(modal_channels[1])
        else:
            self.fuse1 = NaiveFusion(modal_channels[0])
            self.fuse2 = NaiveFusion(modal_channels[1])
            
        # Deep L3/L4
        if opt_fusion_deep:
            self.fuse3 = CrossModalCalibrationFusion(modal_channels[2])
            self.fuse4 = CrossModalCalibrationFusion(modal_channels[3])
        else:
            self.fuse3 = NaiveFusion(modal_channels[2])
            self.fuse4 = NaiveFusion(modal_channels[3])

        # ======================================================================
        # 3. Bottleneck (核心修改点 1)
        # ======================================================================
        # 标准设计：瓶颈层输出通道翻倍 (128 -> 256)
        # 原来是: modal_channels[3] -> modal_channels[3]
        
        enc_out_c = modal_channels[3]      # 128
        bot_out_c = modal_channels[3] * 2  # 256 (翻倍)

        if opt_encoder:
            # 假设 StripPooling3D 支持 (in, out) 参数
            self.bottleneck = StandardBottleneck(enc_out_c, bot_out_c)#StripPooling3D(enc_out_c, bot_out_c)
        else:
            self.bottleneck = StandardBottleneck(enc_out_c, bot_out_c)
            
        # ======================================================================
        # 4. Decoder (核心修改点 2)
        # ======================================================================
        self.decoders = nn.ModuleList()
        
        # --- Decoder Stage 1 (最深层) ---
        # 这一层接收 Bottleneck 的输出，所以输入通道是 bot_out_c (256)
        # Skip connection 来自 Encoder L3 (128)
        # 输出目标通常恢复到 L3 的大小 (128) 或 L2 的大小 (64)？
        # 根据你原本的代码逻辑: Up(in, skip, out)
        # 原代码: Up(modal_channels[3], modal_channels[2], modal_channels[2])
        # 修改后: Up(bot_out_c,         modal_channels[2], modal_channels[2])
        
        self.decoders.append(Up(bot_out_c, modal_channels[2], modal_channels[2]))
        
        # --- Decoder Stage 2 & 3 (保持不变) ---
        # 接下来的层输入来自上一层 Decoder 的输出 (modal_channels[2])
        self.decoders.append(Up(modal_channels[2], modal_channels[1], modal_channels[1]))
        self.decoders.append(Up(modal_channels[1], modal_channels[0], modal_channels[0]))

        # 5. Output Heads
        self.outc = nn.Conv3d(modal_channels[0], n_classes, 1)
        if self.deep_sup:
            self.ds2 = nn.Conv3d(modal_channels[2], n_classes, 1)
            self.ds1 = nn.Conv3d(modal_channels[1], n_classes, 1)

    def forward(self, x_t1, x_t2, x_flair):
        t1 = self.encoder_t1(x_t1); t2 = self.encoder_t2(x_t2); flair = self.encoder_flair(x_flair)

        fused_skips = []
        x1, _ = self.fuse1(t1[0], t2[0], flair[0]); fused_skips.append(x1)
        x2, _ = self.fuse2(t1[1], t2[1], flair[1]); fused_skips.append(x2)
        x3, _ = self.fuse3(t1[2], t2[2], flair[2]); fused_skips.append(x3)
        x4, _ = self.fuse4(t1[3], t2[3], flair[3])
        
        # Bottleneck 现在会输出 2倍通道
        x = self.bottleneck(x4) 
        
        # Decoder 1 会处理这个 2倍通道的输入
        skip3 = fused_skips.pop(); x = self.decoders[0](x, skip3); ds2 = self.ds2(x) if self.deep_sup else None
        skip2 = fused_skips.pop(); x = self.decoders[1](x, skip2); ds1 = self.ds1(x) if self.deep_sup else None
        skip1 = fused_skips.pop(); x = self.decoders[2](x, skip1); final = self.outc(x)

        if self.training and self.deep_sup:
            return final, ds1, ds2
        return final


import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict

# ==============================================================================
# 1. 在这里导入你的模型类
#    (请根据你的文件结构修改 import 路径，或者直接把模型定义粘贴到这个脚本里)
# ==============================================================================
try:
    # 假设你的模型文件叫 three_encoder_lite.py，类名是 ThreeEncoder_Lite_NoDS_Net
    # from networks.three_encoder_lite import ThreeEncoder_Lite_NoDS_Net 
    pass 
except ImportError:
    pass

# 为了演示，我这里假设你已经引入了 ThreeEncoder_Lite_NoDS_Net
# 如果报错 "NameError", 请确保你 import 了该类，或者把该类的代码贴在上面

# ==============================================================================
# 2. 核心统计工具函数 (基于 Hook)
# ==============================================================================
def count_flops_params(model, input_data):
    """
    使用 Forward Hook 精确统计每一层的 FLOPs 和 Params
    """
    layer_stats = OrderedDict()
    
    # --- FLOPs 计算规则 ---
    def conv3d_flops(module, input, output):
        # output: (B, Cout, D, H, W)
        batch_size = output.shape[0]
        output_elements = output.numel() // batch_size
        
        # MACs = (Cin / groups) * K * K * K
        kernel_ops = (module.in_channels // module.groups) * np.prod(module.kernel_size)
        bias_ops = 1 if module.bias is not None else 0
        
        return batch_size * output_elements * (kernel_ops + bias_ops)

    def linear_flops(module, input, output):
        batch_size = output.shape[0]
        weight_ops = module.in_features * module.out_features
        bias_ops = module.out_features if module.bias is not None else 0
        total_ops = batch_size * (weight_ops + bias_ops)
        # 处理多维输入 (B, *, Cin)
        if input[0].dim() > 2:
            num_tokens = input[0].numel() // (input[0].shape[-1] * batch_size)
            total_ops *= num_tokens
        return total_ops

    # --- Hook 函数 ---
    def hook_fn(module, input, output, name):
        if isinstance(module, nn.Conv3d):
            flops = conv3d_flops(module, input, output)
        elif isinstance(module, nn.Linear):
            flops = linear_flops(module, input, output)
        elif isinstance(module, (nn.BatchNorm3d, nn.LayerNorm, nn.GroupNorm, nn.InstanceNorm3d)):
            flops = input[0].numel() 
        elif isinstance(module, (nn.ReLU, nn.GELU, nn.Sigmoid, nn.Softmax, nn.SiLU)):
            flops = input[0].numel() 
        elif isinstance(module, (nn.AdaptiveAvgPool3d, nn.AvgPool3d, nn.MaxPool3d, nn.Upsample)):
             flops = input[0].numel() 
        else:
            flops = 0
            
        params = sum(p.numel() for p in module.parameters())
        
        if name not in layer_stats:
            layer_stats[name] = {'flops': 0, 'params': params}
        layer_stats[name]['flops'] += flops

    # --- 注册 Hooks ---
    hooks = []
    for name, module in model.named_modules():
        # 跳过容器层，只统计叶子层
        if len(list(module.children())) == 0: 
            if name == "": continue
            # 使用闭包绑定 name
            h = module.register_forward_hook(lambda m, i, o, n=name: hook_fn(m, i, o, n))
            hooks.append(h)

    # --- 前向传播 ---
    model.eval()
    with torch.no_grad():
        model(*input_data)

    for h in hooks: h.remove()
    return layer_stats

def analyze_complexity(model, patch_size=(96, 96, 96), input_mode='split'):
    """
    input_mode: 'split' (x1, x2, x3) or 'cat' (concat_x)
    """
    device = next(model.parameters()).device
    model_name = model.__class__.__name__
    print(f"\n{'='*20} Analyzing: {model_name} {'='*20}")
    print(f"Input Size: {patch_size} | Mode: {input_mode}")

    # 1. 构造输入
    x1 = torch.randn(1, 1, *patch_size).to(device)
    x2 = torch.randn(1, 1, *patch_size).to(device)
    x3 = torch.randn(1, 1, *patch_size).to(device)
    
    if input_mode == 'split':
        input_data = (x1, x2, x3)
    else:
        # Concatenate along channel dim (dim=1) -> (1, 3, D, H, W)
        x_cat = torch.cat([x1, x2, x3], dim=1)
        input_data = (x_cat, ) # Tuple
    
    # 2. 运行统计
    layer_stats = count_flops_params(model, input_data)
    
    # ... (后续代码保持不变，直到 groups 定义) ...
    
    # 3. 定义分组 (为了兼容不同模型，我们稍微放宽匹配规则)
    groups = {
        "Encoder": ["enc", "down"],
        "Fusion/Skip": ["fuse", "cat"], 
        "Bottleneck": ["bottle", "mid"],
        "Decoder": ["dec", "up"],
        "Heads": ["out", "ds"]
    }
    
    # ... (后续打印逻辑保持不变) ...
    # 复制你原来的打印代码到这里即可
    # 为了完整性，这里我重写了打印部分：
    
    group_stats = {k: {'flops': 0, 'params': 0} for k in groups}
    group_stats["Other"] = {'flops': 0, 'params': 0}
    
    total_flops = 0
    
    for layer_name, stats in layer_stats.items():
        total_flops += stats['flops']
        matched = False
        for g_name, keywords in groups.items():
            for kw in keywords:
                # lower() 用于大小写不敏感匹配
                if kw in layer_name.lower():
                    group_stats[g_name]['flops'] += stats['flops']
                    group_stats[g_name]['params'] += stats['params']
                    matched = True
                    break
            if matched: break
        
        if not matched:
            group_stats["Other"]['flops'] += stats['flops']
            group_stats["Other"]['params'] += stats['params']

    real_total_params = sum(p.numel() for p in model.parameters())

    print(f"\n{'-'*80}")
    print(f"{'Component':<30} | {'Params (M)':<12} | {'FLOPs (G)':<12} | {'% FLOPs':<8}")
    print(f"{'-'*80}")
    
    for name, stat in group_stats.items():
        p_m = stat['params'] / 1e6
        f_g = stat['flops'] / 1e9
        ratio = (stat['flops'] / total_flops) * 100 if total_flops > 0 else 0
        print(f"{name:<30} | {p_m:<12.3f} | {f_g:<12.3f} | {ratio:>6.1f}%")
        
    print(f"{'-'*80}")
    print(f"{'TOTAL':<30} | {real_total_params/1e6:<12.3f} | {total_flops/1e9:<12.3f} | 100.0%")
    print(f"{'='*80}\n")



def analyze_complexity(model, patch_size=(96, 96, 96)):
    device = next(model.parameters()).device
    model_name = model.__class__.__name__
    print(f"\n{'='*20} Analyzing: {model_name} {'='*20}")
    print(f"Input Size: {patch_size}")

    # 1. 构造三流输入 (根据你的模型 forward 需求)
    # 你的 forward(self, x_t1, x_t2, x_flair)
    x1 = torch.randn(1, 1, *patch_size).to(device)
    x2 = torch.randn(1, 1, *patch_size).to(device)
    x3 = torch.randn(1, 1, *patch_size).to(device)
    
    # 2. 运行统计
    layer_stats = count_flops_params(model, (x1, x2, x3))
    
    # 3. 定义分组 (根据你的变量命名习惯)
    groups = {
        "Encoder (SelectiveAxial)": ["encoder"],
        "Fusion (Shallow)": ["fuse1", "fuse2"],
        "Fusion (Deep)": ["fuse3", "fuse4"],
        "Bottleneck (StripPool)": ["bottleneck", "transformer"],
        "Decoder (Up)": ["decoder"],
        "Heads (DS)": ["ds", "outc"]
    }
    
    group_stats = {k: {'flops': 0, 'params': 0} for k in groups}
    group_stats["Other"] = {'flops': 0, 'params': 0}
    
    total_flops = 0
    
    for layer_name, stats in layer_stats.items():
        total_flops += stats['flops']
        matched = False
        for g_name, keywords in groups.items():
            for kw in keywords:
                if kw in layer_name:
                    group_stats[g_name]['flops'] += stats['flops']
                    group_stats[g_name]['params'] += stats['params']
                    matched = True
                    break
            if matched: break
        
        if not matched:
            group_stats["Other"]['flops'] += stats['flops']
            group_stats["Other"]['params'] += stats['params']

    real_total_params = sum(p.numel() for p in model.parameters())

    # 4. 打印报表
    print(f"\n{'-'*80}")
    print(f"{'Component':<30} | {'Params (M)':<12} | {'FLOPs (G)':<12} | {'% FLOPs':<8}")
    print(f"{'-'*80}")
    
    for name, stat in group_stats.items():
        p_m = stat['params'] / 1e6
        f_g = stat['flops'] / 1e9
        ratio = (stat['flops'] / total_flops) * 100 if total_flops > 0 else 0
        print(f"{name:<30} | {p_m:<12.3f} | {f_g:<12.3f} | {ratio:>6.1f}%")
        
    print(f"{'-'*80}")
    print(f"{'TOTAL':<30} | {real_total_params/1e6:<12.3f} | {total_flops/1e9:<12.3f} | 100.0%")
    print(f"{'='*80}\n")

# ==============================================================================
# 3. 主程序：在这里实例化并调用
# ==============================================================================
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # ---------------------------------------------------------
    # [直接实例化] 
    # 请确保 ThreeEncoder_Lite_NoDS_Net 类已导入或定义在上方 
    # ---------------------------------------------------------
    
    
    # 实例化
    # 注意：这里你可以随意修改 base_c 看看对 FLOPs 的影响
    #my_model = ThreeEncoder_Lite_Net(n_classes=2, base_c=16, n_levels=4).to(device)
    #my_model =ThreeEncoder_SelectiveAxial_Net(n_classes=2, base_c=16, n_levels=4).to(device)
    #my_model = ThreeEncoderNaiveFusionUNet(n_classes=2, base_c=16, n_levels=4).to(device)
    my_model = Ablation_ThreeEncoder_Final_Net(n_classes=2, base_c=16, n_levels=4).to(device)
    # 运行分析
    analyze_complexity(my_model, patch_size=(96, 96, 96))


# #############################################################
# if __name__ == "__main__":
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     patch_size = (96, 96, 96)
    
#     # 模拟 args 参数，为了方便测试
#     class Args:
#         num_classes = 2
#         base_c = 16
#     args = Args()

#     print(f"当前设备: {device}")
#     print(f"Patch Size: {patch_size}")
#     print(f"Base Channels: {args.base_c}")

#     # --------------------------------------------------------------------------
#     # 1. 测试你的标准 UNet3D (基准对比)
#     # --------------------------------------------------------------------------
#     try:
#         print("\n>>> 正在初始化 UNet3D ...")
#         # 直接使用你的初始化逻辑
#         model_unet = UNet3D(3, args.num_classes, base_c=args.base_c).to(device)
        
#         # 标准 UNet 通常接受拼接后的输入 (B, 3, D, H, W)，所以 mode='cat'
#         analyze_complexity(model_unet, patch_size, input_mode='cat')
#     except NameError:
#         print("错误: 未找到 'UNet3D' 类，请在脚本顶部 import 它。")
#     except Exception as e:
#         print(f"UNet3D 测试出错: {e}")

#     # --------------------------------------------------------------------------
#     # 2. 测试你的 ResUNet3D (基准对比)
#     # --------------------------------------------------------------------------
#     try:
#         print("\n>>> 正在初始化 ResUNet3D ...")
#         # 直接使用你的初始化逻辑
#         model_res = ResUNet3D(3, args.num_classes, base_c=args.base_c).to(device)
        
#         # ResUNet 也接受拼接后的输入 (B, 3, D, H, W)
#         analyze_complexity(model_res, patch_size, input_mode='cat')
#     except NameError:
#         print("错误: 未找到 'ResUNet3D' 类，请在脚本顶部 import 它。")
#     except Exception as e:
#         print(f"ResUNet3D 测试出错: {e}")

#     # --------------------------------------------------------------------------
#     # 3. 测试你的自定义三流模型 (你的主要工作)
#     # --------------------------------------------------------------------------
#     try:
#         print("\n>>> 正在初始化 自定义三流模型 ...")
#         # 这里实例化你的核心模型
#         # my_model = ThreeEncoderNaiveFusionUNet(n_classes=args.num_classes, base_c=args.base_c).to(device)
        
#         # 注意：你的模型接受 3 个独立的输入，所以 mode='split'
#         # analyze_complexity(my_model, patch_size, input_mode='split')
#         pass 
#     except Exception as e:
#         print(f"自定义模型测试出错: {e}")










# import torch
# import torch.nn as nn
# import numpy as np
# from collections import OrderedDict
# import pandas as pd

# # ==============================================================================
# # 1. 模型定义区域 
# #    (请确保你的 ThreeEncoder_Final_SOTA_Net 及相关类已定义或导入)
# # ==============================================================================
# try:
#     # 示例导入，请替换为你实际的文件路径
#     # from networks.three_encoder_final import ThreeEncoder_Final_SOTA_Net
#     pass
# except ImportError:
#     pass

# # ==============================================================================
# # 2. 核心分析工具 (Refined Analyzer)
# # ==============================================================================
# class ModelAnalyzer:
#     def __init__(self, model, input_data):
#         self.model = model
#         self.input_data = input_data
#         self.hooks = []
#         self.layer_stats = OrderedDict()

#     def _register_hooks(self):
#         def hook_fn(module, input, output, name):
#             # 获取输入分辨率 (D, H, W)
#             input_shape = input[0].shape
#             resolution = f"{input_shape[2]}x{input_shape[3]}x{input_shape[4]}" if len(input_shape) > 4 else "N/A"
            
#             flops = 0
#             params = sum(p.numel() for p in module.parameters())

#             # --- FLOPs Calculation Rules ---
#             if isinstance(module, nn.Conv3d):
#                 batch_size = output.shape[0]
#                 output_elements = output.numel() // batch_size
#                 # MACs = (Cin / groups) * K * K * K
#                 kernel_ops = (module.in_channels // module.groups) * np.prod(module.kernel_size)
#                 bias_ops = 1 if module.bias is not None else 0
#                 flops = batch_size * output_elements * (kernel_ops + bias_ops)
            
#             elif isinstance(module, nn.Linear):
#                 batch_size = input[0].shape[0]
#                 weight_ops = module.in_features * module.out_features
#                 bias_ops = module.out_features if module.bias is not None else 0
#                 # Handle multi-dimensional input (B, *, Cin)
#                 num_tokens = input[0].numel() // input[0].shape[-1]
#                 flops = num_tokens * (weight_ops + bias_ops) // batch_size * batch_size # Normalize per batch
            
#             elif isinstance(module, (nn.BatchNorm3d, nn.LayerNorm, nn.GroupNorm, nn.InstanceNorm3d)):
#                 flops = input[0].numel() # Element-wise ops
            
#             elif isinstance(module, (nn.ReLU, nn.GELU, nn.Sigmoid, nn.Softmax, nn.SiLU, nn.LeakyReLU)):
#                 flops = input[0].numel() # Element-wise activation
            
#             elif isinstance(module, (nn.AdaptiveAvgPool3d, nn.AvgPool3d, nn.MaxPool3d, nn.Upsample)):
#                 flops = input[0].numel() # Simple mapping

#             # Store stats
#             if name not in self.layer_stats:
#                 self.layer_stats[name] = {
#                     'flops': 0, 
#                     'params': params, 
#                     'resolution': resolution,
#                     'module_type': type(module).__name__
#                 }
#             self.layer_stats[name]['flops'] += flops

#         # Recursively register hooks for leaf nodes
#         for name, module in self.model.named_modules():
#             if len(list(module.children())) == 0: 
#                 if name == "": continue
#                 h = module.register_forward_hook(lambda m, i, o, n=name: hook_fn(m, i, o, n))
#                 self.hooks.append(h)

#     def run(self):
#         self.model.eval()
#         self._register_hooks()
        
#         with torch.no_grad():
#             self.model(*self.input_data)
        
#         for h in self.hooks: h.remove()
#         return self.layer_stats

# # ==============================================================================
# # 3. 高级分组统计与打印
# # ==============================================================================
# def print_analysis(model, patch_size=(96, 96, 96)):
#     device = next(model.parameters()).device
#     print(f"\n{'='*30} 深度模型分析 {'='*30}")
#     print(f"Model: {model.__class__.__name__}")
#     print(f"Input: {patch_size}")

#     # 构造输入
#     x1 = torch.randn(1, 1, *patch_size).to(device)
#     x2 = torch.randn(1, 1, *patch_size).to(device)
#     x3 = torch.randn(1, 1, *patch_size).to(device)

#     # 运行分析
#     analyzer = ModelAnalyzer(model, (x1, x2, x3))
#     raw_stats = analyzer.run()

#     # --- 定义精细化分组 ---
#     groups = OrderedDict({
#         "Encoder [Stem] (High Res)": ["encoder", "stem"], 
#         "Encoder [Axial Block]":     ["encoder", "axial"],
#         "Encoder [Downsample]":      ["encoder", "conv1"],
#         "Encoder [Other]":           ["encoder"],          
#         "Fusion [Shallow]":          ["fuse1", "fuse2"],
#         "Fusion [Deep]":             ["fuse3", "fuse4"],
#         "Bottleneck":                ["bottleneck", "transformer"],
#         "Decoder":                   ["decoder"],
#         "Heads":                     ["ds", "outc"]
#     })

#     grouped_stats = {k: {'params': 0, 'flops': 0} for k in groups}
#     grouped_stats["[Uncategorized]"] = {'params': 0, 'flops': 0}
    
#     total_flops = 0
    
#     for name, stat in raw_stats.items():
#         total_flops += stat['flops']
#         matched = False
        
#         # 匹配逻辑
#         for group_name, keywords in groups.items():
#             is_hit = True
#             for kw in keywords:
#                 if kw not in name:
#                     is_hit = False
#                     break
            
#             if is_hit:
#                 grouped_stats[group_name]['flops'] += stat['flops']
#                 grouped_stats[group_name]['params'] += stat['params']
#                 matched = True
#                 break 
        
#         if not matched:
#             grouped_stats["[Uncategorized]"]['flops'] += stat['flops']
#             grouped_stats["[Uncategorized]"]['params'] += stat['params']

#     # --- 打印表格 ---
#     df_data = []
#     for k, v in grouped_stats.items():
#         if v['flops'] == 0 and v['params'] == 0: continue
#         df_data.append({
#             "Module Group": k,
#             "Params (M)": v['params'] / 1e6,
#             "FLOPs (G)": v['flops'] / 1e9,
#             "Ratio (%)": (v['flops'] / total_flops) * 100 if total_flops > 0 else 0
#         })
    
#     df = pd.DataFrame(df_data)
    
#     print("-" * 70)
#     print(f"{'Module Group':<30} | {'Params (M)':>10} | {'FLOPs (G)':>10} | {'Ratio':>6}")
#     print("-" * 70)
    
#     for _, row in df.iterrows():
#         # [修正] 这里使用正确的列名 'Ratio (%)'
#         print(f"{row['Module Group']:<30} | {row['Params (M)']:>10.3f} | {row['FLOPs (G)']:>10.3f} | {row['Ratio (%)']:>5.1f}%")
        
#     print("-" * 70)
#     real_total_params = sum(p.numel() for p in model.parameters())
#     print(f"{'TOTAL':<30} | {real_total_params/1e6:>10.3f} | {total_flops/1e9:>10.3f} | 100.0%")
#     print("=" * 70 + "\n")

# # ==============================================================================
# # 4. 执行区域
# # ==============================================================================
# if __name__ == "__main__":
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
#     # 请确保 ThreeEncoder_Final_SOTA_Net 类已定义
#     # 这里使用 base_c=16 进行测试
    
#     try:
#         # 实例化模型
#         model = ThreeEncoder_EncoderOpt_Net(n_classes=2, base_c=16, n_levels=4).to(device)
        
#         # 运行详细分析
#         print_analysis(model, patch_size=(96, 96, 96))
        
#     except NameError:
#         print("错误: 请先定义或导入 'ThreeEncoder_Final_SOTA_Net' 类。")