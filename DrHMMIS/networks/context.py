import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.gate_cnn_transformer import ModalEncoder3D,TransformerBlock,Up,Light_CrossAttention_Block_Symmetric
from networks.mogai_block import CSSD_Up,SelectiveAxialEncoder3D,TransformerEnhancedEncoder3D
# --- 1. 原版 L1, L2 模块 (完全保留) ---
class FrequencyAwareFusion(nn.Module):
    """
    (原版 L1, L2) - 你反馈效果最好的版本
    """
    def __init__(self, in_channels):
        super().__init__()
        self.low_pass = nn.AvgPool3d(kernel_size=3, stride=1, padding=1)
        
        # 低频竞争
        self.low_freq_competitor = nn.Sequential(
            nn.Conv3d(in_channels * 3, in_channels // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels // 2, 3, kernel_size=1),
            nn.Softmax(dim=1)
        )
        
        # 高频门控
        self.high_freq_gate_t1 = nn.Sequential(nn.Conv3d(in_channels * 3, in_channels // 2, 1), nn.ReLU(), nn.Conv3d(in_channels // 2, 1, 1), nn.Sigmoid())
        self.high_freq_gate_t2 = nn.Sequential(nn.Conv3d(in_channels * 3, in_channels // 2, 1), nn.ReLU(), nn.Conv3d(in_channels // 2, 1, 1), nn.Sigmoid())
        self.high_freq_gate_flair = nn.Sequential(nn.Conv3d(in_channels * 3, in_channels // 2, 1), nn.ReLU(), nn.Conv3d(in_channels // 2, 1, 1), nn.Sigmoid())
        
        self.final_conv = nn.Sequential(
            nn.Conv3d(in_channels * 2, in_channels, 3, 1, 1),
            nn.BatchNorm3d(in_channels), nn.ReLU(inplace=True),
            nn.Conv3d(in_channels, in_channels, 3, 1, 1),
            nn.BatchNorm3d(in_channels), nn.ReLU(inplace=True)
        )

    def forward(self, x_t1, x_t2, x_flair):
        l_t1 = self.low_pass(x_t1); h_t1 = x_t1 - l_t1
        l_t2 = self.low_pass(x_t2); h_t2 = x_t2 - l_t2
        l_flair = self.low_pass(x_flair); h_flair = x_flair - l_flair
        
        low_cat = torch.cat([l_t1, l_t2, l_flair], dim=1)
        low_weights = self.low_freq_competitor(low_cat)
        l_fused = (l_t1 * low_weights[:, 0:1]) + (l_t2 * low_weights[:, 1:2]) + (l_flair * low_weights[:, 2:3])
        
        high_cat = torch.cat([h_t1, h_t2, h_flair], dim=1)
        g_h_t1 = self.high_freq_gate_t1(high_cat)
        g_h_t2 = self.high_freq_gate_t2(high_cat)
        g_h_flair = self.high_freq_gate_flair(high_cat)
        
        h_fused = (h_t1 * g_h_t1) + (h_t2 * g_h_t2) + (h_flair * g_h_flair)
        out = self.final_conv(torch.cat([l_fused, h_fused], dim=1))
        return out, {'t2_high_gate': g_h_t2}



class FrequencyAwareFusion_WithMax(nn.Module):
    """
    [L1, L2 改进版] 
    在原版 AvgPool 基础上增加 MaxPool 辅助权重计算。
    关键：返回 (out, dict) 元组，以适配主网络的解包逻辑。
    """
    def __init__(self, in_channels):
        super().__init__()
        # 1. 物理频率分离器 (保持 AvgPool)
        self.low_pass_avg = nn.AvgPool3d(kernel_size=3, stride=1, padding=1)
        
        # 2. [新增] 峰值提取器 (Max Pooling)
        self.peak_extractor = nn.MaxPool3d(kernel_size=3, stride=1, padding=1)
        
        # 3. 低频竞争 (输入通道翻倍: 3*(Avg+Max)=6C)
        self.low_freq_competitor = nn.Sequential(
            nn.Conv3d(in_channels * 6, in_channels // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels // 2, 3, kernel_size=1),
            nn.Softmax(dim=1) 
        )
        
        # 4. 高频门控 (保持不变)
        self.gate_t1 = nn.Sequential(nn.Conv3d(in_channels*3, in_channels//2, 1), nn.ReLU(), nn.Conv3d(in_channels//2, 1, 1), nn.Sigmoid())
        self.gate_t2 = nn.Sequential(nn.Conv3d(in_channels*3, in_channels//2, 1), nn.ReLU(), nn.Conv3d(in_channels//2, 1, 1), nn.Sigmoid())
        self.gate_flair = nn.Sequential(nn.Conv3d(in_channels*3, in_channels//2, 1), nn.ReLU(), nn.Conv3d(in_channels//2, 1, 1), nn.Sigmoid())
        
        self.final_conv = nn.Sequential(
            nn.Conv3d(in_channels * 2, in_channels, 3, 1, 1),
            nn.BatchNorm3d(in_channels), nn.ReLU(inplace=True),
            nn.Conv3d(in_channels, in_channels, 3, 1, 1),
            nn.BatchNorm3d(in_channels), nn.ReLU(inplace=True)
        )

    def forward(self, x_t1, x_t2, x_flair):
        # A. 频率分解
        l_t1 = self.low_pass_avg(x_t1); h_t1 = x_t1 - l_t1
        l_t2 = self.low_pass_avg(x_t2); h_t2 = x_t2 - l_t2
        l_flair = self.low_pass_avg(x_flair); h_flair = x_flair - l_flair
        
        # B. [改进] 低频融合 (引入 MaxPool)
        m_t1 = self.peak_extractor(x_t1)
        m_t2 = self.peak_extractor(x_t2)
        m_flair = self.peak_extractor(x_flair)
        
        # 拼接 Avg 和 Max 用于决策
        decision_cat = torch.cat([l_t1, m_t1, l_t2, m_t2, l_flair, m_flair], dim=1)
        w_low = self.low_freq_competitor(decision_cat)
        
        # 加权 (依然融合 l_tx)
        l_fused = (l_t1 * w_low[:, 0:1]) + (l_t2 * w_low[:, 1:2]) + (l_flair * w_low[:, 2:3])
        
        # C. 高频融合
        high_cat = torch.cat([h_t1, h_t2, h_flair], dim=1)
        g_h_t1 = self.gate_t1(high_cat); g_h_t2 = self.gate_t2(high_cat); g_h_flair = self.gate_flair(high_cat)
        h_fused = (h_t1 * g_h_t1) + (h_t2 * g_h_t2) + (h_flair * g_h_flair)
        
        out = self.final_conv(torch.cat([l_fused, h_fused], dim=1))
        
        # [关键修复] 返回元组，匹配 x, _ = ...
        return out, {'t2_high_gate': g_h_t2}

# --- 2. 新设计的底层模块 (L3, L4) ---

class CrossModalCalibrationFusion(nn.Module):
    """
    (L3, L4 新模块) - 交叉模态校准融合
    作用：在进入局部融合之前，先在通道维度上“统筹”三个模态的信息。
    原理：类似于 SE-Block，但是是 Joint SE。
         它看着 T1+T2+FLAIR 的全局信息，来决定 T1 该保留哪些通道，T2 该保留哪些通道。
    """
    def __init__(self, in_channels, num_heads=4, reduction_ratio=4):
        super().__init__()
        
        # --- A. 全局统筹模块 (Global Calibration) ---
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        
        # 输入是 3 * C (三个模态的全局向量拼在一起)
        # 输出是 3 * C (三个模态各自的通道权重)
        self.calibration_fc = nn.Sequential(
            nn.Linear(in_channels * 3, (in_channels * 3) // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear((in_channels * 3) // reduction_ratio, in_channels * 3, bias=False),
            nn.Sigmoid()
        )
        
        # --- B. 局部融合模块 (保留原版 SGAF 的核心逻辑) ---
        # 1. 局部空间门控 (依然保留，处理局部特征)
        self.spatial_gate_t1 = nn.Sequential(nn.Conv3d(in_channels, in_channels//4, 1), nn.ReLU(), nn.Conv3d(in_channels//4, 1, 1), nn.Sigmoid())
        self.spatial_gate_t2 = nn.Sequential(nn.Conv3d(in_channels, in_channels//4, 1), nn.ReLU(), nn.Conv3d(in_channels//4, 1, 1), nn.Sigmoid())
        self.spatial_gate_flair = nn.Sequential(nn.Conv3d(in_channels, in_channels//4, 1), nn.ReLU(), nn.Conv3d(in_channels//4, 1, 1), nn.Sigmoid())
        
        # 2. 共识 Query 生成
        self.query_gen = nn.Sequential(
            nn.Conv3d(in_channels * 3, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm3d(in_channels), nn.ReLU(inplace=True)
        )
        
        # 3. 注意力块
        self.attention_block = Light_CrossAttention_Block_Symmetric(
            in_channels=in_channels,
            num_heads=num_heads,
            reduction_ratio=reduction_ratio
        )

    def forward(self, x_t1, x_t2, x_flair):
        b, c, _, _, _ = x_t1.size()
        
        # --- Step 1: 全局统筹 (Calibration) ---
        # 提取全局描述符
        v_t1 = self.avg_pool(x_t1).view(b, c)
        v_t2 = self.avg_pool(x_t2).view(b, c)
        v_flair = self.avg_pool(x_flair).view(b, c)
        
        # 拼接 -> 联合推断 -> 生成权重
        v_cat = torch.cat([v_t1, v_t2, v_flair], dim=1) # [B, 3C]
        weights = self.calibration_fc(v_cat)            # [B, 3C]
        
        # 分割权重
        w_t1_calib, w_t2_calib, w_flair_calib = torch.split(weights, c, dim=1)
        
        # 应用校准 (扩展维度以匹配 5D 张量)
        x_t1_c = x_t1 * w_t1_calib.view(b, c, 1, 1, 1)
        x_t2_c = x_t2 * w_t2_calib.view(b, c, 1, 1, 1)
        x_flair_c = x_flair * w_flair_calib.view(b, c, 1, 1, 1)
        
        # --- Step 2: 局部融合 (SGAF 逻辑) ---
        # 使用校准后的特征进行原本的 Symmetric Fusion
        
        # 空间门控
        g_t1 = self.spatial_gate_t1(x_t1_c)
        g_t2 = self.spatial_gate_t2(x_t2_c)
        g_flair = self.spatial_gate_flair(x_flair_c)
        
        x_t1_final = x_t1_c * g_t1
        x_t2_final = x_t2_c * g_t2
        x_flair_final = x_flair_c * g_flair
        
        x_all = torch.cat([x_t1_final, x_t2_final, x_flair_final], dim=1)
        
        # 共识 Query
        x_q = self.query_gen(x_all)
        
        # 注意力融合
        x_fused = self.attention_block(x_q=x_q, x_kv_concat=x_all)
        
        return x_fused, {'global_weight_t2': w_t2_calib}



class LayerNorm(nn.Module):
    """ 支持 Channels First 的 LayerNorm (保持不变) """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]
            return x

# --- [创新模块] 轴向管状提取分支 ---
class AxialTubularExcitation(nn.Module):
    def __init__(self, dim, kernel_size=7):
        super().__init__()
        pad = (kernel_size - 1) // 2
        
        # 深度方向
        self.conv_d = nn.Conv3d(dim, dim, kernel_size=(kernel_size, 1, 1), 
                                padding=(pad, 0, 0), groups=dim)
        # 高度方向
        self.conv_h = nn.Conv3d(dim, dim, kernel_size=(1, kernel_size, 1), 
                                padding=(0, pad, 0), groups=dim)
        # 宽度方向
        self.conv_w = nn.Conv3d(dim, dim, kernel_size=(1, 1, kernel_size), 
                                padding=(0, 0, pad), groups=dim)
        
        self.act = nn.GELU()
        # 轻量级融合
        self.fusion = nn.Conv3d(dim * 3, dim, kernel_size=1)

    def forward(self, x):
        d = self.conv_d(x)
        h = self.conv_h(x)
        w = self.conv_w(x)
        # 融合三个轴向特征
        out = self.fusion(torch.cat([d, h, w], dim=1))
        return self.act(out)

# ===============================================================================
#   2. 轴向增强型 ResBlock (Axial-ResBlock)
#   在标准 ResBlock 的第二个卷积处并联一个 Axial 分支
# ===============================================================================
class AxialResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        
        # Conv 1: 负责改变通道数或下采样 (Standard)
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # Conv 2: 负责特征提炼 (Standard)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
        
        # [创新点] Branch 2: 轴向增强分支
        # 并联在 conv2 旁边，专门负责抓取管状结构
        self.axial_branch = AxialTubularExcitation(out_channels, kernel_size=7)
        # 这里的 BN 是给 axial 分支用的
        self.bn_axial = nn.BatchNorm3d(out_channels)

        # Shortcut (残差连接)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_channels)
            )

    def forward(self, x):
        residual = self.shortcut(x)
        
        # 第一层卷积
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        # 第二层卷积 (双流: 3x3 局部 + 7x1 轴向)
        feat_local = self.bn2(self.conv2(out))
        feat_axial = self.bn_axial(self.axial_branch(out))
        
        # 融合 (相加)
        out = feat_local + feat_axial
        
        # 加上残差
        out += residual
        out = self.relu(out)
        return out

# ===============================================================================
#   3. 轴向增强型编码器 (AxialEnhancedEncoder3D)
#   结构模仿原始 ModalEncoder3D，但替换 Block
# ===============================================================================
class AxialEnhancedEncoder3D(nn.Module):
    def __init__(self, in_ch, channels):
        super().__init__()
        # channels e.g., [32, 64, 128, 256]
        
        # Stage 0 (Stem)
        self.stem = nn.Sequential(
            nn.Conv3d(in_ch, channels[0], kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(channels[0]),
            nn.ReLU(inplace=True)
        )
        
        # Stage 1 (保持分辨率)
        self.layer1 = AxialResBlock(channels[0], channels[0], stride=1)
        
        # Stage 2 (下采样 -> C1)
        self.layer2 = AxialResBlock(channels[0], channels[1], stride=2)
        
        # Stage 3 (下采样 -> C2)
        self.layer3 = AxialResBlock(channels[1], channels[2], stride=2)
        
        # Stage 4 (下采样 -> C3)
        self.layer4 = AxialResBlock(channels[2], channels[3], stride=2)

    def forward(self, x):
        x0 = self.stem(x)     # [B, 32, D, H, W]
        x1 = self.layer1(x0)  # [B, 32, D, H, W] (Level 0)
        x2 = self.layer2(x1)  # [B, 64, D/2, H/2, W/2] (Level 1)
        x3 = self.layer3(x2)  # [B, 128, D/4, H/4, W/4] (Level 2)
        x4 = self.layer4(x3)  # [B, 256, D/8, H/8, W/8] (Level 3)
        
        # 注意：这里返回的是 x1, x2, x3, x4，对应原代码的逻辑
        return [x1, x2, x3, x4]


class StripPooling3D(nn.Module):
    """
    [Bottleneck] 3D 条纹池化模块
    
    与 Encoder 的 Axial Convolution 形成呼应：
    Encoder 关注局部的轴向特征，Bottleneck 关注全局的轴向长依赖。
    非常适合细长的 EPVS 结构。
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        mid_channels = out_channels // 2
        
        # 1. 深度轴 (Depth) 池化: 聚合 H, W -> 保留 D
        self.pool_d = nn.AdaptiveAvgPool3d((None, 1, 1))
        # 2. 高度轴 (Height) 池化: 聚合 D, W -> 保留 H
        self.pool_h = nn.AdaptiveAvgPool3d((1, None, 1))
        # 3. 宽度轴 (Width) 池化: 聚合 D, H -> 保留 W
        self.pool_w = nn.AdaptiveAvgPool3d((1, 1, None))
        
        # 1x1 卷积用于特征变换
        self.conv_d = nn.Conv3d(in_channels, mid_channels, 3, padding=1, bias=False)
        self.conv_h = nn.Conv3d(in_channels, mid_channels, 3, padding=1, bias=False)
        self.conv_w = nn.Conv3d(in_channels, mid_channels, 3, padding=1, bias=False)
        
        self.fusion_conv = nn.Sequential(
            nn.Conv3d(mid_channels * 3, out_channels, 1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # x: [B, C, D, H, W]
        b, c, d, h, w = x.shape
        
        # 1. 沿三个轴分别池化，捕捉长距离依赖
        # 比如 pool_d 输出 [B, C, D, 1, 1]，它知道"沿着深度方向，哪里有信号"
        p_d = self.pool_d(x)
        p_h = self.pool_h(x)
        p_w = self.pool_w(x)
        
        # 2. 卷积处理
        p_d = F.interpolate(self.conv_d(p_d), size=(d, h, w), mode='trilinear', align_corners=True)
        p_h = F.interpolate(self.conv_h(p_h), size=(d, h, w), mode='trilinear', align_corners=True)
        p_w = F.interpolate(self.conv_w(p_w), size=(d, h, w), mode='trilinear', align_corners=True)
        
        # 3. 融合：将三个方向的长距离信息叠加
        # 这样，网络不仅知道"哪里有点"，还知道"这个点属于哪条长线"
        out = self.fusion_conv(torch.cat([p_d, p_h, p_w], dim=1))
        
        # 4. 残差连接 (注入原始特征)
        # 注意：这里我们通常希望加强原始特征
        return out + x
# --- 3. 主网络 ---

class ThreeEncoder_Calibration_Net(nn.Module):
    def __init__(self, n_classes, base_c=32, n_levels=4):
        super().__init__()
        self.n_classes = n_classes; self.n_levels = n_levels
        modal_channels = [base_c * (2**i) for i in range(n_levels)]
        
        self.encoder_t1 = ModalEncoder3D(in_ch=1, channels=modal_channels)
        self.encoder_t2 = ModalEncoder3D(in_ch=1, channels=modal_channels)
        self.encoder_flair = ModalEncoder3D(in_ch=1, channels=modal_channels)

        # L1, L2: 使用原版 FAF (保留物理特性)
        # self.fuse1 = FrequencyAwareFusion(modal_channels[0]) 
        # self.fuse2 = FrequencyAwareFusion(modal_channels[1]) 
        self.fuse1 = FrequencyAwareFusion_WithMax(modal_channels[0]) 
        self.fuse2 = FrequencyAwareFusion_WithMax(modal_channels[1]) 
        # L3, L4: 使用新的 CMCF (增加全局统筹)
        self.fuse3 = CrossModalCalibrationFusion(modal_channels[2], num_heads=4)
        self.fuse4 = CrossModalCalibrationFusion(modal_channels[3], num_heads=8)
        
        self.transformer_block = TransformerBlock(modal_channels[3], modal_channels[3] * 2, 8)
        
        self.decoders = nn.ModuleList()
        for level in reversed(range(n_levels - 1)):
            c_up = modal_channels[level+1]; c_skip = modal_channels[level]
            self.decoders.append(Up(c_up + c_skip, c_skip)) 

        self.outc = nn.Conv3d(base_c, n_classes, kernel_size=1)

    def forward(self, x_t1, x_t2, x_flair):
        fused_skips = []
        t1_feats = self.encoder_t1(x_t1); t2_feats = self.encoder_t2(x_t2); flair_feats = self.encoder_flair(x_flair)

        # 浅层：物理融合 (原版)
        x1_fused, _ = self.fuse1(t1_feats[0], t2_feats[0], flair_feats[0])
        fused_skips.append(x1_fused)

        x2_fused, _ = self.fuse2(t1_feats[1], t2_feats[1], flair_feats[1])
        fused_skips.append(x2_fused)
        
        # 深层：统筹校准融合 (新版)
        x3_fused, _ = self.fuse3(t1_feats[2], t2_feats[2], flair_feats[2])
        fused_skips.append(x3_fused)
        
        x4_fused, _ = self.fuse4(t1_feats[3], t2_feats[3], flair_feats[3])
        
        # Bottleneck
        x_bot = x4_fused
        b, c, d, h, w = x_bot.shape
        x_flat = x_bot.flatten(2).transpose(1, 2) 
        x_trans = self.transformer_block(x_flat)
        x_bot_trans = x_trans.transpose(1, 2).view(b, c, d, h, w)
        
        x = x_bot_trans
        for i in range(self.n_levels - 1):
            skip = fused_skips.pop()
            x = self.decoders[i](x, skip)

        return self.outc(x)



class ThreeEncoder_Calibration_Net_again1(nn.Module):
    def __init__(self, n_classes, base_c=32, n_levels=4):
        super().__init__()
        self.n_classes = n_classes
        modal_channels = [base_c * (2**i) for i in range(n_levels)]
        
        # --- Encoders ---
        self.encoder_t1 = ModalEncoder3D(in_ch=1, channels=modal_channels)
        self.encoder_t2 = ModalEncoder3D(in_ch=1, channels=modal_channels)
        self.encoder_flair = ModalEncoder3D(in_ch=1, channels=modal_channels)

        # --- Fusion (Encoder) ---
        # 浅层：使用 FrequencyAwareFusion_Clean (之前给你的纯 AvgPool 版本)
        self.fuse1 = FrequencyAwareFusion(modal_channels[0]) 
        self.fuse2 = FrequencyAwareFusion(modal_channels[1]) 
        # 深层：使用 Calibration Fusion
        self.fuse3 = CrossModalCalibrationFusion(modal_channels[2], num_heads=4)
        self.fuse4 = CrossModalCalibrationFusion(modal_channels[3], num_heads=8)
        
        # --- Bottleneck ---
        self.transformer_block = TransformerBlock(modal_channels[3], modal_channels[3] * 2, 8)
        
        # --- Decoders (SGCB - New) ---
        self.decoders = nn.ModuleList()
        for level in reversed(range(n_levels - 1)):
            c_up = modal_channels[level+1]
            c_skip = modal_channels[level]
            
            self.decoders.append(
                DilatedContextRefinementBlock(in_channels=c_up, out_channels=c_skip)
            )

        self.outc = nn.Conv3d(modal_channels[0], n_classes, kernel_size=1)

    def forward(self, x_t1, x_t2, x_flair):
        fused_skips = []
        
        # Encoding
        t1 = self.encoder_t1(x_t1); t2 = self.encoder_t2(x_t2); flair = self.encoder_flair(x_flair)

        # Fusion
        x1, _ = self.fuse1(t1[0], t2[0], flair[0]); fused_skips.append(x1)
        x2, _ = self.fuse2(t1[1], t2[1], flair[1]); fused_skips.append(x2)
        x3, _ = self.fuse3(t1[2], t2[2], flair[2]); fused_skips.append(x3)
        x4, _ = self.fuse4(t1[3], t2[3], flair[3])
        
        # Bottleneck
        b, c, d, h, w = x4.shape
        x_trans = self.transformer_block(x4.flatten(2).transpose(1, 2)).transpose(1, 2).view(b, c, d, h, w)
        x = x_trans
        
        # Decoding with SGCB
        # 注意 SGCB 的参数顺序: forward(x_deep, x_skip)
        
        skip3 = fused_skips.pop()
        x = self.decoders[0](x, skip3) 
        
        skip2 = fused_skips.pop()
        x = self.decoders[1](x, skip2)
        
        skip1 = fused_skips.pop()
        x = self.decoders[2](x, skip1)

        return self.outc(x)
    

class ThreeEncoder_GeometryAware_Net(nn.Module):

    def __init__(self, n_classes, base_c=32, n_levels=4):
        super().__init__()
        self.n_classes = n_classes
        modal_channels = [base_c * (2**i) for i in range(n_levels)]
        
        # --- 1. Encoder: Micro-Geometry Perception ---
        # 使用 Axial ResNet，在浅层和中层提取"局部的管状片段"
        self.encoder_t1 = AxialEnhancedEncoder3D(in_ch=1, channels=modal_channels)
        self.encoder_t2 = AxialEnhancedEncoder3D(in_ch=1, channels=modal_channels)
        self.encoder_flair = AxialEnhancedEncoder3D(in_ch=1, channels=modal_channels)

        # --- 2. Fusion: Texture/Frequency Perception ---
        # 负责处理模态间的噪声差异 (稳健版)
        self.fuse1 = FrequencyAwareFusion(modal_channels[0]) 
        self.fuse2 = FrequencyAwareFusion(modal_channels[1]) 
        self.fuse3 = CrossModalCalibrationFusion(modal_channels[2])
        self.fuse4 = CrossModalCalibrationFusion(modal_channels[3])
        
        # --- 3. Bottleneck: Macro-Geometry Aggregation ---
        # [关键修改] 使用 Strip Pooling 替代 Transformer
        # 将 Encoder 提取的"局部片段" 沿着 D/H/W 轴聚合，形成"全局的长距离连接"
        # 从而确认这些片段是否构成了血管/间隙网络
        self.bottleneck = StripPooling3D(in_channels=modal_channels[3], out_channels=modal_channels[3])
        
        # --- 4. Decoder: Restoration ---
        # 保持最简，专注于恢复分辨率
        self.decoders = nn.ModuleList()
        for level in reversed(range(n_levels - 1)):
            self.decoders.append(Up(modal_channels[level+1] + modal_channels[level], modal_channels[level]))

        self.outc = nn.Conv3d(modal_channels[0], n_classes, kernel_size=1)

    def forward(self, x_t1, x_t2, x_flair):
        # Encoding
        t1 = self.encoder_t1(x_t1); t2 = self.encoder_t2(x_t2); flair = self.encoder_flair(x_flair)

        # Fusion
        fused_skips = []
        x1, _ = self.fuse1(t1[0], t2[0], flair[0]); fused_skips.append(x1)
        x2, _ = self.fuse2(t1[1], t2[1], flair[1]); fused_skips.append(x2)
        x3, _ = self.fuse3(t1[2], t2[2], flair[2]); fused_skips.append(x3)
        x4, _ = self.fuse4(t1[3], t2[3], flair[3])
        
        # Bottleneck (Strip Pooling)
        # x4: [B, 256, D/8, H/8, W/8] -> [B, 256, D/8, H/8, W/8] (Enhanced with Global Axial Context)
        x = self.bottleneck(x4)
        
        # Decoding
        skip3 = fused_skips.pop(); x = self.decoders[0](x, skip3) 
        skip2 = fused_skips.pop(); x = self.decoders[1](x, skip2)
        skip1 = fused_skips.pop(); x = self.decoders[2](x, skip1)

        return self.outc(x)
    

# ... 引入其他模块 ...

class ThreeEncoder_CSSD_Net(nn.Module):
    def __init__(self, n_classes, base_c=32, n_levels=4):
        super().__init__()
        modal_channels = [base_c * (2**i) for i in range(n_levels)]
        
        # 1. Encoder (保留昨天的改进：Axial-Enhanced ResNet)
        self.encoder_t1 = AxialEnhancedEncoder3D(in_ch=1, channels=modal_channels)
        self.encoder_t2 = AxialEnhancedEncoder3D(in_ch=1, channels=modal_channels)
        self.encoder_flair = AxialEnhancedEncoder3D(in_ch=1, channels=modal_channels)

        # 2. Fusion (保留行而有效的版本)
        self.fuse1 = FrequencyAwareFusion(modal_channels[0]) 
        self.fuse2 = FrequencyAwareFusion(modal_channels[1]) 
        self.fuse3 = CrossModalCalibrationFusion(modal_channels[2])
        self.fuse4 = CrossModalCalibrationFusion(modal_channels[3])
        
        # Bottleneck (去掉 Transformer，直接用 Conv 或者 PPM，这里用最稳的 Conv)
        # 为了配合 PixelShuffle，我们确保 Bottleneck 输出维度正确
        self.bottleneck = nn.Sequential(
            nn.Conv3d(modal_channels[3], modal_channels[3], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(modal_channels[3]),
            nn.ReLU(True)
        )
        
        # 3. Decoder: [关键修改] 使用 CSSD
        self.decoders = nn.ModuleList()
        
        # L3 -> L2
        self.decoders.append(CSSD_Up(
            in_channels=modal_channels[3], 
            skip_channels=modal_channels[2], 
            out_channels=modal_channels[2]
        ))
        
        # L2 -> L1
        self.decoders.append(CSSD_Up(
            in_channels=modal_channels[2], 
            skip_channels=modal_channels[1], 
            out_channels=modal_channels[1]
        ))
        
        # L1 -> L0
        self.decoders.append(CSSD_Up(
            in_channels=modal_channels[1], 
            skip_channels=modal_channels[0], 
            out_channels=modal_channels[0]
        ))

        self.outc = nn.Conv3d(modal_channels[0], n_classes, kernel_size=1)

    def forward(self, x_t1, x_t2, x_flair):
        # Encoding
        t1 = self.encoder_t1(x_t1); t2 = self.encoder_t2(x_t2); flair = self.encoder_flair(x_flair)

        # Fusion
        fused_skips = []
        x1, _ = self.fuse1(t1[0], t2[0], flair[0]); fused_skips.append(x1)
        x2, _ = self.fuse2(t1[1], t2[1], flair[1]); fused_skips.append(x2)
        x3, _ = self.fuse3(t1[2], t2[2], flair[2]); fused_skips.append(x3)
        x4, _ = self.fuse4(t1[3], t2[3], flair[3])
        
        # Bottleneck
        x = self.bottleneck(x4)
        
        # Decoding
        skip3 = fused_skips.pop(); x = self.decoders[0](x, skip3) 
        skip2 = fused_skips.pop(); x = self.decoders[1](x, skip2)
        skip1 = fused_skips.pop(); x = self.decoders[2](x, skip1)

        return self.outc(x)
    

class ThreeEncoder_SelectiveAxial_Net(nn.Module):
    def __init__(self, n_classes, base_c=32, n_levels=4):
        super().__init__()
        modal_channels = [base_c * (2**i) for i in range(n_levels)]
        
        # 1. Encoder: 选择性轴向增强 (针对小目标优化)
        self.encoder_t1 = SelectiveAxialEncoder3D(in_ch=1, channels=modal_channels)
        self.encoder_t2 = SelectiveAxialEncoder3D(in_ch=1, channels=modal_channels)
        self.encoder_flair = SelectiveAxialEncoder3D(in_ch=1, channels=modal_channels)

        # 2. Fusion: 稳健的频率和校准
        self.fuse1 = FrequencyAwareFusion(modal_channels[0]) 
        self.fuse2 = FrequencyAwareFusion(modal_channels[1]) 
        self.fuse3 = CrossModalCalibrationFusion(modal_channels[2])
        self.fuse4 = CrossModalCalibrationFusion(modal_channels[3])
        
        # Bottleneck (Strip Pooling 配合 Encoder 的 Axial 特性效果最好)
        # 如果你想最简单，也可以换回 TransformerBlock
        self.bottleneck = StripPooling3D(modal_channels[3], modal_channels[3])
        
        # 3. Decoder: 原始 Up
        self.decoders = nn.ModuleList()
        for level in reversed(range(n_levels - 1)):
            c_deep = modal_channels[level+1]
            c_skip = modal_channels[level]
            self.decoders.append(Up(c_deep + c_skip, c_skip))
            
        # 4. Deep Supervision Heads
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

        # if self.training:
        #     return final, ds1, ds2
        return final
    

# ... 引入其他模块 ...

class ThreeEncoder_TransformerSelective_Net(nn.Module):
    def __init__(self, n_classes, base_c=32, n_levels=4):
        super().__init__()
        modal_channels = [base_c * (2**i) for i in range(n_levels)]
        
        # 1. Encoder: Transformer-Enhanced Selective Axial Encoder
        # 故事线：先用 Transformer 思想在底层去噪(Stem)，再用 SK-Net 思想在深层选形(Layers)
        self.encoder_t1 = TransformerEnhancedEncoder3D(in_ch=1, channels=modal_channels)
        self.encoder_t2 = TransformerEnhancedEncoder3D(in_ch=1, channels=modal_channels)
        self.encoder_flair = TransformerEnhancedEncoder3D(in_ch=1, channels=modal_channels)

        # 2. Fusion: 稳健的频率和校准 (保持不变)
        self.fuse1 = FrequencyAwareFusion(modal_channels[0]) 
        self.fuse2 = FrequencyAwareFusion(modal_channels[1]) 
        self.fuse3 = CrossModalCalibrationFusion(modal_channels[2])
        self.fuse4 = CrossModalCalibrationFusion(modal_channels[3])
        
        # Bottleneck (StripPooling 配合 Axial 效果好，建议保留)
        self.bottleneck = StripPooling3D(modal_channels[3], modal_channels[3])
        
        # 3. Decoder: 原始 Up (保持简单)
        self.decoders = nn.ModuleList()
        for level in reversed(range(n_levels - 1)):
            c_deep = modal_channels[level+1]
            c_skip = modal_channels[level]
            self.decoders.append(Up(c_deep + c_skip, c_skip))
            
        # 4. Deep Supervision
        self.ds2 = nn.Conv3d(modal_channels[2], n_classes, 1)
        self.ds1 = nn.Conv3d(modal_channels[1], n_classes, 1)
        self.outc = nn.Conv3d(modal_channels[0], n_classes, 1)

    def forward(self, x_t1, x_t2, x_flair):
        # 逻辑保持不变...
        t1 = self.encoder_t1(x_t1); t2 = self.encoder_t2(x_t2); flair = self.encoder_flair(x_flair)
        # ... (后续 Fusion, Decode 逻辑同前) ...
        fused_skips = []
        x1, _ = self.fuse1(t1[0], t2[0], flair[0]); fused_skips.append(x1)
        x2, _ = self.fuse2(t1[1], t2[1], flair[1]); fused_skips.append(x2)
        x3, _ = self.fuse3(t1[2], t2[2], flair[2]); fused_skips.append(x3)
        x4, _ = self.fuse4(t1[3], t2[3], flair[3])
        
        x = self.bottleneck(x4)
        
        skip3 = fused_skips.pop(); x = self.decoders[0](x, skip3); ds2 = self.ds2(x)
        skip2 = fused_skips.pop(); x = self.decoders[1](x, skip2); ds1 = self.ds1(x)
        skip1 = fused_skips.pop(); x = self.decoders[2](x, skip1); final = self.outc(x)

        # if self.training:
        #     return final, ds1, ds2
        return final