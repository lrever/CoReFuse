import torch
import torch.nn as nn
import torch.nn.functional as F

# ==============================================================================
# Part 1: 基础组件 (Basic Components)
# ==============================================================================

class DSConv3d(nn.Module):
    """ Depthwise Separable Convolution 3D """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
        super().__init__()
        self.depthwise = nn.Conv3d(in_channels, in_channels, kernel_size, stride, padding, 
                                   groups=in_channels, bias=bias)
        self.pointwise = nn.Conv3d(in_channels, out_channels, 1, 1, 0, bias=bias)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class DoubleConv(nn.Module):
    """ Standard U-Net Double Convolution """
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class Up(nn.Module):
    """ Decoder Up-sampling Block """
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        # 拼接后的通道数 = in_channels + skip_channels
        self.conv = DoubleConv(in_channels + skip_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # 处理可能的尺寸不匹配 (Padding)
        diffZ = x2.size(2) - x1.size(2)
        diffY = x2.size(3) - x1.size(3)
        diffX = x2.size(4) - x1.size(4)
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2,
                        diffZ // 2, diffZ - diffZ // 2])
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

# ==============================================================================
# Part 2: 编码器组件 (Encoder Components)
# ==============================================================================

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

class SelectiveFusion(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super().__init__()
        mid_channels = max(in_channels // reduction, 16)
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, mid_channels, bias=False),
            nn.ReLU(True),
            nn.Linear(mid_channels, in_channels * 2, bias=False)
        )

    def forward(self, x_local, x_axial):
        b, c, _, _, _ = x_local.size()
        s = self.avg_pool(x_local + x_axial).view(b, c)
        attn = F.softmax(self.fc(s).view(b, 2, c), dim=1)
        return x_local * attn[:, 0, :].view(b, c, 1, 1, 1) + x_axial * attn[:, 1, :].view(b, c, 1, 1, 1)

class HybridResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels); self.relu = nn.ReLU(True)
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

class DynamicEncoder(nn.Module):
    def __init__(self, in_ch, channels, opt_encoder=False):
        super().__init__()
        if opt_encoder: 
            self.stem = nn.Sequential(
                nn.Conv3d(in_ch, channels[0] // 2, 3, 1, 1, bias=False), nn.BatchNorm3d(channels[0] // 2), nn.ReLU(True),
                nn.Conv3d(channels[0] // 2, channels[0] // 2, 3, 1, 1, bias=False), nn.BatchNorm3d(channels[0] // 2), nn.ReLU(True),
                DSConv3d(channels[0] // 2, channels[0], 3, 1, 1, bias=False), nn.BatchNorm3d(channels[0]), nn.ReLU(True)
            )
            Block = HybridResBlock
        else: 
            self.stem = DoubleConv(in_ch, channels[0], stride=1)
            Block = DoubleConv
        
        self.layer1 = Block(channels[0], channels[0], stride=1)
        self.layer2 = Block(channels[0], channels[1], stride=2)
        self.layer3 = Block(channels[1], channels[2], stride=2)
        self.layer4 = Block(channels[2], channels[3], stride=2)

    def forward(self, x):
        x0 = self.stem(x)
        x1 = self.layer1(x0); x2 = self.layer2(x1); x3 = self.layer3(x2); x4 = self.layer4(x3)
        return [x1, x2, x3, x4]

# ==============================================================================
# Part 3: 融合组件 (Fusion Components - N Modality Ready)
# ==============================================================================

class FrequencyAwareFusion_UltraLite(nn.Module):
    """ N-Modality Supported Frequency Aware Fusion """
    def __init__(self, in_channels, num_modalities=3): # 默认3，但支持动态
        super().__init__()
        self.num_modalities = num_modalities
        self.low_pass = nn.AvgPool3d(3, 1, 1)
        total_in_channels = in_channels * num_modalities
        
        # Competitor
        self.competitor = nn.Sequential(
            nn.Conv3d(total_in_channels, total_in_channels, kernel_size=3, padding=1, groups=total_in_channels, bias=False),
            nn.Conv3d(total_in_channels, total_in_channels, kernel_size=1, groups=num_modalities, bias=False),
            nn.ReLU(True),
            nn.Conv3d(total_in_channels, num_modalities, kernel_size=1, bias=True),
            nn.Softmax(dim=1)
        )
        
        # Shared Gate
        self.shared_gate = nn.Sequential(
            DSConv3d(total_in_channels, in_channels//2, 3, 1, 1, bias=False),
            nn.ReLU(True),
            nn.Conv3d(in_channels//2, num_modalities, 1, bias=True),
            nn.Sigmoid()
        )
        
        # Final
        self.final = nn.Sequential(
            DSConv3d(in_channels*2, in_channels, 3, 1, 1, bias=False),
            nn.BatchNorm3d(in_channels), 
            nn.ReLU(True)
        )

    def forward(self, *features):
        # 兼容性处理：如果网络初始化时没有传 num_modalities，这里可能会出错
        # 假设 features 长度就是 num_modalities
        num_inputs = len(features)
        
        l_feats, h_feats = [], []
        for x in features:
            l = self.low_pass(x)
            l_feats.append(l); h_feats.append(x - l)
        
        # Low Freq
        low_cat = torch.cat(l_feats, dim=1)
        low_w = self.competitor(low_cat)
        ws = torch.split(low_w, 1, dim=1)
        l_fused = sum(feat * w for feat, w in zip(l_feats, ws))
        
        # High Freq
        high_cat = torch.cat(h_feats, dim=1)
        gates = self.shared_gate(high_cat)
        gs = torch.split(gates, 1, dim=1)
        h_fused = sum(feat * g for feat, g in zip(h_feats, gs))
        
        return self.final(torch.cat([l_fused, h_fused], dim=1)), {}

class Light_CrossAttention_Block_Symmetric_N(nn.Module):
    def __init__(self, in_channels, num_modalities=3, num_heads=4, reduction_ratio=4):
        super().__init__()
        self.embed_dim = in_channels
        total_channels = in_channels * num_modalities
        
        self.query_proj = nn.Conv3d(in_channels, self.embed_dim, kernel_size=1)
        self.key_proj = nn.Conv3d(total_channels, self.embed_dim, kernel_size=1)
        self.value_proj = nn.Conv3d(total_channels, self.embed_dim, kernel_size=1)
        
        self.kv_reduction = nn.Conv3d(total_channels, total_channels, kernel_size=reduction_ratio, stride=reduction_ratio, groups=total_channels)
        self.attn = nn.MultiheadAttention(self.embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(self.embed_dim); self.norm2 = nn.LayerNorm(self.embed_dim)
        self.mlp = nn.Sequential(nn.Linear(self.embed_dim, self.embed_dim * 4), nn.GELU(), nn.Linear(self.embed_dim * 4, self.embed_dim))

    def forward(self, x_q, x_kv_concat):
        B, C, D, H, W = x_q.shape
        q_feat = self.query_proj(x_q)
        Q = q_feat.flatten(2).transpose(1, 2)

        if all(s >= self.kv_reduction.stride[0] for s in (D, H, W)):
            all_features_reduced = self.kv_reduction(x_kv_concat)
        else:
            all_features_reduced = x_kv_concat
            
        k_feat = self.key_proj(all_features_reduced); v_feat = self.value_proj(all_features_reduced)
        K = k_feat.flatten(2).transpose(1, 2); V = v_feat.flatten(2).transpose(1, 2)
        
        attn_out, _ = self.attn(Q, K, V)
        x = self.norm(Q + attn_out)
        x_mlp = self.mlp(x); x = self.norm2(x + x_mlp)
        return x.transpose(1, 2).reshape(B, C, D, H, W) + x_q

class CrossModalCalibrationFusion(nn.Module):
    """ N-Modality Supported Calibration Fusion """
    def __init__(self, in_channels, num_modalities=3, num_heads=4, reduction_ratio=4):
        super().__init__()
        self.num_modalities = num_modalities
        self.in_channels = in_channels
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        total_channels = in_channels * num_modalities
        
        self.calibration_fc = nn.Sequential(
            nn.Linear(total_channels, total_channels // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(total_channels // reduction_ratio, total_channels, bias=False),
            nn.Sigmoid()
        )
        
        self.spatial_gates = nn.ModuleList([
            nn.Sequential(nn.Conv3d(in_channels, in_channels//4, 1), nn.ReLU(), nn.Conv3d(in_channels//4, 1, 1), nn.Sigmoid()) 
            for _ in range(num_modalities)
        ])
        
        self.query_gen = nn.Sequential(nn.Conv3d(total_channels, in_channels, kernel_size=1, bias=False), nn.BatchNorm3d(in_channels), nn.ReLU(inplace=True))
        
        self.attention_block = Light_CrossAttention_Block_Symmetric_N(in_channels, num_modalities, num_heads, reduction_ratio)

    def forward(self, *features):
        b, c, _, _, _ = features[0].size()
        vecs = [self.avg_pool(x).view(b, c) for x in features]
        v_cat = torch.cat(vecs, dim=1)
        weights = self.calibration_fc(v_cat)
        ws_calib = torch.split(weights, c, dim=1)
        
        feats_calib = [x * w.view(b, c, 1, 1, 1) for x, w in zip(features, ws_calib)]
        feats_final = [x_c * gate(x_c) for x_c, gate in zip(feats_calib, self.spatial_gates)]
        
        x_all = torch.cat(feats_final, dim=1)
        x_q = self.query_gen(x_all)
        x_fused = self.attention_block(x_q=x_q, x_kv_concat=x_all)
        
        return x_fused, {'global_weight': ws_calib}

# ==============================================================================
# Part 4: 瓶颈层 (Bottleneck)
# ==============================================================================

class StripPooling3D(nn.Module):
    def __init__(self, in_channels, out_channels=None):
        super().__init__()
        if out_channels is None: out_channels = in_channels
        self.channel_adjust = nn.Identity()
        if in_channels != out_channels:
            self.channel_adjust = nn.Sequential(nn.Conv3d(in_channels, out_channels, 1, bias=False), nn.BatchNorm3d(out_channels), nn.ReLU(True))
        
        mid_channels = out_channels // 2
        self.pool_d = nn.AdaptiveAvgPool3d((None, 1, 1))
        self.pool_h = nn.AdaptiveAvgPool3d((1, None, 1))
        self.pool_w = nn.AdaptiveAvgPool3d((1, 1, None))
        
        self.conv_d = nn.Conv3d(out_channels, mid_channels, kernel_size=(3, 1, 1), padding=(1, 0, 0), bias=False)
        self.conv_h = nn.Conv3d(out_channels, mid_channels, kernel_size=(1, 3, 1), padding=(0, 1, 0), bias=False)
        self.conv_w = nn.Conv3d(out_channels, mid_channels, kernel_size=(1, 1, 3), padding=(0, 0, 1), bias=False)
        
        self.fusion = nn.Sequential(nn.Conv3d(mid_channels * 3, out_channels, 1, bias=False), nn.Sigmoid())

    def forward(self, x):
        x = self.channel_adjust(x)
        d, h, w = x.shape[2:]
        p_d = F.interpolate(self.conv_d(self.pool_d(x)), size=(d,h,w), mode='trilinear', align_corners=False)
        p_h = F.interpolate(self.conv_h(self.pool_h(x)), size=(d,h,w), mode='trilinear', align_corners=False)
        p_w = F.interpolate(self.conv_w(self.pool_w(x)), size=(d,h,w), mode='trilinear', align_corners=False)
        attn = self.fusion(torch.cat([p_d, p_h, p_w], dim=1))
        return x + x * attn

# ==============================================================================
# Part 5: 主网络 (Main Architecture)
# ==============================================================================

class Ablation_NEncoder_Final_Net(nn.Module):
    def __init__(self, n_classes, num_modalities=3, base_c=32, n_levels=4, 
                 opt_encoder=True,        
                 opt_fusion_shallow=True, 
                 opt_fusion_deep=True,    
                 deep_sup=False):         
        super().__init__()
        self.num_modalities = num_modalities
        self.deep_sup = deep_sup
        modal_channels = [base_c * (2**i) for i in range(n_levels)]
        
        # 1. Encoders
        self.encoders = nn.ModuleList([
            DynamicEncoder(1, modal_channels, opt_encoder) for _ in range(num_modalities)
        ])

        # 2. Fusion
        # 传递 num_modalities 参数给 Fusion 模块
        if opt_fusion_shallow:
            self.fuse1 = FrequencyAwareFusion_UltraLite(modal_channels[0], num_modalities)
            self.fuse2 = FrequencyAwareFusion_UltraLite(modal_channels[1], num_modalities)
        else:
            # NaiveFusion 需要自己定义，这里假设如果 False 则使用简单的 Concat+Conv
            # 为了代码完整性，这里用一个简单的替代
            self.fuse1 = nn.Identity() # 占位，实际需要 NaiveFusion 实现
            self.fuse2 = nn.Identity()
            print("Warning: NaiveFusion not implemented in this script, using Identity (Will Fail if opt_fusion_shallow=False)")

        if opt_fusion_deep:
            self.fuse3 = CrossModalCalibrationFusion(modal_channels[2], num_modalities)
            self.fuse4 = CrossModalCalibrationFusion(modal_channels[3], num_modalities)
        else:
            self.fuse3 = nn.Identity()
            self.fuse4 = nn.Identity()

        # 3. Bottleneck
        if opt_encoder:
            self.bottleneck = StripPooling3D(modal_channels[3], modal_channels[3])
        else:
            # 简单的 Conv 瓶颈
            self.bottleneck = DoubleConv(modal_channels[3], modal_channels[3])
            
        # 4. Decoder
        self.decoders = nn.ModuleList()
        # Up(in_channels, skip_channels, out_channels)
        self.decoders.append(Up(modal_channels[3], modal_channels[2], modal_channels[2]))
        self.decoders.append(Up(modal_channels[2], modal_channels[1], modal_channels[1]))
        self.decoders.append(Up(modal_channels[1], modal_channels[0], modal_channels[0]))

        # 5. Output Heads
        self.outc = nn.Conv3d(modal_channels[0], n_classes, 1)
        if self.deep_sup:
            self.ds2 = nn.Conv3d(modal_channels[2], n_classes, 1)
            self.ds1 = nn.Conv3d(modal_channels[1], n_classes, 1)

    def forward(self, x):
        # 1. Input Handling
        if isinstance(x, torch.Tensor):
            if x.shape[1] != self.num_modalities:
                raise ValueError(f"Input channels {x.shape[1]} != initialized modalities {self.num_modalities}")
            inputs = torch.chunk(x, self.num_modalities, dim=1)
        else:
            inputs = x

        # 2. Encoding
        enc_outputs = []
        for i, encoder in enumerate(self.encoders):
            enc_outputs.append(encoder(inputs[i]))

        # 3. Fusion
        # Transpose: [Modality][Level] -> [Level][Modality]
        levels_features = list(zip(*enc_outputs))
        
        fused_skips = []
        # 使用 * 解包参数传递给 Fusion
        x1, _ = self.fuse1(*levels_features[0]); fused_skips.append(x1)
        x2, _ = self.fuse2(*levels_features[1]); fused_skips.append(x2)
        x3, _ = self.fuse3(*levels_features[2]); fused_skips.append(x3)
        x4, _ = self.fuse4(*levels_features[3])
        
        # 4. Bottleneck & Decoder
        x = self.bottleneck(x4)
        
        skip3 = fused_skips.pop(); x = self.decoders[0](x, skip3); ds2 = self.ds2(x) if self.deep_sup else None
        skip2 = fused_skips.pop(); x = self.decoders[1](x, skip2); ds1 = self.ds1(x) if self.deep_sup else None
        skip1 = fused_skips.pop(); x = self.decoders[2](x, skip1); final = self.outc(x)

        if self.training and self.deep_sup:
            return final, ds1, ds2
        return final

# ==============================================================================
# Part 6: 测试运行 (Sanity Check)
# ==============================================================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 假设有 2 个模态 (FLAIR, T1)，3 个类别 (Bg, WMH, Other)
    num_modalities = 2
    model = Ablation_NEncoder_Final_Net(
        n_classes=3, 
        num_modalities=num_modalities, 
        base_c=32, 
        deep_sup=False
        
    ).to(device)
    
    # 输入: (Batch, Channels=2, D, H, W)
    dummy_input = torch.randn(2, num_modalities, 128, 128, 48).to(device)
    
    print(f"Model created. Input shape: {dummy_input.shape}")
    
    output = model(dummy_input)
    
    if isinstance(output, tuple):
        print(f"Output shapes (Deep Sup): Final={output[0].shape}, DS1={output[1].shape}, DS2={output[2].shape}")
    else:
        print(f"Output shape: {output.shape}")
    
    print("Forward pass successful!")