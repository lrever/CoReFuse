import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

# ==============================================================================
# Part 1: 基础组件 (保持不变，直接复用)
# ==============================================================================

class DSConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
        super().__init__()
        self.depthwise = nn.Conv3d(in_channels, in_channels, kernel_size, stride, padding, 
                                   groups=in_channels, bias=bias)
        self.pointwise = nn.Conv3d(in_channels, out_channels, 1, bias=bias)
    def forward(self, x):
        return self.pointwise(self.depthwise(x))

class Attention_Gate(nn.Module):
    """ Decoder Skip Connection Gate """
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(nn.Conv3d(F_g, F_int, 1), nn.InstanceNorm3d(F_int, affine=True))
        self.W_x = nn.Sequential(nn.Conv3d(F_l, F_int, 1), nn.InstanceNorm3d(F_int, affine=True))
        self.psi = nn.Sequential(nn.Conv3d(F_int, 1, 1), nn.InstanceNorm3d(1, affine=True), nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)
    def forward(self, g, x):
        g1 = self.W_g(g); x1 = self.W_x(x)
        psi = self.psi(self.relu(g1 + x1))
        return x * psi

class Up_AG(nn.Module):
    """ Decoder Block with Attention Gate """
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.ag = Attention_Gate(F_g=in_ch, F_l=skip_ch, F_int=skip_ch // 2)
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch + skip_ch, out_ch, 3, padding=1, bias=False), nn.InstanceNorm3d(out_ch, affine=True), nn.GELU(),
            nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False), nn.InstanceNorm3d(out_ch, affine=True), nn.GELU()
        )
    def forward(self, x, skip):
        x = self.up(x)
        diffD = skip.size(2) - x.size(2); diffH = skip.size(3) - x.size(3); diffW = skip.size(4) - x.size(4)
        if diffD != 0 or diffH != 0 or diffW != 0:
            x = F.pad(x, [diffW // 2, diffW - diffW // 2, diffH // 2, diffH - diffH // 2, diffD // 2, diffD - diffD // 2])
        skip = self.ag(g=x, x=skip)
        return self.conv(torch.cat([x, skip], dim=1))

class CoordAtt3D(nn.Module):
    """ Bottleneck """
    def __init__(self, inp, reduction=32):
        super().__init__()
        self.pool_d = nn.AdaptiveAvgPool3d((None, 1, 1))
        self.pool_h = nn.AdaptiveAvgPool3d((1, None, 1))
        self.pool_w = nn.AdaptiveAvgPool3d((1, 1, None))
        mip = max(8, inp // reduction)
        self.conv1 = nn.Conv3d(inp, mip, 1); self.bn1 = nn.InstanceNorm3d(mip, affine=True); self.act = nn.Hardswish()
        self.conv_d = nn.Conv3d(mip, inp, 1); self.conv_h = nn.Conv3d(mip, inp, 1); self.conv_w = nn.Conv3d(mip, inp, 1)
    def forward(self, x):
        identity = x
        x_d = self.pool_d(x); x_h = self.pool_h(x); x_w = self.pool_w(x)
        y_d = self.conv_d(self.act(self.bn1(self.conv1(x_d)))).sigmoid()
        y_h = self.conv_h(self.act(self.bn1(self.conv1(x_h)))).sigmoid()
        y_w = self.conv_w(self.act(self.bn1(self.conv1(x_w)))).sigmoid()
        return identity * y_d * y_h * y_w

# ==============================================================================
# Part 2: 动态化融合模块 (Dynamic Fusion Modules) - 核心修改区域
# ==============================================================================

class AdaptiveGatedStem_Dynamic(nn.Module):
    """ [动态] 自适应门控干 """
    def __init__(self, num_modalities, in_channels=1, base_c=16):
        super().__init__()
        self.num_modalities = num_modalities
        
        # 1. 上下文提取器: 输入通道 = 模态数 * 单模态通道
        self.ctx_conv = nn.Sequential(
            nn.Conv3d(num_modalities * in_channels, base_c, 3, padding=1, bias=False),
            nn.InstanceNorm3d(base_c, affine=True),
            nn.GELU()
        )
        
        # 2. 动态生成每个模态的 Gate 和 Feature 变换
        # 使用 ModuleList 来存储 M 个分支
        self.transforms = nn.ModuleList([
            nn.Conv3d(base_c, in_channels, 1) for _ in range(num_modalities)
        ])
        self.gates = nn.ModuleList([
            nn.Sequential(nn.Conv3d(base_c, 1, 1), nn.Sigmoid()) for _ in range(num_modalities)
        ])

    def forward(self, x_list):
        # x_list 是一个包含 M 个 tensor 的列表
        # 1. 拼接提取上下文
        ctx = self.ctx_conv(torch.cat(x_list, dim=1))
        
        # 2. 对每个模态应用修正
        outs = []
        for i, x in enumerate(x_list):
            # 原始输入 + Gate * Transform(Context)
            out = x + self.gates[i](ctx) * self.transforms[i](ctx)
            outs.append(out)
            
        return outs

class FrequencyAwareFusion_Dynamic(nn.Module):
    """ [动态] 浅层融合 (Balanced) """
    def __init__(self, in_channels, num_modalities):
        super().__init__()
        self.num_modalities = num_modalities
        self.low_pass = nn.AvgPool3d(3, 1, 1)
        mid_channels = num_modalities * (in_channels // 2)
        total_in = in_channels * num_modalities
        
        # 竞争网络: Groups = num_modalities
        self.competitor = nn.Sequential(
            nn.Conv3d(total_in, mid_channels, kernel_size=3, padding=1, groups=num_modalities, bias=False), 
            nn.ReLU(True),
            nn.Conv3d(mid_channels, num_modalities, 1, bias=True), # 输出 M 个权重
            nn.Softmax(dim=1)
        )
        
        # 共享门控
        self.shared_gate = nn.Sequential(
            DSConv3d(total_in, in_channels//2, 3, 1, 1, bias=False),
            nn.ReLU(True),
            nn.Conv3d(in_channels//2, num_modalities, 1, bias=True), # 输出 M 个 Gate
            nn.Sigmoid()
        )
        
        # 最终融合
        self.final = nn.Sequential(
            nn.Conv3d(in_channels * 2, in_channels, kernel_size=3, padding=1, bias=False), # concat(low, high) -> out
            nn.InstanceNorm3d(in_channels, affine=True), nn.GELU()
        )

    def forward(self, x_list):
        # 分离高低频
        l_list = [self.low_pass(x) for x in x_list]
        h_list = [x - l for x, l in zip(x_list, l_list)]
        
        # Low Freq Fusion
        low_cat = torch.cat(l_list, dim=1)
        low_w = self.competitor(low_cat) # [B, M, D, H, W]
        # split weights: [w1, w2, ..., wm]
        ws = torch.split(low_w, 1, dim=1)
        
        l_fused = 0
        for l, w in zip(l_list, ws):
            l_fused += l * w
            
        # High Freq Fusion
        high_cat = torch.cat(h_list, dim=1)
        gates = self.shared_gate(high_cat)
        gs = torch.split(gates, 1, dim=1)
        
        h_fused = 0
        for h, g in zip(h_list, gs):
            h_fused += h * g
            
        return self.final(torch.cat([l_fused, h_fused], dim=1)), {}

class Light_CrossAttention_Block_Dynamic(nn.Module):
    """ [动态] Attention Block """
    def __init__(self, in_channels, num_modalities, num_heads=4, reduction_ratio=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = in_channels // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Q: 来自融合后的 query_gen (1个)
        self.q_proj = nn.Conv3d(in_channels, in_channels, 1, bias=False)
        
        # K, V: 来自拼接的所有模态 (M个)
        total_in = in_channels * num_modalities
        self.k_proj = nn.Conv3d(total_in, in_channels, 1, bias=False)
        self.v_proj = nn.Conv3d(total_in, in_channels, 1, bias=False)
        self.out_proj = nn.Conv3d(in_channels, in_channels, 1, bias=False)
        
        self.kv_reduction = nn.Identity()
        if reduction_ratio > 1:
            self.kv_reduction = nn.Conv3d(
                total_in, total_in,
                kernel_size=reduction_ratio, stride=reduction_ratio, 
                groups=total_in, bias=False
            )
        self.norm = nn.InstanceNorm3d(in_channels, affine=True)

    def forward(self, x_q, x_kv_concat):
        B, C, D, H, W = x_q.shape
        N_q = D * H * W
        
        q = self.q_proj(x_q).view(B, self.num_heads, self.head_dim, N_q)
        
        x_kv_reduced = self.kv_reduction(x_kv_concat)
        _, _, D_k, H_k, W_k = x_kv_reduced.shape
        N_k = D_k * H_k * W_k
        
        k = self.k_proj(x_kv_reduced).view(B, self.num_heads, self.head_dim, N_k)
        v = self.v_proj(x_kv_reduced).view(B, self.num_heads, self.head_dim, N_k)
        
        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(dim=-1)
        
        out = (attn @ v.transpose(-2, -1)).transpose(-2, -1)
        out = out.reshape(B, C, D, H, W)
        out = self.out_proj(out)
        
        return self.norm(x_q + out)

class CrossModalCalibrationFusion_Dynamic(nn.Module):
    """ [动态] 深层融合 """
    def __init__(self, in_channels, num_modalities, num_heads=4, reduction_ratio=4):
        super().__init__()
        self.num_modalities = num_modalities
        
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        total_in = in_channels * num_modalities
        
        # 1. Calibration
        self.calibration_fc = nn.Sequential(
            nn.Linear(total_in, total_in // 4, bias=False), 
            nn.ReLU(inplace=True),
            nn.Linear(total_in // 4, total_in, bias=False),
            nn.Sigmoid()
        )
        
        # 2. Spatial Gates (ModuleList for M modalities)
        self.spatial_gates = nn.ModuleList([
            nn.Sequential(nn.Conv3d(in_channels, in_channels//4, 1), nn.ReLU(), nn.Conv3d(in_channels//4, 1, 1), nn.Sigmoid())
            for _ in range(num_modalities)
        ])
        
        # 3. Query Generation
        self.query_gen = nn.Sequential(
            nn.Conv3d(total_in, in_channels, kernel_size=1, bias=False),
            nn.InstanceNorm3d(in_channels, affine=True), nn.GELU()
        )
        
        # 4. Attention
        self.attention_block = Light_CrossAttention_Block_Dynamic(
            in_channels, num_modalities, num_heads, reduction_ratio
        )

    def forward(self, x_list):
        b, c, _, _, _ = x_list[0].shape
        
        # Calibration
        # cat([pool(x1), pool(x2)...])
        v_cat = torch.cat([self.avg_pool(x) for x in x_list], dim=1).view(b, -1)
        weights = self.calibration_fc(v_cat)
        ws = torch.split(weights, c, dim=1) # [w1, w2, ...]
        
        calibrated_list = []
        for x, w in zip(x_list, ws):
            calibrated_list.append(x * w.view(b, c, 1, 1, 1))
            
        # Spatial Gating
        gated_list = []
        for x_c, gate_net in zip(calibrated_list, self.spatial_gates):
            g = gate_net(x_c)
            gated_list.append(x_c * g)
            
        x_all = torch.cat(gated_list, dim=1)
        
        # Attention
        x_q = self.query_gen(x_all)
        x_fused = self.attention_block(x_q=x_q, x_kv_concat=x_all)
        
        return x_fused, {}

# ==============================================================================
# Part 3: 编码器 (Encoder) - 复用之前的
# ==============================================================================
# 为了完整性，这里必须包含 Encoder 的定义，因为它被 ModuleList 调用
class SelectiveFusion(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super().__init__()
        mid_channels = max(in_channels // reduction, 16)
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(nn.Linear(in_channels, mid_channels, bias=False), nn.GELU(), nn.Linear(mid_channels, in_channels * 2, bias=False))
    def forward(self, x_local, x_axial):
        b, c, d, h, w = x_local.size()
        s = self.avg_pool(x_local + x_axial).view(b, c)
        attn = F.softmax(self.fc(s).view(b, 2, c), dim=1)
        return x_local * attn[:, 0, :].view(b, c, 1, 1, 1) + x_axial * attn[:, 1, :].view(b, c, 1, 1, 1)

class EnhancedHybridBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, local_kernel=3, axial_kernel=7, expansion=2):
        super().__init__()
        self.pre_conv = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.pre_conv = nn.Sequential(nn.Conv3d(in_channels, out_channels, 3, stride, 1, bias=False), nn.InstanceNorm3d(out_channels, affine=True), nn.GELU())
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(nn.Conv3d(in_channels, out_channels, 1, stride, bias=False), nn.InstanceNorm3d(out_channels, affine=True))
        
        mid_c = out_channels * expansion
        self.local_branch = nn.Sequential(
            nn.Conv3d(out_channels, mid_c, 1, bias=False), nn.InstanceNorm3d(mid_c, affine=True), nn.GELU(),
            nn.Conv3d(mid_c, mid_c, local_kernel, padding=local_kernel//2, groups=mid_c, bias=False), nn.InstanceNorm3d(mid_c, affine=True), nn.GELU(),
            nn.Conv3d(mid_c, out_channels, 1, bias=False), nn.InstanceNorm3d(out_channels, affine=True)
        )
        pad = axial_kernel // 2
        self.axial_branch = nn.Sequential(
             nn.Conv3d(out_channels, out_channels, (axial_kernel, 1, 1), padding=(pad, 0, 0), groups=out_channels, bias=False),
             nn.Conv3d(out_channels, out_channels, (1, axial_kernel, 1), padding=(0, pad, 0), groups=out_channels, bias=False),
             nn.Conv3d(out_channels, out_channels, (1, 1, axial_kernel), padding=(0, 0, pad), groups=out_channels, bias=False),
             nn.InstanceNorm3d(out_channels, affine=True), nn.GELU(),
             nn.Conv3d(out_channels, out_channels, 1, bias=False), nn.InstanceNorm3d(out_channels, affine=True)
        )
        self.selection = SelectiveFusion(out_channels)
        self.act_out = nn.GELU()
    def forward(self, x):
        res = self.shortcut(x); x = self.pre_conv(x)
        return self.act_out(self.selection(self.local_branch(x), self.axial_branch(x)) + res)

class DynamicEncoder(nn.Module):
    def __init__(self, in_ch, channels):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv3d(in_ch, channels[0] // 2, 3, 1, 1, bias=False), nn.InstanceNorm3d(channels[0] // 2, affine=True), nn.GELU(),
            nn.Conv3d(channels[0] // 2, channels[0] // 2, 3, 1, 1, bias=False), nn.InstanceNorm3d(channels[0] // 2, affine=True), nn.GELU(),
            DSConv3d(channels[0] // 2, channels[0], 3, 1, 1, bias=False), nn.InstanceNorm3d(channels[0], affine=True), nn.GELU()
        )
        self.layers = nn.ModuleList([
            EnhancedHybridBlock(channels[0], channels[0], stride=1),
            EnhancedHybridBlock(channels[0], channels[1], stride=2),
            EnhancedHybridBlock(channels[1], channels[2], stride=2),
            EnhancedHybridBlock(channels[2], channels[3], stride=2)
        ])
    def forward(self, x):
        x = self.stem(x); feats = []
        for layer in self.layers:
            x = checkpoint.checkpoint(layer, x, use_reentrant=False)
            feats.append(x)
        return feats

# ==============================================================================
# Part 4: 主网络 (Orchestrator)
# ==============================================================================

class Optimized_DynamicModal_Net(nn.Module):
    def __init__(self, n_classes=4, base_c=16, num_modalities=1, deep_sup=False):
        super().__init__()
        self.num_modalities = num_modalities
        self.deep_sup = deep_sup
        self.channels = [base_c * (2**i) for i in range(4)]
        
        # 1. 动态 Stem
        self.adaptive_stem = AdaptiveGatedStem_Dynamic(num_modalities, in_channels=1, base_c=base_c)
        
        # 2. 动态 Encoders (ModuleList)
        self.encoders = nn.ModuleList([
            DynamicEncoder(1, self.channels) for _ in range(num_modalities)
        ])

        # 3. 动态 Fusions
        self.fuse1 = FrequencyAwareFusion_Dynamic(self.channels[0], num_modalities)
        self.fuse2 = FrequencyAwareFusion_Dynamic(self.channels[1], num_modalities)
        self.fuse3 = CrossModalCalibrationFusion_Dynamic(self.channels[2], num_modalities, reduction_ratio=4)
        self.fuse4 = CrossModalCalibrationFusion_Dynamic(self.channels[3], num_modalities, reduction_ratio=4)
        
        # 4. Bottleneck
        self.bottleneck = CoordAtt3D(self.channels[3], reduction=16)

        # 5. Decoders
        self.decoders = nn.ModuleList([
            Up_AG(self.channels[3], self.channels[2], self.channels[2]),
            Up_AG(self.channels[2], self.channels[1], self.channels[1]),
            Up_AG(self.channels[1], self.channels[0], self.channels[0])
        ])

        # 6. Heads
        self.outc = nn.Conv3d(self.channels[0], n_classes, 1)
        if self.deep_sup:
            self.ds2 = nn.Conv3d(self.channels[2], n_classes, 1)
            self.ds1 = nn.Conv3d(self.channels[1], n_classes, 1)

    def forward(self, x):
        """
        x: [B, num_modalities, D, H, W] or [B, num_modalities*1, D, H, W]
        """
        # 1. 切分输入 (Split Input)
        # 将 [B, M, D, H, W] 切分为 M 个 [B, 1, D, H, W]
        x_list = torch.chunk(x, chunks=self.num_modalities, dim=1)
        
        # 2. 自适应 Stem
        stem_out_list = self.adaptive_stem(x_list)
        
        # 3. 并行编码
        # enc_feats_list[i] 包含第 i 个模态的 [L1, L2, L3, L4] 特征
        enc_feats_list = [] 
        for i in range(self.num_modalities):
            enc_feats_list.append(self.encoders[i](stem_out_list[i]))
            
        # 4. 逐层融合
        # 收集每一层的所有模态特征
        # layer_feats[0] = [Mod1_L1, Mod2_L1, ...]
        fused_skips = []
        
        # Layer 1
        l1_feats = [ef[0] for ef in enc_feats_list]
        f1, _ = self.fuse1(l1_feats); fused_skips.append(f1)
        
        # Layer 2
        l2_feats = [ef[1] for ef in enc_feats_list]
        f2, _ = self.fuse2(l2_feats); fused_skips.append(f2)
        
        # Layer 3
        l3_feats = [ef[2] for ef in enc_feats_list]
        f3, _ = self.fuse3(l3_feats); fused_skips.append(f3)
        
        # Layer 4 (Bottom)
        l4_feats = [ef[3] for ef in enc_feats_list]
        f4, _ = self.fuse4(l4_feats)

        # 5. 瓶颈
        x = self.bottleneck(f4)
        
        # 6. 解码
        s = fused_skips.pop(); x = self.decoders[0](x, s); d2 = self.ds2(x) if self.deep_sup else None
        s = fused_skips.pop(); x = self.decoders[1](x, s); d1 = self.ds1(x) if self.deep_sup else None
        s = fused_skips.pop(); x = self.decoders[2](x, s); final = self.outc(x)

        if self.training and self.deep_sup:
            return final, d1, d2
        return final

# ==============================================================================
# 自测代码
# ==============================================================================
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 场景 1: 3模态 (BraTS)
    print("--- Testing 3 Modalities ---")
    model_3 = Optimized_DynamicModal_Net(n_classes=4, base_c=16, num_modalities=3).to(device)
    x_3 = torch.randn(2, 3, 96, 96, 96).to(device) # Input shape [B, 3, D, H, W]
    with torch.no_grad():
        out = model_3(x_3)
    print("3-Modal Output:", out[0].shape if isinstance(out, tuple) else out.shape)
    
    # 场景 2: 1模态 (单模态测试)
    print("\n--- Testing 1 Modality ---")
    model_1 = Optimized_DynamicModal_Net(n_classes=4, base_c=16, num_modalities=1).to(device)
    x_1 = torch.randn(2, 1, 96, 96, 96).to(device)
    with torch.no_grad():
        out = model_1(x_1)
    print("1-Modal Output:", out[0].shape if isinstance(out, tuple) else out.shape)