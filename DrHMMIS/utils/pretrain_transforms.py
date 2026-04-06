import torch
import numpy as np
from monai.transforms import Transform, Compose, ToTensord
from scipy.ndimage import binary_opening, binary_closing, label
# 文件: utils/pretrain_transforms.py (新增内容)


from monai.transforms import MapTransform


class RandomChannelMaskingd(Transform):
    """
    对图像的每个通道，随机 mask 掉一定比例的 patch。

    输出:
    - input_image: 被 mask 的图像
    - target_image: 原始完整图像
    """
    def __init__(
        self,
        keys=("image",),
        masking_ratio=0.3,
        patch_size=(16, 16, 16),
        output_key="input_image",
        target_key="target_image"
    ):
        self.keys = keys
        self.masking_ratio = masking_ratio
        self.patch_size = patch_size
        self.output_key = output_key
        self.target_key = target_key

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            img = d[key]
            corrupted_img = img.clone()

            C, D, H, W = img.shape
            patch_D, patch_H, patch_W = self.patch_size

            num_patches = (D // patch_D) * (H // patch_H) * (W // patch_W)
            num_to_mask = int(num_patches * self.masking_ratio)

            for c in range(C):
                # 随机选择 patch
                d_indices = torch.randint(0, D // patch_D, (num_to_mask,))
                h_indices = torch.randint(0, H // patch_H, (num_to_mask,))
                w_indices = torch.randint(0, W // patch_W, (num_to_mask,))

                for di, hi, wi in zip(d_indices, h_indices, w_indices):
                    d_start, h_start, w_start = di * patch_D, hi * patch_H, wi * patch_W
                    d_end, h_end, w_end = d_start + patch_D, h_start + patch_H, w_start + patch_W

                    corrupted_img[c, d_start:d_end, h_start:h_end, w_start:w_end] = img.min()

            # 保存 input 和 target
            d[self.output_key] = corrupted_img
            d[self.target_key] = img

        return d

class DebugShapeTransformd(Transform):
    """
    一个用于调试的变换，它只打印出指定键的数据形状，然后原封不动地返回数据。
    """
    def __init__(self, keys: list, name: str = "Unnamed Step"):
        self.keys = keys
        self.name = name

    def __call__(self, data):
        print(f"\n--- [Debug Shape] at step: '{self.name}' ---")
        for key in self.keys:
            if key in data:
                print(f"    Key: '{key}', Shape: {data[key].shape}, Dtype: {data[key].dtype}")
        print("------------------------------------------")
        return data
# ==============================================================================
# 1. 核心自定义变换：伪标签生成器
# ==============================================================================

class PseudoLabelGeneratord(Transform):
    """
    一个在线生成伪标签的MONAI变换。
    它接收一个4通道的MRI图像，并根据我们设计的启发式规则，
    生成ET, ED, NCR三个区域的伪标签掩码。
    """
    def __init__(
        self,
        keys: str = "image",
        output_keys: tuple = ("et_mask", "ed_mask", "ncr_mask"),
        flair_z_score_threshold: float = 0.7,
        enhancement_threshold: float = 0.3,
        t1_low_signal_threshold: float = -0.2
    ):
        """
        初始化.
        Args:
            keys (str): 输入的4通道图像键。
            output_keys (tuple): 输出的三个掩码的键。
            ...thresholds: 用于生成掩码的可调参数。
        """
        super().__init__()
        self.keys = keys
        self.et_key, self.ed_key, self.ncr_key = output_keys
        self.flair_thresh = flair_z_score_threshold
        self.enhance_thresh = enhancement_threshold
        self.t1_low_thresh = t1_low_signal_threshold

    def __call__(self, data):
        d = dict(data)
        image = d[self.keys] # 形状: (4, H, W, D)

        # 假设通道顺序: 0:T1, 1:T1ce, 2:T2, 3:FLAIR
        t1_arr = image[0, ...]
        t1ce_arr = image[1, ...]
        t2_arr = image[2, ...]
        flair_arr = image[3, ...]

        # 步骤A: 创建脑部掩码 (BraTS数据>0即可)
        brain_mask = (t1_arr > t1_arr.min()) # 使用min()比0更稳健

        # 步骤B: Z-score标准化 (只在脑区内)
        modalities = [t1_arr, t1ce_arr, t2_arr, flair_arr]
        normalized_modalities = []
        for mod_arr in modalities:
            brain_pixels = mod_arr[brain_mask]
            mean = brain_pixels.mean()
            std = brain_pixels.std()
            normalized_arr = (mod_arr - mean) / (std + 1e-8)
            normalized_arr[~brain_mask] = 0
            normalized_modalities.append(normalized_arr)
        t1_norm, t1ce_norm, t2_norm, flair_norm = normalized_modalities
        
        # 步骤C: 生成总病灶伪掩码
        total_lesion_mask = (flair_norm > self.flair_thresh) & brain_mask
        # 清理: 去除小噪点，并只保留最大的连通区域
        total_lesion_mask = binary_opening(total_lesion_mask, structure=np.ones((3,3,3)))
        labels, num_features = label(total_lesion_mask)
        if num_features > 0:
            component_sizes = np.bincount(labels.ravel())[1:] # 忽略背景0
            largest_component_label = component_sizes.argmax() + 1
            total_lesion_mask = (labels == largest_component_label)
        
        # 步骤D: 生成伪增强区 (ET)
        enhancement_map = t1ce_norm - t1_norm
        pseudo_et_mask = (enhancement_map > self.enhance_thresh) & total_lesion_mask
        
        # 步骤E: 区分ED和NCR
        non_enhancing_lesion_mask = total_lesion_mask & (~pseudo_et_mask)
        pseudo_ncr_mask = non_enhancing_lesion_mask & (t1_norm < self.t1_low_thresh)
        pseudo_ed_mask = non_enhancing_lesion_mask & (~pseudo_ncr_mask)
        
        # 将生成的掩码添加回字典，并增加一个通道维度以符合MONAI习惯
        d[self.et_key] = torch.from_numpy(pseudo_et_mask).unsqueeze(0)
        d[self.ed_key] = torch.from_numpy(pseudo_ed_mask).unsqueeze(0)
        d[self.ncr_key] = torch.from_numpy(pseudo_ncr_mask).unsqueeze(0)

        return d

# ==============================================================================
# 2. 核心自定义变换：伪标签引导的掩码器
# ==============================================================================

class PseudoLabelGuidedMaskingd(Transform):
    """
    基于生成的伪标签，对多模态图像进行特异性、随机性的掩码。
    """
    def __init__(
        self,
        keys: str = "image",
        mask_keys: tuple = ("et_mask", "ed_mask", "ncr_mask"),
        masking_ratio: float = 0.7,
        output_key: str = "input_image",
        target_key: str = "target_image"
    ):
        super().__init__()
        self.keys = keys
        
        # 在这里定义了三个属性，名字都以 _mask_key 结尾
        self.et_mask_key, self.ed_mask_key, self.ncr_mask_key = mask_keys
        
        self.masking_ratio = masking_ratio
        self.output_key = output_key
        self.target_key = target_key

        # 定义每个区域要掩码的模态通道索引 (0:T1, 1:T1ce, 2:T2, 3:FLAIR)
        self.modality_map = {"ET": 1, "ED": 3, "NCR": 2}

    def __call__(self, data):
        d = dict(data)
        image = d[self.keys]
        
        corrupted_image = image.clone()
        
        # [*** 核心修正点 ***]
        # 此处使用了与 __init__ 中完全一致的、正确的属性名
        masks = {
            "ET": d.get(self.et_mask_key), 
            "ED": d.get(self.ed_mask_key), 
            "NCR": d.get(self.ncr_mask_key)
        }

        for region_name, mask in masks.items():
            if mask is None or mask.sum() == 0:
                continue
            
            if mask.ndim > image.ndim: mask = mask.squeeze(0)
            
            foreground_indices = torch.where(mask > 0)
            num_foreground = foreground_indices[0].shape[0]
            num_to_mask = int(num_foreground * self.masking_ratio)
            
            if num_to_mask == 0: continue

            perm = torch.randperm(num_foreground)
            indices_to_mask = perm[:num_to_mask]

            # MONAI 和 PyTorch 的维度顺序通常是 (C, D, H, W)
            # 所以 torch.where 的输出索引顺序是 (z, y, x)
            z, y, x = foreground_indices[0][indices_to_mask], foreground_indices[1][indices_to_mask], foreground_indices[2][indices_to_mask]
            
            channel_to_mask = self.modality_map[region_name]
            
            # 执行掩码 (将对应位置的值设为图像的最小值)
            corrupted_image[channel_to_mask, z, y, x] = image.min()
        
        d[self.output_key] = corrupted_image
        d[self.target_key] = d[self.keys]
        return d

# ==============================================================================
# 3. 主函数：构建并返回完整的变换流程
# ==============================================================================

def get_pretrain_transforms(patch_size: tuple = (96, 96, 96)):
    """
    [最终增强版] 包含几何增强的完整预训练变换流程。
    """
    from monai.transforms import (
        RandSpatialCropd,
        RandAffined,  # 使用功能更强大的 RandAffined
    )

    pretrain_transforms = Compose([
        # 假设数据已加载和合并
        # 输入: {"image": (4, H, W, D)}
        
        # 1. 随机裁剪出固定大小的训练区块
        RandSpatialCropd(keys=["image"], roi_size=patch_size, random_size=False),
        # 输出: {"image": (4, 96, 96, 96)}
        
        # 2. **[新增]** 应用随机的几何增强（旋转和翻转）
        RandAffined(
            keys=["image"],
            prob=0.8,  # 80%的概率应用此变换
            rotate_range=(np.pi/12, np.pi/12, np.pi/12), # 在各轴上随机旋转±15度
            scale_range=(0.9, 1.1), # 随机缩放90%到110%
            mode='bilinear',
            padding_mode='border',
        ),
        # 输出: {"image": (4, 96, 96, 96)} (图像内容已被旋转/缩放)
        
        # 3. 核心步骤1: 在线生成伪标签
        # 它会在已经被几何增强过的图像上生成伪标签
        PseudoLabelGeneratord(keys="image"),
        # 输出: {"image":..., "et_mask":..., "ed_mask":..., "ncr_mask":...}
        
        # 4. 核心步骤2: 应用伪标签引导的掩码
        # 它会基于增强后的图像，创建“损坏的输入”和“完好的目标”
        PseudoLabelGuidedMaskingd(
            keys="image", 
            masking_ratio=0.7,
        ),
        # 输出: {"input_image": (损坏的), "target_image": (完好的), ...}
        
        # 5. 将最终的输入和目标转换为PyTorch张量
        ToTensord(keys=["input_image", "target_image"])
    ])
    
    return pretrain_transforms


def get_pretrain_transforms_offline(patch_size: tuple = (96, 96, 96)):
    """
    [最终修正版] 分别处理image和mask的维度，彻底解决维度问题。
    """
    from monai.transforms import RandSpatialCropd, RandAffined, ToTensord, Compose, EnsureChannelFirstd
    import numpy as np # 确保导入了numpy

    image_keys = ["image"]
    mask_keys = ["et_mask", "ed_mask", "ncr_mask",'unsym_mask']
    all_keys = image_keys + mask_keys
    
    return Compose([
        # [*** 核心修正点 ***]
        # 我们不再对所有key使用同一个EnsureChannelFirstd，而是精确控制
        # 1. 只对3D的mask添加通道维度，image保持不变。
        EnsureChannelFirstd(keys=mask_keys, channel_dim="no_channel"),
        
        # 经过上一步，所有数据都已是标准的4D (C,D,H,W) 格式，可以安全地进行后续操作

        # 2. 同步地随机裁剪图像和所有伪标签
        #RandSpatialCropd(keys=all_keys, roi_size=patch_size, random_size=False),
        
        # 3. 对图像和所有伪标签应用相同的几何增强
        RandAffined(
            keys=all_keys,
            prob=0.8,
            rotate_range=(np.pi/12, np.pi/12, np.pi/12),
            scale_range=(0.9, 1.1),
            mode=('bilinear', 'nearest', 'nearest', 'nearest','nearest'),
            padding_mode='border',
        ),
        
        # 4. 核心步骤: 应用伪标签引导的掩码
        PseudoLabelGuidedMaskingd(
            keys="image", 
            mask_keys=("et_mask", "ed_mask", "ncr_mask"),
            masking_ratio=0.7,
        ),
        
        # 5. 转换为张量
        ToTensord(keys=["input_image", "target_image"])
    ])
def get_pretrain_transforms_offline_random(patch_size: tuple = (96, 96, 96)):
    from monai.transforms import RandSpatialCropd, RandAffined, ToTensord, Compose, EnsureChannelFirstd
    import numpy as np

    image_keys = ["image"]
    mask_keys = ["et_mask", "ed_mask", "ncr_mask", "unsym_mask"]
    all_keys = image_keys + mask_keys

    return Compose([
        # 只对 mask 添加通道维度，image 保持原样
        EnsureChannelFirstd(keys=mask_keys, channel_dim="no_channel"),

        # 裁剪
        RandSpatialCropd(keys=all_keys, roi_size=patch_size, random_size=False),

        # 几何增强
        RandAffined(
            keys=all_keys,
            prob=0.8,
            rotate_range=(np.pi/12, np.pi/12, np.pi/12),
            scale_range=(0.9, 1.1),
            mode=('bilinear', 'nearest', 'nearest', 'nearest', 'nearest'),
            padding_mode='border',
        ),

        # 只做 image 的通道级 patch 掩码
        RandomChannelMaskingd(
            keys=["image"],
            masking_ratio=0.3,        # 每通道随机mask 30%
            patch_size=(16, 16, 16)   # 掩码 patch 大小
        ),

        # 转成 tensor
        ToTensord(keys=["input_image", "target_image"])
    ])

# ==============================================================================
# 示例用法和测试
# ==============================================================================
if __name__ == '__main__':
    # 这个部分用于直接测试此文件的功能是否正常
    
    # 1. 创建一个假的4通道Numpy数组来模拟合并后的图像
    print("--- 创建一个假的4通道测试图像 ---")
    test_image_np = np.zeros((4, 128, 128, 128))
    # 模拟一个大脑
    test_image_np[:, 10:118, 10:118, 10:118] = 1.0 
    # 模拟一个高信号的总病灶区 (在FLAIR通道)
    test_image_np[3, 60:80, 60:80, 60:80] = 5.0 
    # 模拟一个增强区 (在T1ce通道)
    test_image_np[1, 65:75, 65:75, 65:75] = 6.0 
    # 模拟一个坏死区 (在T1通道信号低)
    test_image_np[0, 68:72, 68:72, 68:72] = 0.1

    # 2. 模拟一个MONAI数据字典
    data_dict = {"image": test_image_np}
    
    # 3. 构建并应用变换流程 (为了测试，我们只应用核心的两个变换)
    print("\n--- 构建并应用核心变换流程 ---")
    test_transforms = Compose([
        # 假设图像已加载和合并
        PseudoLabelGeneratord(keys="image"),
        PseudoLabelGuidedMaskingd(keys="image", masking_ratio=0.5)
    ])
    
    result_dict = test_transforms(data_dict)
    
    # 4. 检查输出结果
    print("\n--- 检查输出结果 ---")
    for key, value in result_dict.items():
        if isinstance(value, torch.Tensor):
            print(f"键: '{key}', 类型: {type(value)}, 形状: {value.shape}")
        else:
            print(f"键: '{key}', 类型: {type(value)}")

    # 验证输入和目标是否不同
    input_img = result_dict["input_image"]
    target_img = result_dict["target_image"]
    difference = torch.abs(input_img - target_img).sum()
    print(f"\n输入图像和目标图像之间的差异总和: {difference.item()}")
    if difference > 0:
        print("测试成功：掩码策略已成功应用，输入图像已被修改！")
    else:
        print("测试失败：输入图像未被修改。")