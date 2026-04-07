# DrHMMIS (Dual-resolution Hybrid Multi-Modal Image Segmentation)

DrHMMIS 是一个专注于 **3D 医学图像分割 (Medical Image Segmentation)** 的深度学习项目。本项目针对多模态 MRI 数据进行了深度优化，特别适用于 **WMH (White Matter Hyperintensities, 白质高信号)** 和 **BraTS (Brain Tumor Segmentation, 脑肿瘤分割)** 等复杂脑部病灶分割任务。

本项目核心特色在于采用了**多模态特征融合 (Multi-modal Feature Fusion)**、**自适应门控机制 (Adaptive Gating)** 以及**频域感知 (Frequency-Aware)** 技术，旨在提高复杂医学影像特征的提取能力与分割精度。

---

## 🚀 主要特性 (Features)

- **多编码器架构 (Multi-Encoder Structure)**：支持多种模态（如 T1, T2, FLAIR, T1ce）的独立特征提取与深度特征交互。
- **频域感知融合 (Frequency-Aware Fusion, FAF)**：将特征解耦为高低频分支，通过竞争性权重聚合，既保留了低频的解剖结构信息，又强化了高频的病灶边缘细节。
- **轻量级交叉注意力 (Light Cross-Attention)**：用于深层特征的跨模态交互，在提升性能的同时显著降低了三维注意力计算的开销。
- **轴向管状激发 (Axial Tubular Excitation)**：集成在 `HybridResBlock` 中，用于捕捉 3D 空间中的长程依赖关系。
- **解剖感知损失 (Anatomy-aware Loss)**：在 BraTS 任务中 (`train_brats_1.py`) 引入了解剖区域 Loss 管理器，对不同脑区（如背景、肿瘤核心、水肿区等）的损失进行动态加权。
- **滑动窗口推理 (Sliding Window Inference)**：原生支持针对大尺寸 3D 医疗影像的无缝滑动窗口推理。

---

## 📂 项目结构 (Repository Structure)

```text
DrHMMIS/
├── networks/               # 核心模型定义目录
│   ├── dual_encoder.py     # 双编码器基础网络
│   └── dual_enocder_unet.py# 包含各种机制的 UNet 变体模型
├── dataloaders/            # 数据加载、预处理与增强 (Dataset & Transforms)
├── utils/                  # 损失函数 (Loss)、评价指标 (Metrics) 与工具函数
├── data/                   # 数据集索引与存放路径
├── train_wmh.py            # WMH (白质高信号) 训练脚本
├── train_brats_1.py        # BraTS (脑肿瘤) 训练脚本
├── train_fully_supervised_3D.py # 3D 全监督通用训练脚本
├── val_3D.py               # 3D 验证脚本
├── test_wmh.py             # WMH 测试与推理脚本
└── requirements.txt        # 项目依赖环境配置
```

---

## 🛠️ 环境依赖 (Dependencies)

本项目基于 **PyTorch** 和 **MONAI** 框架构建。您可以直接通过前面生成的 `requirements.txt` 进行安装。

核心依赖如下：
- `torch` >= 1.10.0
- `torchvision`
- `monai` (用于 Loss、Metrics 及 Sliding Window Inference)
- `SimpleITK`, `nibabel` (医学图像读写)
- `h5py` (处理 H5 格式的切片/体素数据)
- `scipy`, `tqdm`, `tensorboardX`

**安装命令:**
```bash
pip install -r requirements.txt
```

---

## 🧠 支持的数据集 (Datasets)

1. **WMH (White Matter Hyperintensities)**
   - **输入模态**: FLAIR, T1 等。
   - **预处理**: 使用 `RobustIntensityNormalize` 进行基于百分位数的鲁棒归一化处理。
2. **BraTS 2021/2023 (Brain Tumor Segmentation)**
   - **输入模态**: FLAIR, T1w, T1ce, T2w。
   - **分割目标**: 增强肿瘤 (ET)、肿瘤核心 (TC)、全肿瘤 (WT)。

*注：项目目前使用 `h5py` 读取预处理后的 `.h5` 格式 3D patch 文件进行训练。*

---

## 💻 快速开始 (Getting Started)

### 1. 训练 (Training)

以训练 BraTS 数据集为例，运行 `train_brats_1.py`：

```bash
python train_brats_1.py \
    --root_path /path/to/your/brats_h5_data \
    --exp DrHMMIS_BraTS \
    --model dual_encoder_unet \
    --max_iterations 30000 \
    --batch_size 2 \
    --patch_size 96 96 96 \
    --base_lr 0.001 \
    --gpu 0
```

以训练 WMH 数据集为例：

```bash
python train_wmh.py \
    --root_path /path/to/your/wmh_h5_data \
    --batch_size 2 \
    --gpu 0
```

### 2. 测试与验证 (Testing & Validation)

使用推断脚本进行模型评估，支持滑动窗口策略：

```bash
python test_wmh.py \
    --root_path /path/to/test_data \
    --model dual_encoder_unet \
    --model_path /path/to/saved/best_model.pth \
    --patch_size 96 96 96 \
    --gpu 0
```

---

## 📈 补充信息

- **日志与可视化**: 训练过程中会使用 `TensorBoard` 自动记录 Loss、Dice 准确率指标。日志文件将保存在 `--exp` 参数指定的目录下（例如 `logs_experiment_.../`）。您可以运行 `tensorboard --logdir=./` 查看。
- **自定义模型拓展**: 如果需要在这个框架上引入新的模型组件，请在 `networks/` 中建立新的网络结构，并在 `train_*.py` 中实例化相应的网络类即可。