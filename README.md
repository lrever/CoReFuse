# DrHMMIS (Dual-resolution Hybrid Multi-Modal Image Segmentation)

DrHMMIS is a deep learning project focusing on **3D Medical Image Segmentation**. This project is deeply optimized for multi-modal MRI data, making it particularly suitable for complex brain lesion segmentation tasks such as **WMH (White Matter Hyperintensities)** and **BraTS (Brain Tumor Segmentation)**.

The core features of this project include **Multi-modal Feature Fusion** and an **Adaptive Gating** mechanism. These innovations are designed to improve the feature extraction capability and overall segmentation accuracy for complex medical images.

---

## 🚀 Features

- **Multi-Encoder Structure**: Supports independent feature extraction and deep feature interaction across multiple modalities (e.g., T1, T2, FLAIR, T1ce).
- **Advanced Feature Fusion**: Aggregates multi-modal features via competitive weighting, preserving structural anatomical information while enhancing delicate lesion boundary details.
- **Light Cross-Attention**: Designed for cross-modal interaction of deep features, significantly reducing computational overhead for 3D attention while improving performance.
- **Axial Tubular Excitation**: Integrated into the `HybridResBlock` to capture long-range dependencies in 3D spatial dimensions.
- **Anatomy-aware Loss**: In the BraTS task (`train_brats_1.py`), an Anatomy Loss Manager is introduced to dynamically weight the loss across different brain regions (e.g., background, tumor core, edema).
- **Sliding Window Inference**: Natively supports seamless sliding window inference for large-sized 3D medical images.

---

## 📂 Repository Structure

```text
DrHMMIS/
├── networks/               # Core model definitions directory
│   ├── dual_encoder.py     # Base dual-encoder network
│   └── dual_enocder_unet.py# UNet variant models with various mechanisms
├── dataloaders/            # Data loading, preprocessing, and augmentation (Dataset & Transforms)
├── utils/                  # Loss functions, evaluation metrics, and utility functions
├── data/                   # Dataset indices and storage paths
├── train_wmh.py            # Training script for WMH
├── train_brats_1.py        # Training script for BraTS
├── train_fully_supervised_3D.py # General fully supervised 3D training script
├── val_3D.py               # 3D validation script
├── test_wmh.py             # Inference and testing script for WMH
└── requirements.txt        # Project dependencies
```

---

## 🛠️ Dependencies

This project is built on the **PyTorch** and **MONAI** frameworks. You can easily install the required environment via the provided `requirements.txt`.

Core dependencies include:
- `torch` >= 1.10.0
- `torchvision`
- `monai` (for Loss, Metrics, and Sliding Window Inference)
- `SimpleITK`, `nibabel` (for medical image I/O)
- `h5py` (for handling sliced/voxel data in H5 format)
- `scipy`, `tqdm`, `tensorboardX`

**Installation:**
```bash
pip install -r requirements.txt
```

---

## 🧠 Supported Datasets

1. **WMH (White Matter Hyperintensities)**
   - **Input Modalities**: FLAIR, T1, etc.
   - **Preprocessing**: Uses `RobustIntensityNormalize` for robust percentile-based normalization.
2. **BraTS 2021/2023 (Brain Tumor Segmentation)**
   - **Input Modalities**: FLAIR, T1w, T1ce, T2w.
   - **Target Classes**: Enhancing Tumor (ET), Tumor Core (TC), Whole Tumor (WT).

*Note: The project currently loads preprocessed 3D patch files in `.h5` format via `h5py` during training.*

---

## 💻 Getting Started

### 1. Training

Taking the BraTS dataset as an example, run `train_brats_1.py`:

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

Taking the WMH dataset as an example:

```bash
python train_wmh.py \
    --root_path /path/to/your/wmh_h5_data \
    --batch_size 2 \
    --gpu 0
```

### 2. Testing & Validation

Use the inference script to evaluate the model (supports sliding window strategy):

```bash
python test_wmh.py \
    --root_path /path/to/test_data \
    --model dual_encoder_unet \
    --model_path /path/to/saved/best_model.pth \
    --patch_size 96 96 96 \
    --gpu 0
```

---

## 📈 Additional Notes

- **Logging and Visualization**: `TensorBoard` is employed to automatically log training Loss and Dice accuracy metrics. The log files will be saved in the directory specified by the `--exp` argument (e.g., `logs_experiment_.../`). You can view it by running `tensorboard --logdir=./`.
- **Custom Model Extension**: If you need to introduce new model components within this framework, simply create the new network structure in `networks/` and instantiate the corresponding network class in `train_*.py`.
