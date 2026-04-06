#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2019/12/14 下午4:41
# @Author  : chuyu zhang
# @File    : metrics.py
# @Software: PyCharm


import numpy as np
from medpy import metric
import torch
import torch.nn.functional as F
from .losses import create_window_3d # 从您已有的losses.py中导入辅助函数

def psnr_metric(img1, img2, data_range=255.0):
    """
    计算两张图像之间的峰值信噪比 (PSNR)。
    Args:
        img1, img2 (torch.Tensor): 两张需要比较的图像。
        data_range (float): 图像数据可能的动态范围 (例如，8位图像是255)。
                            如果您的数据已标准化到[-1, 1]，data_range应为2.0。
                            如果数据是[0, 1]，data_range应为1.0。
    """
    mse = F.mse_loss(img1, img2)
    psnr = 10 * torch.log10((data_range ** 2) / mse)
    return psnr

def ssim_metric(img1, img2, window_size=11, data_range=1.0, channel=4, size_average=True):
    """
    计算两张3D图像之间的结构相似性指数 (SSIM)。
    这是我们之前SSIM_Loss的“指标”版本，返回相似度而不是损失。
    """
    window = create_window_3d(window_size, channel)
    
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    mu1 = F.conv3d(img1, window, padding='same', groups=channel)
    mu2 = F.conv3d(img2, window, padding='same', groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv3d(img1 * img1, window, padding='same', groups=channel) - mu1_sq
    sigma2_sq = F.conv3d(img2 * img2, window, padding='same', groups=channel) - mu2_sq
    sigma12 = F.conv3d(img1 * img2, window, padding='same', groups=channel) - mu1_mu2
    
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean([1, 2, 3, 4])

def cal_dice(prediction, label, num=2):
    total_dice = np.zeros(num-1)
    for i in range(1, num):
        prediction_tmp = (prediction == i)
        label_tmp = (label == i)
        prediction_tmp = prediction_tmp.astype(np.float)
        label_tmp = label_tmp.astype(np.float)

        dice = 2 * np.sum(prediction_tmp * label_tmp) / (np.sum(prediction_tmp) + np.sum(label_tmp))
        total_dice[i - 1] += dice

    return total_dice


def calculate_metric_percase(pred, gt):
    dc = metric.binary.dc(pred, gt)
    jc = metric.binary.jc(pred, gt)
    hd = metric.binary.hd95(pred, gt)
    asd = metric.binary.asd(pred, gt)

    return dc, jc, hd, asd


def dice(input, target, ignore_index=None):
    smooth = 1.
    # using clone, so that it can do change to original target.
    iflat = input.clone().view(-1)
    tflat = target.clone().view(-1)
    if ignore_index is not None:
        mask = tflat == ignore_index
        tflat[mask] = 0
        iflat[mask] = 0
    intersection = (iflat * tflat).sum()

    return (2. * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth)


def calculate_precision_recall(pred, gt):
    # pred 和 gt 应该是二值化的 numpy 数组
    tp = np.sum((pred == 1) & (gt == 1))
    fp = np.sum((pred == 1) & (gt == 0))
    fn = np.sum((pred == 0) & (gt == 1))
    
    # 加 1e-5 防止除以零
    precision = tp / (tp + fp + 1e-5)
    recall = tp / (tp + fn + 1e-5)
    return precision, recall