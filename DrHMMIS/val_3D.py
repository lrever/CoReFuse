import math
from glob import glob
import h5py
import nibabel as nib
import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F
from medpy import metric
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
def random_mask(patch, mask_size=(5, 5, 5), mask_num=3):
    """
    对给定的 3D patch 进行随机掩码，生成指定大小和数量的掩码块。
    :param patch: 输入 3D patch (numpy array)
    :param mask_size: 每个掩码块的大小 (W, H, D)
    :param mask_num: 掩码块的数量
    :return: 添加随机掩码后的 patch
    """
    masked_patch = patch.copy()
    w, h, d = patch.shape

    for _ in range(mask_num):
        # 随机生成掩码块的起始位置
        start_w = np.random.randint(0, max(1, w - mask_size[0]))
        start_h = np.random.randint(0, max(1, h - mask_size[1]))
        start_d = np.random.randint(0, max(1, d - mask_size[2]))

        # 应用掩码块
        end_w = start_w + mask_size[0]
        end_h = start_h + mask_size[1]
        end_d = start_d + mask_size[2]
        masked_patch[start_w:end_w, start_h:end_h, start_d:end_d] = 0  # 设置掩码块为 0

    return masked_patch
def test_single_case(net, image, image1, image2, image3,stride_xy, stride_z, patch_size, num_classes):
    w, h, d = image.shape

    # if the size of image is less than patch_size, then padding it
    add_pad = False
    if w < patch_size[0]:
        w_pad = patch_size[0] - w
        add_pad = True
    else:
        w_pad = 0
    if h < patch_size[1]:
        h_pad = patch_size[1] - h
        add_pad = True
    else:
        h_pad = 0
    if d < patch_size[2]:
        d_pad = patch_size[2] - d
        add_pad = True
    else:
        d_pad = 0
    wl_pad, wr_pad = w_pad // 2, w_pad - w_pad // 2
    hl_pad, hr_pad = h_pad // 2, h_pad - h_pad // 2
    dl_pad, dr_pad = d_pad // 2, d_pad - d_pad // 2
    if add_pad:
        image = np.pad(image, [(wl_pad, wr_pad), (hl_pad, hr_pad),
                               (dl_pad, dr_pad)], mode='constant', constant_values=0)
        image1 = np.pad(image1, [(wl_pad, wr_pad), (hl_pad, hr_pad),
                                 (dl_pad, dr_pad)], mode='constant', constant_values=0)
        image2 = np.pad(image2, [(wl_pad, wr_pad), (hl_pad, hr_pad),
                                 (dl_pad, dr_pad)], mode='constant', constant_values=0)
        image3 = np.pad(image3, [(wl_pad, wr_pad), (hl_pad, hr_pad),
                                 (dl_pad, dr_pad)], mode='constant', constant_values=0)
    ww, hh, dd = image.shape

    sx = math.ceil((ww - patch_size[0]) / stride_xy) + 1
    sy = math.ceil((hh - patch_size[1]) / stride_xy) + 1
    sz = math.ceil((dd - patch_size[2]) / stride_z) + 1
    # print("{}, {}, {}".format(sx, sy, sz))
    score_map = np.zeros((num_classes,) + image.shape).astype(np.float32)
    cnt = np.zeros(image.shape).astype(np.float32)

    for x in range(0, sx):
        xs = min(stride_xy * x, ww - patch_size[0])
        for y in range(0, sy):
            ys = min(stride_xy * y, hh - patch_size[1])
            for z in range(0, sz):
                zs = min(stride_z * z, dd - patch_size[2])
                test_patch = np.stack((
                    image[xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]],
                    image1[xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]],
                    image2[xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]],
                    image3[xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]]
                ), axis=0)
                test_patch = np.expand_dims(test_patch, axis=0).astype(np.float32)
                test_patch = torch.from_numpy(test_patch).cuda()
                # test_patch = image[xs:xs+patch_size[0],
                #                    ys:ys+patch_size[1], zs:zs+patch_size[2]]
                # test_patch = np.expand_dims(np.expand_dims(
                #     test_patch, axis=0), axis=0).astype(np.float32)
                # test_patch = torch.from_numpy(test_patch).cuda()

                # test_patch1 = image1[xs:xs+patch_size[0],
                #                    ys:ys+patch_size[1], zs:zs+patch_size[2]]
                # test_patch1 = np.expand_dims(np.expand_dims(
                #     test_patch1, axis=0), axis=0).astype(np.float32)
                # test_patch1 = torch.from_numpy(test_patch1).cuda()

                # test_patch2 = image2[xs:xs+patch_size[0],
                #                    ys:ys+patch_size[1], zs:zs+patch_size[2]]
                # test_patch2 = np.expand_dims(np.expand_dims(
                #     test_patch2, axis=0), axis=0).astype(np.float32)
                # test_patch2 = torch.from_numpy(test_patch2).cuda()
                # test_patch3 = image3[xs:xs+patch_size[0],
                #                    ys:ys+patch_size[1], zs:zs+patch_size[2]]
                # test_patch3 = np.expand_dims(np.expand_dims(
                #     test_patch3, axis=0), axis=0).astype(np.float32)
                # test_patch3 = torch.from_numpy(test_patch3).cuda()
                with torch.no_grad():
                    #y1=net(test_patch,test_patch1,test_patch2,test_patch3)
                    y1 = net(test_patch)['out'][-1]#, _
                    #y1=y1[0]
                    # ensure y1 is a tensor and apply softmax
                    y = torch.softmax(y1, dim=1)
                y = y.cpu().data.numpy()
                y = y[0, :, :, :, :]
                score_map[:, xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]] \
                    += y
                cnt[xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]] \
                    += 1
    score_map = score_map / np.expand_dims(cnt, axis=0)
    label_map = np.argmax(score_map, axis=0)

    if add_pad:
        label_map = label_map[wl_pad:wl_pad + w,
                              hl_pad:hl_pad + h, dl_pad:dl_pad + d]
        score_map = score_map[:, wl_pad:wl_pad +
                              w, hl_pad:hl_pad + h, dl_pad:dl_pad + d]
    return label_map,score_map

def test_single_case_epvs_1(net, image, image1, image2,mask,stride_xy, stride_z, patch_size, num_classes):
    w, h, d = image.shape

    # if the size of image is less than patch_size, then padding it
    add_pad = False
    if w < patch_size[0]:
        w_pad = patch_size[0] - w
        add_pad = True
    else:
        w_pad = 0
    if h < patch_size[1]:
        h_pad = patch_size[1] - h
        add_pad = True
    else:
        h_pad = 0
    if d < patch_size[2]:
        d_pad = patch_size[2] - d
        add_pad = True
    else:
        d_pad = 0
    wl_pad, wr_pad = w_pad // 2, w_pad - w_pad // 2
    hl_pad, hr_pad = h_pad // 2, h_pad - h_pad // 2
    dl_pad, dr_pad = d_pad // 2, d_pad - d_pad // 2
    if add_pad:
        image = np.pad(image, [(wl_pad, wr_pad), (hl_pad, hr_pad),
                               (dl_pad, dr_pad)], mode='constant', constant_values=0)
        image1 = np.pad(image1, [(wl_pad, wr_pad), (hl_pad, hr_pad),
                                 (dl_pad, dr_pad)], mode='constant', constant_values=0)
        image2 = np.pad(image2, [(wl_pad, wr_pad), (hl_pad, hr_pad),
                                 (dl_pad, dr_pad)], mode='constant', constant_values=0)
        mask = np.pad(mask, [(wl_pad, wr_pad), (hl_pad, hr_pad),
                                 (dl_pad, dr_pad)], mode='constant', constant_values=0)
    ww, hh, dd = image.shape

    sx = math.ceil((ww - patch_size[0]) / stride_xy) + 1
    sy = math.ceil((hh - patch_size[1]) / stride_xy) + 1
    sz = math.ceil((dd - patch_size[2]) / stride_z) + 1
    # print("{}, {}, {}".format(sx, sy, sz))
    score_map = np.zeros((num_classes,) + image.shape).astype(np.float32)
    cnt = np.zeros(image.shape).astype(np.float32)

    for x in range(0, sx):
        xs = min(stride_xy * x, ww - patch_size[0])
        for y in range(0, sy):
            ys = min(stride_xy * y, hh - patch_size[1])
            for z in range(0, sz):
                zs = min(stride_z * z, dd - patch_size[2])
                test_patch = np.stack((
                    image[xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]],
                    image1[xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]],
                    image2[xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]],
                ), axis=0)
                test_patch = torch.from_numpy(test_patch).cuda()
                test_patch = test_patch.unsqueeze(0)
                crop_mask = mask[xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]]
                
                crop_mask = torch.from_numpy(crop_mask)
                crop_mask= crop_mask.cuda()
                crop_mask = crop_mask.unsqueeze(0)
                crop_mask = crop_mask.long()
                atlas_one_hot = F.one_hot(crop_mask, num_classes=11)
                atlas_input = atlas_one_hot.permute(0, 4, 1, 2, 3).float()
                test_patch_t1 = image1[xs:xs+patch_size[0],
                                   ys:ys+patch_size[1], zs:zs+patch_size[2]]
                test_patch_t2 = image2[xs:xs+patch_size[0],
                                   ys:ys+patch_size[1], zs:zs+patch_size[2]]
                test_patch_flair = image[xs:xs+patch_size[0],
                                   ys:ys+patch_size[1], zs:zs+patch_size[2]]
                # # test_patch = np.expand_dims(test_patch, axis=0).astype(np.float32)
                test_patch_t1 = torch.from_numpy(test_patch_t1).cuda()
                test_patch_t2 = torch.from_numpy(test_patch_t2).cuda()
                test_patch_flair = torch.from_numpy(test_patch_flair).cuda()
                # test_patch = image[xs:xs+patch_size[0],
                #                    ys:ys+patch_size[1], zs:zs+patch_size[2]]
                # test_patch = np.expand_dims(np.expand_dims(
                #     test_patch, axis=0), axis=0).astype(np.float32)

                # test_patch1 = image1[xs:xs+patch_size[0],
                #                    ys:ys+patch_size[1], zs:zs+patch_size[2]]
                # test_patch1 = np.expand_dims(np.expand_dims(
                #     test_patch1, axis=0), axis=0).astype(np.float32)
                # test_patch1 = torch.from_numpy(test_patch1).cuda()

                # test_patch2 = image2[xs:xs+patch_size[0],
                #                    ys:ys+patch_size[1], zs:zs+patch_size[2]]
                # test_patch2 = np.expand_dims(np.expand_dims(
                #     test_patch2, axis=0), axis=0).astype(np.float32)
                # test_patch2 = torch.from_numpy(test_patch2).cuda()
                with torch.no_grad():
                    test_patch_t1=test_patch_t1.unsqueeze(0)
                    test_patch_t1=test_patch_t1.unsqueeze(0)
                    test_patch_t2=test_patch_t2.unsqueeze(0)
                    test_patch_t2=test_patch_t2.unsqueeze(0)
                    test_patch_flair=test_patch_flair.unsqueeze(0)
                    test_patch_flair=test_patch_flair.unsqueeze(0)
                    #print(test_patch.shape)
                    # batch_size, channels, depth, height, width = test_patch.size()
                    # test_patch = test_patch.permute(0, 2, 1, 3, 4).reshape(-1, channels, height, width)
                    y1=net(test_patch)
                    #y1,_=net(test_patch_t1,test_patch_t2,test_patch_flair)
                    #y1=net(test_patch_t1,test_patch_t2,test_patch_flair)
                    #y1 = y1.view(batch_size, num_classes,depth, height, width)
                    #y1 = net(test_patch)['out'][-1]#, _
                    #y1=y1[0]
                    # ensure y1 is a tensor and apply softmax
                    y = torch.softmax(y1, dim=1)

                y = y.cpu().data.numpy()
                y = y[0, :, :, :, :]
                score_map[:, xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]] \
                    += y
                cnt[xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]] \
                    += 1
    score_map = score_map / np.expand_dims(cnt, axis=0)
    label_map = np.argmax(score_map, axis=0)

    if add_pad:
        label_map = label_map[wl_pad:wl_pad + w,
                              hl_pad:hl_pad + h, dl_pad:dl_pad + d]
        score_map = score_map[:, wl_pad:wl_pad +
                              w, hl_pad:hl_pad + h, dl_pad:dl_pad + d]
    return label_map

def test_single_case_epvs(net, image, image1, image2,stride_xy, stride_z, patch_size, num_classes):
    w, h, d = image.shape

    # if the size of image is less than patch_size, then padding it
    add_pad = False
    if w < patch_size[0]:
        w_pad = patch_size[0] - w
        add_pad = True
    else:
        w_pad = 0
    if h < patch_size[1]:
        h_pad = patch_size[1] - h
        add_pad = True
    else:
        h_pad = 0
    if d < patch_size[2]:
        d_pad = patch_size[2] - d
        add_pad = True
    else:
        d_pad = 0
    wl_pad, wr_pad = w_pad // 2, w_pad - w_pad // 2
    hl_pad, hr_pad = h_pad // 2, h_pad - h_pad // 2
    dl_pad, dr_pad = d_pad // 2, d_pad - d_pad // 2
    if add_pad:
        image = np.pad(image, [(wl_pad, wr_pad), (hl_pad, hr_pad),
                               (dl_pad, dr_pad)], mode='constant', constant_values=0)
        image1 = np.pad(image1, [(wl_pad, wr_pad), (hl_pad, hr_pad),
                                 (dl_pad, dr_pad)], mode='constant', constant_values=0)
        image2 = np.pad(image2, [(wl_pad, wr_pad), (hl_pad, hr_pad),
                                 (dl_pad, dr_pad)], mode='constant', constant_values=0)
    ww, hh, dd = image.shape

    sx = math.ceil((ww - patch_size[0]) / stride_xy) + 1
    sy = math.ceil((hh - patch_size[1]) / stride_xy) + 1
    sz = math.ceil((dd - patch_size[2]) / stride_z) + 1
    # print("{}, {}, {}".format(sx, sy, sz))
    score_map = np.zeros((num_classes,) + image.shape).astype(np.float32)
    cnt = np.zeros(image.shape).astype(np.float32)

    for x in range(0, sx):
        xs = min(stride_xy * x, ww - patch_size[0])
        for y in range(0, sy):
            ys = min(stride_xy * y, hh - patch_size[1])
            for z in range(0, sz):
                zs = min(stride_z * z, dd - patch_size[2])
                input_batch=[]
                # flair=torch.from_numpy(image[xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]]).cuda()
                # t1=torch.from_numpy(image1[xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]]).cuda()
                # t2=torch.from_numpy(image2[xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]]).cuda()
                # flair=flair.unsqueeze(0)
                # t1=t1.unsqueeze(0)
                # t2=t2.unsqueeze(0)
                # flair=flair.unsqueeze(0)
                # t1=t1.unsqueeze(0)
                # t2=t2.unsqueeze(0)
                # input_batch.append(flair)
                # input_batch.append(t1)
                # input_batch.append(t2)



                test_patch = np.stack((
                    image[xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]],
                    image1[xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]],
                    image2[xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]]
                ), axis=0)
                
                # test_patch = image1[xs:xs+patch_size[0],
                #                    ys:ys+patch_size[1], zs:zs+patch_size[2]]
                # test_patch = np.expand_dims(test_patch, axis=0).astype(np.float32)
                test_patch = torch.from_numpy(test_patch).cuda()
                # test_patch = image[xs:xs+patch_size[0],
                #                    ys:ys+patch_size[1], zs:zs+patch_size[2]]
                # test_patch = np.expand_dims(np.expand_dims(
                #     test_patch, axis=0), axis=0).astype(np.float32)
                # test_patch = torch.from_numpy(test_patch).cuda()

                # test_patch1 = image1[xs:xs+patch_size[0],
                #                    ys:ys+patch_size[1], zs:zs+patch_size[2]]
                # test_patch1 = np.expand_dims(np.expand_dims(
                #     test_patch1, axis=0), axis=0).astype(np.float32)
                # test_patch1 = torch.from_numpy(test_patch1).cuda()

                # test_patch2 = image2[xs:xs+patch_size[0],
                #                    ys:ys+patch_size[1], zs:zs+patch_size[2]]
                # test_patch2 = np.expand_dims(np.expand_dims(
                #     test_patch2, axis=0), axis=0).astype(np.float32)
                # test_patch2 = torch.from_numpy(test_patch2).cuda()
                with torch.no_grad():
                    test_patch=test_patch.unsqueeze(0)
                    #print(test_patch.shape)
                    # batch_size, channels, depth, height, width = test_patch.size()
                    # test_patch = test_patch.permute(0, 2, 1, 3, 4).reshape(-1, channels, height, width)
                    y1=net(test_patch)
                    #y1 = y1.view(batch_size, num_classes,depth, height, width)
                    #y1 = net(test_patch)['out'][-1]#, _
                    #y1=y1[0]
                    # ensure y1 is a tensor and apply softmax
                    y = torch.softmax(y1, dim=1)
                y = y.cpu().data.numpy()
                y = y[0, :, :, :, :]
                score_map[:, xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]] \
                    += y
                cnt[xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]] \
                    += 1
    score_map = score_map / np.expand_dims(cnt, axis=0)
    label_map = np.argmax(score_map, axis=0)

    if add_pad:
        label_map = label_map[wl_pad:wl_pad + w,
                              hl_pad:hl_pad + h, dl_pad:dl_pad + d]
        score_map = score_map[:, wl_pad:wl_pad +
                              w, hl_pad:hl_pad + h, dl_pad:dl_pad + d]
    return label_map
def cal_metric(gt, pred):
    if pred.sum() > 0 and gt.sum() > 0:  
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return np.array([dice, hd95])
    else:
        print("111")
        return np.zeros(2)
def cal_metric_recall(gt, pred):
    if pred.sum() > 0 and gt.sum() > 0:
        # 1. 原有的 Dice 和 HD95
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        
        # 2. 新增 Recall 和 Precision 计算
        # 使用 numpy 计算 True Positive (TP), False Positive (FP), False Negative (FN)
        tp = ((pred == 1) & (gt == 1)).sum()
        fp = ((pred == 1) & (gt == 0)).sum()
        fn = ((pred == 0) & (gt == 1)).sum()
        
        # 加 1e-5 防止除以零
        recall = tp / (tp + fn + 1e-5)      # Recall = TP / GT_Total
        precision = tp / (tp + fp + 1e-5)   # Precision = TP / Pred_Total
        
        # 或者如果你想用 medpy 库 (如果有的话):
        # recall = metric.binary.sensitivity(pred, gt)
        # precision = metric.binary.precision(pred, gt)
        
        # 返回长度为 4 的数组
        return np.array([dice, hd95, recall, precision])
    else:
        print("Empty prediction or ground truth detected.")
        # 3. 注意：这里必须改为返回 4 个 0，以匹配上面的维度
        return np.zeros(4)
def save_nifti(np_data, ref_img, save_path, dtype=np.uint8):
    img = sitk.GetImageFromArray(np_data.astype(dtype))
    img.CopyInformation(ref_img)
    sitk.WriteImage(img, save_path)

def save_overlay_pngs(heatmap, background, save_dir, case_id, axis='x', colormap='jet', alpha=0.5):
    os.makedirs(save_dir, exist_ok=True)
    assert heatmap.shape == background.shape

    # 归一化热力图
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

    if axis == 'z':
        slices = range(heatmap.shape[0])
        get_slice = lambda i: (heatmap[i], background[i])
    elif axis == 'y':
        slices = range(heatmap.shape[1])
        get_slice = lambda i: (heatmap[:, i, :], background[:, i, :])
    elif axis == 'x':
        slices = range(heatmap.shape[2])
        get_slice = lambda i: (heatmap[:, :, i], background[:, :, i])
    else:
        raise ValueError("Invalid axis")

    for i in slices:
        heat, bg = get_slice(i)
        plt.figure(figsize=(6, 6))
        plt.imshow(bg, cmap='gray')
        plt.imshow(heat, cmap=colormap, alpha=alpha)
        plt.axis("off")
        plt.savefig(os.path.join(save_dir, f"case{case_id:03d}_{axis}{i:03d}.png"),
                    bbox_inches='tight', pad_inches=0)
        plt.close()

def test_all_case(net, base_dir, test_list="full_test.list", num_classes=2, patch_size=(48, 160, 160), stride_xy=32, stride_z=24,save_pred_dir=None,axis='x'):
    with open(base_dir + '/{}'.format(test_list), 'r') as f:
        image_list = f.readlines()
    image_list = [base_dir + "/{}.h5".format(
        item.replace('\n', '').split(",")[0]) for item in image_list]
    total_metric = np.zeros((num_classes - 1, 2))
    print("Validation begin")
    all_metrics = {'WT': [], 'TC': [], 'ET': []}
    for image_path in tqdm(image_list):
        h5f = h5py.File(image_path, 'r')
        image = h5f['flair'][:]
        image1 = h5f['t1'][:]
        image2 = h5f['t2'][:]
        image3 = h5f['t1ce'][:]
        label = h5f['seg'][:]

        prediction,score_map = test_single_case(
            net, image, image1, image2, image3,stride_xy, stride_z, patch_size, num_classes=num_classes)
        #-------------------------------------------------------------------------------------------------
        # for i in range(1, num_classes):
        #     total_metric[i - 1, :] += cal_metric(label == i, prediction == i)
        #--------------------------------------------------------------------------------------------------
        # 各类区域指标
        all_metrics['WT'].append(cal_metric(np.isin(label, [1, 2, 3]), np.isin(prediction, [1, 2, 3])))
        all_metrics['TC'].append(cal_metric(np.isin(label, [1, 3]), np.isin(prediction, [1, 3])))
        all_metrics['ET'].append(cal_metric(label == 3, prediction == 3))

    # 计算均值
    avg_results = {}
    for key in all_metrics:
        metrics = np.array(all_metrics[key])
        avg_results[key] = {
            'dice': metrics[:, 0].mean(),
            'hd95': metrics[:, 1].mean()
        }
    idx=100
    if save_pred_dir:
            case_dir = os.path.join(save_pred_dir, f"case_{idx:03d}")
            os.makedirs(case_dir, exist_ok=True)
            
            # 创建临时参考图像
            tmp_img = sitk.GetImageFromArray(image)  # 使用最后一个通道(通常是Flair)作为参考
            tmp_img.SetSpacing((1.0, 1.0, 1.0))
            tmp_img.SetOrigin((0.0, 0.0, 0.0))
            tmp_img.SetDirection((1.0, 0.0, 0.0,
                                 0.0, 1.0, 0.0,
                                 0.0, 0.0, 1.0))
            
            # 保存预测结果和热力图
            print(prediction.shape)
            save_nifti(prediction, tmp_img, os.path.join(case_dir, "pred_TC.nii.gz"))
            print(score_map.shape)
            # 保存WT热力图(概率图)
            wt_heatmap = score_map[1] + score_map[2] + score_map[3]  # 根据类别顺序调整
            save_nifti(wt_heatmap, tmp_img, os.path.join(case_dir, "WT_heatmap.nii.gz"), dtype=np.float32)
            
            # 保存最后一个通道图像(通常是Flair)
            save_nifti(image, tmp_img, os.path.join(case_dir, "image_last.nii.gz"), dtype=np.float32)
            
            # 保存叠加PNG图片
            png_dir = os.path.join(case_dir, "overlay_pngs")
            save_overlay_pngs(wt_heatmap,
                             image,
                             png_dir, idx, axis=axis)
    return avg_results
    # print("Validation end")
    # return total_metric / len(image_list)


# def test_all_case_epvs(net, base_dir, test_list="full_test.list", num_classes=2, patch_size=(48, 160, 160), stride_xy=32, stride_z=24):
#     print(test_list)
#     with open(base_dir + '/{}'.format(test_list), 'r') as f:
#         image_list = f.readlines()
#     image_list = [base_dir + "/{}".format(
#         item.replace('\n', '').split(",")[0]) for item in image_list]
#     total_metric = np.zeros((num_classes - 1, 2))
#     print("Validation begin")
#     for image_path in tqdm(image_list):
#         h5f = h5py.File(image_path, 'r')
#         image = h5f['flair'][:]
#         image1 = h5f['t1'][:]
#         image2 = h5f['t2'][:]
#         label = h5f['seg'][:]
#         mask = h5f['mask'][:]
#         prediction = test_single_case_epvs_1(
#             net, image, image1, image2,mask,stride_xy, stride_z, patch_size, num_classes=num_classes)
#         #-------------------------------------------------------------------------------------------------
#         for i in range(1, num_classes):
#             total_metric[i - 1, :] += cal_metric(label == i, prediction == i)
#             #print(total_metric)

#     return total_metric / len(image_list)
def test_all_case_epvs(net, base_dir, test_list="full_test.list", num_classes=2, patch_size=(48, 160, 160), stride_xy=32, stride_z=24):
    print(test_list)
    with open(base_dir + '/{}'.format(test_list), 'r') as f:
        image_list = f.readlines()
    image_list = [base_dir + "/{}".format(
        item.replace('\n', '').split(",")[0]) for item in image_list]
    
    # 假设 cal_metric 返回 [Dice, Precision]
    # total_metric 将存储 [总Dice, 总Precision]
    total_metric = np.zeros((num_classes - 1, 2)) 
    
    # --- 新增：故障检测计数器 ---
    failure_all_negative = 0  # 计数器: "预测全为阴"
    failure_high_fp = 0       # 计数器: "假阳多"
    total_cases_with_epvs = 0 # 分母：总共有多少个病例是真的有EPVS的
    
    # 你可以调整这个阈值
    LOW_PRECISION_THRESHOLD = 0.3 # 定义：低于10%的Precision被认为是“假阳多”
    # ---------------------------

    print("Validation begin")
    for image_path in tqdm(image_list):
        h5f = h5py.File(image_path, 'r')
        image = h5f['flair'][:]
        image1 = h5f['t1'][:]
        image2 = h5f['t2'][:]
        label = h5f['seg'][:]
        mask = h5f['mask'][:]
        prediction = test_single_case_epvs_1(
            net, image, image1, image2, mask, stride_xy, stride_z, patch_size, num_classes=num_classes)
        #-------------------------------------------------------------------------------------------------

        # --- 新增：故障检测逻辑 ---
        # 我们只关心EPVS (class 1)
        label_is_epvs = (label == 1)
        pred_is_epvs = (prediction == 1)

        total_label_epvs = np.sum(label_is_epvs)
        total_pred_epvs = np.sum(pred_is_epvs)

        if total_label_epvs > 0:
            # 这是一个有效的病例 (标签中确实有EPVS)
            total_cases_with_epvs += 1
            
            # 1. 检测: "预测全为阴" (Model Collapse)
            if total_pred_epvs == 0:
                print(f"\n[Failure: Model Collapse] {image_path.split('/')[-1]} - 标签有 {total_label_epvs} EPVS, 但模型预测为 0。")
                failure_all_negative += 1
            
            # 2. 检测: "假阳多" (High False Positives)
            else:
                tp = np.sum(label_is_epvs & pred_is_epvs)
                fp = np.sum(~label_is_epvs & pred_is_epvs)
                
                # Precision = TP / (TP + FP) = TP / total_pred_epvs
                precision = tp / total_pred_epvs
                
                if precision < LOW_PRECISION_THRESHOLD:
                    print(f"\n[Failure: High FP] {image_path.split('/')[-1]} - Precision极低: {precision:.4f} (TPs: {tp}, FPs: {fp})")
                    failure_high_fp += 1
                print(f"\n[success: ] {image_path.split('/')[-1]} - Precision: {precision:.4f} (TPs: {tp}, FPs: {fp})")
        # --- 故障检测结束 ---
        
        # 原始指标计算
        for i in range(1, num_classes):
            # 我们直接使用上面计算好的布尔掩码
            total_metric[i - 1, :] += cal_metric(label == i, prediction == i)
            
    # --- 新增：打印总结报告 ---
    num_images = len(image_list)
    print("\n" + "="*30)
    print("--- Validation Summary Report ---")
    print(f"Total cases tested: {num_images}")
    print(f"Total cases with EPVS in label: {total_cases_with_epvs}")

    if total_cases_with_epvs > 0:
        failure_rate_neg = (failure_all_negative / total_cases_with_epvs) * 100
        failure_rate_fp = (failure_high_fp / total_cases_with_epvs) * 100
    else:
        failure_rate_neg = 0.0
        failure_rate_fp = 0.0

    print("\n--- Failure Mode Analysis (在有EPVS的病例中) ---")
    print(f"1. '预测全为阴' (Model Collapse): {failure_all_negative} / {total_cases_with_epvs} ({failure_rate_neg:.2f}%)")
    print(f"2. '假阳多' (High FP, Prec < {LOW_PRECISION_THRESHOLD*100}%): {failure_high_fp} / {total_cases_with_epvs} ({failure_rate_fp:.2f}%)")
    
    avg_metric = total_metric / num_images
    print("\n--- Overall Metrics (Avg. per case) ---")
    # 假设 cal_metric 返回 [Dice, Precision]
    print(f"  Average Dice (Class 1): {avg_metric[0, 0]:.6f}")
    print(f"  Average Precision (Class 1): {avg_metric[0, 1]:.6f}")
    print("="*30 + "\n")
    
    # 返回原始指标
    return avg_metric




REGION_NAMES = {
    0: "Background",
    1: "Basal Ganglia",
    2: "Cerebral White Matter",
    3: "Cortex",
    4: "Thalamus",
    5: "Hippocampus",
    6: "Amygdala",
    7: "Cerebellum",
    8: "Brainstem",
    9: "Ventral Diencephalon",
    10: "Ventricular System & CSF"
}
NUM_REGIONS = 11 # 确保与 REGION_NAMES 匹配
# -----------------------------------

def test_all_case_epvs_check(net, base_dir, test_list="full_test.list", num_classes=2, patch_size=(48, 160, 160), stride_xy=32, stride_z=24):
    print(test_list)
    with open(base_dir + '/{}'.format(test_list), 'r') as f:
        image_list = f.readlines()
    image_list = [base_dir + "/{}".format(
        item.replace('\n', '').split(",")[0]) for item in image_list]
    
    # 假设 cal_metric 返回 [Dice, HD95] (根据你的日志)
    total_metric = np.zeros((num_classes - 1, 4)) 
    
    # --- 新增：FP 区域计数器 ---
    fp_per_region_counter = np.zeros(NUM_REGIONS, dtype=np.int64)
    total_all_fps_count = 0 # 总FP计数器
    
    # --- 故障检测计数器 ---
    failure_all_negative = 0
    failure_high_fp = 0
    total_cases_with_epvs = 0
    
    LOW_PRECISION_THRESHOLD = 0.3 # 阈值 (如你所设)

    print("Validation begin")
    for image_path in tqdm(image_list):
        h5f = h5py.File(image_path, 'r')
        image = h5f['flair'][:]
        image1 = h5f['t1'][:]
        image2 = h5f['t2'][:]
        label = h5f['seg'][:]
        mask = h5f['mask'][:]
        
        # 假设 test_single_case_epvs_1 是你用于滑窗预测的函数
        prediction = test_single_case_epvs_1(
            net, image, image1, image2, mask, stride_xy, stride_z, patch_size, num_classes=num_classes)
        #-------------------------------------------------------------------------------------------------

        # --- 故障检测逻辑 ---
        label_is_epvs = (label == 1)
        pred_is_epvs = (prediction == 1)

        total_label_epvs = np.sum(label_is_epvs)
        total_pred_epvs = np.sum(pred_is_epvs)
        
        # --- 新增：统计FP的区域分布 ---
        # 1. 找到所有假阳性 (FPs) 的位置: (预测为1) AND (标签不为1)
        #    (我们使用 label == 0 作为背景)
        fp_mask = (pred_is_epvs) & (label == 0) 
        current_case_fp_count = np.sum(fp_mask)
        total_all_fps_count += current_case_fp_count
        
        # 2. 遍历11个区域, 统计这些FPs分别落在哪里
        if current_case_fp_count > 0: # 仅当此病例有FPs时才统计
            for i in range(NUM_REGIONS):
                region_i_mask = (mask == i)
                # 计算交集 (即 在区域i内的FPs)
                fps_in_this_region = np.sum(fp_mask & region_i_mask)
                # 累加到总计数器
                fp_per_region_counter[i] += fps_in_this_region
        # --- 统计结束 ---

        if total_label_epvs > 0:
            total_cases_with_epvs += 1
            
            if total_pred_epvs == 0:
                print(f"\n[Failure: Model Collapse] {image_path.split('/')[-1]} - 标签有 {total_label_epvs} EPVS, 但模型预测为 0。")
                failure_all_negative += 1
            
            else:
                tp = np.sum(label_is_epvs & pred_is_epvs)
                # fp = current_case_fp_count (我们刚刚计算过)
                
                precision = tp / total_pred_epvs
                
                if precision < LOW_PRECISION_THRESHOLD:
                    print(f"\n[Failure: High FP] {image_path.split('/')[-1]} - Precision极低: {precision:.4f} (TPs: {tp}, FPs: {current_case_fp_count})")
                    failure_high_fp += 1
                else: # 只有 precision >= 0.3 才打印 success
                    print(f"\n[Success: ] {image_path.split('/')[-1]} - Precision: {precision:.4f} (TPs: {tp}, FPs: {current_case_fp_count})")
        
        # 原始指标计算
        for i in range(1, num_classes):
            total_metric[i - 1, :] += cal_metric_recall(label == i, prediction == i)
            
    # --- 打印总结报告 ---
    num_images = len(image_list)
    print("\n" + "="*30)
    print("--- Validation Summary Report ---")
    print(f"Total cases tested: {num_images}")
    print(f"Total cases with EPVS in label: {total_cases_with_epvs}")

    if total_cases_with_epvs > 0:
        failure_rate_neg = (failure_all_negative / total_cases_with_epvs) * 100
        failure_rate_fp = (failure_high_fp / total_cases_with_epvs) * 100
    else:
        failure_rate_neg = 0.0
        failure_rate_fp = 0.0

    print("\n--- Failure Mode Analysis (在有EPVS的病例中) ---")
    print(f"1. '预测全为阴' (Model Collapse): {failure_all_negative} / {total_cases_with_epvs} ({failure_rate_neg:.2f}%)")
    print(f"2. '假阳多' (High FP, Prec < {LOW_PRECISION_THRESHOLD*100}%): {failure_high_fp} / {total_cases_with_epvs} ({failure_rate_fp:.2f}%)")

    # --- 新增：FP 区域分布报告 ---
    print("\n--- False Positive (FP) Distribution Analysis ---")
    print(f"Total FPs across all subjects: {total_all_fps_count}")

    if total_all_fps_count > 0:
        print(" ID | 区域名称                 |  FP 计数   |  占总FP的比例 (%)")
        print("----+--------------------------+------------+------------------")
        
        # 排序 (从高到低)
        sorted_indices = np.argsort(fp_per_region_counter)[::-1]
        
        for i in sorted_indices:
            name = REGION_NAMES.get(i, f"未知区域 {i}")
            count = fp_per_region_counter[i]
            percent = (count / total_all_fps_count) * 100
            print(f" {i:^2} | {name:<24} | {count:^10} | {percent:^16.2f}%")
    else:
        print("No False Positives found in validation set.")
    # --- 报告结束 ---

    
    avg_metric = total_metric / num_images
    print("\n--- Overall Metrics (Avg. per case) ---")
    # 假设 cal_metric 返回 [Dice, HD95]
    print(f"  Average Dice (Class 1): {avg_metric[0, 0]:.6f}")
    # *** 修复了你的日志BUG: Precision -> HD95 ***
    print(f"  Average HD95 (Class 1): {avg_metric[0, 1]:.6f}")
    print("="*30 + "\n")
    
    return avg_metric