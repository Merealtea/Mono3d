import numpy as np

def make_color_wheel():
    """
    Generate color wheel according Middlebury color code
    :return: Color wheel
    """
    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR

    colorwheel = np.zeros([ncols, 3])

    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.transpose(np.floor(255 * np.arange(0, RY) / RY))
    col += RY

    # YG
    colorwheel[col:col + YG, 0] = 255 - np.transpose(np.floor(255 * np.arange(0, YG) / YG))
    colorwheel[col:col + YG, 1] = 255
    col += YG

    # GC
    colorwheel[col:col + GC, 1] = 255
    colorwheel[col:col + GC, 2] = np.transpose(np.floor(255 * np.arange(0, GC) / GC))
    col += GC

    # CB
    colorwheel[col:col + CB, 1] = 255 - np.transpose(np.floor(255 * np.arange(0, CB) / CB))
    colorwheel[col:col + CB, 2] = 255
    col += CB

    # BM
    colorwheel[col:col + BM, 2] = 255
    colorwheel[col:col + BM, 0] = np.transpose(np.floor(255 * np.arange(0, BM) / BM))
    col += + BM

    # MR
    colorwheel[col:col + MR, 2] = 255 - np.transpose(np.floor(255 * np.arange(0, MR) / MR))
    colorwheel[col:col + MR, 0] = 255

    return colorwheel


def compute_color(u, v):
    """
    compute optical flow color map
    :param u: optical flow horizontal map
    :param v: optical flow vertical map
    :return: optical flow in color code
    """
    [h, w] = u.shape
    img = np.zeros([h, w, 3])
    nanIdx = np.isnan(u) | np.isnan(v)
    u[nanIdx] = 0
    v[nanIdx] = 0

    colorwheel = make_color_wheel()
    ncols = np.size(colorwheel, 0)

    rad = np.sqrt(u ** 2 + v ** 2)

    a = np.arctan2(-v, -u) / np.pi

    fk = (a + 1) / 2 * (ncols - 1) + 1

    k0 = np.floor(fk).astype(int)

    k1 = k0 + 1
    k1[k1 == ncols + 1] = 1
    f = fk - k0

    for i in range(0, np.size(colorwheel, 1)):
        tmp = colorwheel[:, i]
        col0 = tmp[k0 - 1] / 255
        col1 = tmp[k1 - 1] / 255
        col = (1 - f) * col0 + f * col1

        idx = rad <= 1
        col[idx] = 1 - rad[idx] * (1 - col[idx])
        notidx = np.logical_not(idx)

        col[notidx] *= 0.75
        img[:, :, i] = np.uint8(np.floor(255 * col * (1 - nanIdx)))

    return img

UNKNOWN_FLOW_THRESH = 1e7
SMALLFLOW = 0.0
LARGEFLOW = 1e8

# from https://github.com/gengshan-y/VCN
def flow_to_image(flow):
    """
    Convert flow into middlebury color code image
    :param flow: optical flow map
    :return: optical flow image in middlebury color
    """
    u = flow[:, :, 0]
    v = flow[:, :, 1]

    maxu = -999.
    maxv = -999.
    minu = 999.
    minv = 999.

    idxUnknow = (abs(u) > UNKNOWN_FLOW_THRESH) | (abs(v) > UNKNOWN_FLOW_THRESH)
    u[idxUnknow] = 0
    v[idxUnknow] = 0

    maxu = max(maxu, np.max(u))
    minu = min(minu, np.min(u))

    maxv = max(maxv, np.max(v))
    minv = min(minv, np.min(v))

    rad = np.sqrt(u ** 2 + v ** 2)
    maxrad = max(-1, np.max(rad))

    u = u / (maxrad + np.finfo(float).eps)
    v = v / (maxrad + np.finfo(float).eps)

    img = compute_color(u, v)

    idx = np.repeat(idxUnknow[:, :, np.newaxis], 3, axis=2)
    img[idx] = 0

    return np.uint8(img)


# def disp2rgb(disp):
#     H = disp.shape[0]
#     W = disp.shape[1]
#
#     I = disp.flatten()
#
#     # map = np.array([[0, 0, 0, 114], # 黑色
#     #                 [0, 0, 1, 185], # 蓝色
#     #                 [1, 0, 0, 114], # 红色
#     #                 [1, 0, 1, 174], # 紫色
#     #                 [0, 1, 0, 114], # 绿色
#     #                 [0, 1, 1, 185], # 青色
#     #                 [1, 1, 0, 114], # 黄色
#     #                 [1, 1, 1, 0]]) # 白色
#
#     map = np.array([
#         [0, 0, 0, 142],  # 黑色
#         [1, 0, 0, 142],  # 红色
#         [1, 1, 0, 142],  # 黄色
#         [0, 1, 0, 142],  # 绿色
#         [0, 1, 1, 142],  # 青色
#         [0, 0, 1, 142],  # 蓝色
#         [1, 0, 1, 142],  # 品红
#         [1, 1, 1, 0]  # 白色
#     ])
#
#     bins = map[:-1,3]
#     cbins = np.cumsum(bins)
#     bins = bins/cbins[-1]
#     cbins = cbins[:-1]/cbins[-1]
#
#     ind = np.minimum(np.sum(np.repeat(I[None, :], 6, axis=0) > np.repeat(cbins[:, None],
#                                     I.shape[0], axis=1), axis=0), 6)
#     bins = np.reciprocal(bins)
#     cbins = np.append(np.array([[0]]), cbins[:, None])
#
#     I = np.multiply(I - cbins[ind], bins[ind])
#     I = np.minimum(np.maximum(np.multiply(map[ind,0:3], np.repeat(1-I[:,None], 3, axis=1)) \
#          + np.multiply(map[ind+1,0:3], np.repeat(I[:,None], 3, axis=1)),0),1)
#
#     I = np.reshape(I, [H, W, 3]).astype(np.float32)
#
#     return I

import matplotlib.pyplot as plt

def gt_scale_2_rgb(scale, colormap_name='plasma'):
    """
    将尺度因子图转换为 RGB 图像，使用最小-最大归一化和指定的颜色映射表。
    对于 scale == 0 的背景像素，直接填充为白色。
    """
    H, W = scale.shape
    scale_flat = scale.flatten()

    # ---- (1) 创建一个输出图像，初始化成全白 ----
    I_rgb_out = np.ones((H, W, 3), dtype=np.float32)  # 全白

    # ---- (2) 筛选出非背景像素，并做归一化 ----
    non_bg_mask = (scale_flat != 0)
    if np.any(non_bg_mask):
        non_bg_scale = scale_flat[non_bg_mask]

        scale_min = non_bg_scale.min()
        scale_max = non_bg_scale.max()

        if scale_max - scale_min == 0:
            # 所有非背景值相同
            I_normalized = np.zeros_like(non_bg_scale)
        else:
            I_normalized = (non_bg_scale - scale_min) / (scale_max - scale_min)

        # 获取颜色映射表
        cmap = plt.get_cmap(colormap_name)

        # 应用颜色映射表
        I_color_mapped = cmap(I_normalized)[:, :3]  # 忽略 Alpha 通道

        # ---- (3) 将颜色映射回去非背景像素位置 ----
        I_rgb_out_flat = I_rgb_out.reshape(-1, 3)
        I_rgb_out_flat[non_bg_mask] = I_color_mapped.astype(np.float32)
        I_rgb_out = I_rgb_out_flat.reshape(H, W, 3)

    return I_rgb_out


# def disp2rgb(disp, colormap_name='plasma'):
#     """
#     将视差图转换为 RGB 图像，使用最小-最大归一化和指定的颜色映射表。
#
#     参数：
#     - disp: 2D NumPy 数组，视差图。
#     - colormap_name: 字符串，Matplotlib 中的颜色映射表名称。
#
#     返回：
#     - RGB 图像，3D NumPy 数组，形状为 (H, W, 3)。
#     """
#     H, W = disp.shape
#     I = disp.flatten()
#
#     # 最小-最大归一化
#     disp_min = I.min()
#     disp_max = I.max()
#
#     if disp_max - disp_min == 0:
#         # 所有值相同，设为0
#         I_normalized = np.zeros_like(I)
#     else:
#         I_normalized = (I - disp_min) / (disp_max - disp_min)
#
#     # 获取颜色映射表
#     cmap = plt.get_cmap(colormap_name)
#
#     # 应用颜色映射表
#     I_rgb = cmap(I_normalized)[:, :3]  # 忽略 Alpha 通道
#
#     # 重塑为图像格式并转换数据类型
#     I_rgb = I_rgb.reshape(H, W, 3).astype(np.float32)
#
#     return I_rgb



def disp2rgb_normalized(disp_normalized, colormap_name='plasma'):
    """
    将已经归一化的视差图转换为 RGB 图像，使用指定的颜色映射表。

    参数：
    - disp_normalized: 2D NumPy 数组，视差图，已归一化到 [0, 1]。
    - colormap_name: 字符串，Matplotlib 中的颜色映射表名称。

    返回：
    - RGB 图像，3D NumPy 数组，形状为 (H, W, 3)，数据类型为 float32。
    """
    H, W = disp_normalized.shape

    # 获取颜色映射表
    cmap = plt.get_cmap(colormap_name)

    # 应用颜色映射表
    rgb_image = cmap(disp_normalized)[:, :, :3]  # 忽略 Alpha 通道

    # 转换为 float32 类型
    # rgb_image = rgb_image.reshape(H, W, 3).astype(np.float32)

    return rgb_image

def make_colorwheel():
    """
    Generates a color wheel for optical flow visualization as presented in:
        Baker et al. "A Database and Evaluation Methodology for Optical Flow" (ICCV, 2007)
        URL: http://vision.middlebury.edu/flow/flowEval-iccv07.pdf
    Code follows the original C++ source code of Daniel Scharstein.
    Code follows the the Matlab source code of Deqing Sun.
    Returns:
        np.ndarray: Color wheel
    """

    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros((ncols, 3))
    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.floor(255 * np.arange(0, RY) / RY)
    col = col + RY
    # YG
    colorwheel[col:col+YG, 0] = 255 - np.floor(255 * np.arange(0, YG) / YG)
    colorwheel[col:col+YG, 1] = 255
    col = col + YG
    # GC
    colorwheel[col:col+GC, 1] = 255
    colorwheel[col:col+GC, 2] = np.floor(255 * np.arange(0, GC) / GC)
    col = col + GC
    # CB
    colorwheel[col:col+CB, 1] = 255 - np.floor(255 * np.arange(CB) / CB)
    colorwheel[col:col+CB, 2] = 255
    col = col + CB
    # BM
    colorwheel[col:col+BM, 2] = 255
    colorwheel[col:col+BM, 0] = np.floor(255 * np.arange(0, BM) / BM)
    col = col + BM
    # MR
    colorwheel[col:col+MR, 2] = 255 - np.floor(255 * np.arange(MR) / MR)
    colorwheel[col:col+MR, 0] = 255
    return colorwheel


def flow_uv_to_colors(u, v, convert_to_bgr=False):
    """
    Applies the flow color wheel to (possibly clipped) flow components u and v.
    According to the C++ source code of Daniel Scharstein
    According to the Matlab source code of Deqing Sun
    Args:
        u (np.ndarray): Input horizontal flow of shape [H,W]
        v (np.ndarray): Input vertical flow of shape [H,W]
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.
    Returns:
        np.ndarray: Flow visualization image of shape [H,W,3]
    """
    flow_image = np.zeros((u.shape[0], u.shape[1], 3), np.uint8)
    colorwheel = make_colorwheel()  # shape [55x3]
    ncols = colorwheel.shape[0]
    rad = np.sqrt(np.square(u) + np.square(v))
    a = np.arctan2(-v, -u) / np.pi
    fk = (a + 1) / 2 * (ncols - 1)
    k0 = np.floor(fk).astype(np.int32)
    k1 = k0 + 1
    k1[k1 == ncols] = 0
    f = fk - k0
    for i in range(colorwheel.shape[1]):
        tmp = colorwheel[:, i]
        col0 = tmp[k0] / 255.0
        col1 = tmp[k1] / 255.0
        col = (1 - f) * col0 + f * col1
        idx = (rad <= 1)
        col[idx] = 1 - rad[idx] * (1 - col[idx])
        col[~idx] = col[~idx] * 0.75   # out of range
        # Note the 2-i => BGR instead of RGB
        ch_idx = 2 - i if convert_to_bgr else i
        flow_image[:, :, ch_idx] = np.floor(255 * col)
    return flow_image


def visual_scale_map_range_image(scale_map, valid_mask, colormap_name='seismic'):
    scale_display = np.copy(scale_map)

    # 将用于可视化的尺度值裁切到 [0.5, 1.5] 范围
    scale_display[valid_mask] = np.clip(scale_display[valid_mask], 0.85, 1.15)

    # 将尺度的分界线从 1 移至 0（scale - 1）
    deviations = np.zeros_like(scale_display, dtype=np.float32)
    deviations[valid_mask] = scale_display[valid_mask] - 1.0

    # 分别处理大于0和小于0的部分
    pos_mask = deviations > 0
    neg_mask = deviations < 0

    # 初始化归一化后的显示数组
    normalized_display = np.zeros_like(deviations, dtype=np.float32)

    # 处理大于1的尺度
    if np.any(pos_mask):
        pos_devs = deviations[pos_mask]
        pos_max = pos_devs.max()
        pos_min = pos_devs.min()
        if pos_max > pos_min >= 0:
            normalized_display[pos_mask] = (pos_devs - pos_min) / (pos_max - pos_min)  # 归一化到 [0,1]
        else:
            normalized_display[pos_mask] = 0.0  # 如果没有变化，设为0

    # 处理小于1的尺度
    if np.any(neg_mask):
        neg_devs = deviations[neg_mask]
        neg_max = neg_devs.max()
        neg_min = neg_devs.min()
        if neg_min < neg_max <= 0:
            normalized_display[neg_mask] = (neg_devs - neg_min) / (neg_max - neg_min) - 1.0  # 归一化到 [-1,0]
            # normalized_display[neg_mask] = neg_devs / abs(neg_min)  # 归一化到 [-1,0]
        else:
            normalized_display[neg_mask] = 0.0  # 如果没有变化，设为0

    # 设置等于1的尺度为0
    # 由于 deviations = scale -1，等于1的scale对应 deviations = 0，已经在 normalized_display 中为0

    # 对无效像素赋值为0
    normalized_display[~valid_mask] = 0.0

    return normalized_display

def visual_risk_score_map_range_image(risk_score_map):

    risk_score_display = risk_score_map

    pos_mask = risk_score_display > 0
    neg_mask = risk_score_display < 0

    normalized_risk_score = np.zeros_like(risk_score_display, dtype=np.float32)

    # 分段归一化，大于 0 表示朝向自车运动，小于 0 表示远离自车运动
    # 处理大于0的部分
    if np.any(pos_mask):
        pos_risk = risk_score_display[pos_mask]
        pos_max = pos_risk.max()
        pos_min = pos_risk.min()
        if pos_max > pos_min >= 0:
            normalized_risk_score[pos_mask] = (pos_risk - pos_min) / (pos_max - pos_min)
        else:
            normalized_risk_score[pos_mask] = 0.0
    # 处理小于0的部分
    if np.any(neg_mask):
        neg_risk = risk_score_display[neg_mask]
        neg_max = neg_risk.max()
        neg_min = neg_risk.min()
        if neg_min < neg_max <= 0:
            normalized_risk_score[neg_mask] = (neg_risk - neg_min) / (neg_max - neg_min) - 1.0
        else:
            normalized_risk_score[neg_mask] = 0.0
    
    return normalized_risk_score
    