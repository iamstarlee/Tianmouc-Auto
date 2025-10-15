import os
import re
import numpy as np
import torch
import torch.nn.functional as F
from blending import poisson_blending
import cv2

# ========== SD2XY ==========
def SD2XY(sd_raw: torch.Tensor) -> torch.Tensor:
    if len(sd_raw.shape) == 3:
        assert (sd_raw.shape[2] == 2 or sd_raw.shape[0] == 2)
        if sd_raw.shape[2] == 2:
            sd = sd_raw.permute(2, 0, 1).unsqueeze(0)
        else:
            sd = sd_raw.unsqueeze(0)
    else:
        assert (len(sd_raw.shape) == 4 and sd_raw.shape[1] == 2)
        sd = sd_raw

    b, c, h, w = sd.shape
    sdul = sd[:, 0:1, 0::2, ...]
    sdll = sd[:, 0:1, 1::2, ...]
    sdur = sd[:, 1:2, 0::2, ...]
    sdlr = sd[:, 1:2, 1::2, ...]

    target_size = (h, w * 2)
    sdul = F.interpolate(sdul, size=target_size, mode='bilinear', align_corners=False)
    sdll = F.interpolate(sdll, size=target_size, mode='bilinear', align_corners=False)
    sdur = F.interpolate(sdur, size=target_size, mode='bilinear', align_corners=False)
    sdlr = F.interpolate(sdlr, size=target_size, mode='bilinear', align_corners=False)

    sqrt2 = 1.41421356237
    sdx = ((sdul + sdll) / sqrt2 - (sdur + sdlr) / sqrt2) / 2
    sdy = ((sdur - sdlr) / sqrt2 + (sdul - sdll) / sqrt2) / 2

    if len(sd_raw.shape) == 3:
        return sdx.squeeze(0).squeeze(0), sdy.squeeze(0).squeeze(0)
    else:
        return sdx.squeeze(1), sdy.squeeze(1)

# ========== 文件排序 ==========
def sorted_npy_list(folder):
    files = [f for f in os.listdir(folder) if f.endswith(".npy")]
    files.sort(key=lambda x: tuple(int(num) for num in re.search(r"idx(\d+)_frame(\d+)", x).groups()))
    return [os.path.join(folder, f) for f in files]

# ========== 数据路径 ==========
data = 1200
left_sdl_dir = f"xinxinli/denoised_output_cam0/sdl"
left_sdr_dir = f"xinxinli/denoised_output_cam0/sdr"
# right_sdl_dir = "denoised_output_cam1/sdl"
# right_sdr_dir = "denoised_output_cam1/sdr"
save_left_dir = f"xinxinli/Poisson_denoised_out"
# save_right_dir = "Poisson_denoised_out/right"

left_sdl_files = sorted_npy_list(left_sdl_dir)
left_sdr_files = sorted_npy_list(left_sdr_dir)
# right_sdl_files = sorted_npy_list(right_sdl_dir)
# right_sdr_files = sorted_npy_list(right_sdr_dir)

# ========== 主循环 ==========
# 假设已经有 left_sdl_files, left_sdr_files, right_sdl_files, right_sdr_files
# 它们都是完整路径列表，并且按相同顺序配对

for l_sdl, l_sdr in zip(left_sdl_files, left_sdr_files):
    # 从路径提取文件名（去掉目录，只保留文件名）
    lf = os.path.basename(l_sdl)  # 左目输出文件名
    # rf = os.path.basename(r_sdl)  # 右目输出文件名

    # 读取
    left_sdl_data = torch.from_numpy(np.load(l_sdl)).float()
    left_sdr_data = torch.from_numpy(np.load(l_sdr)).float()
    # right_sdl_data = torch.from_numpy(np.load(r_sdl)).float()
    # right_sdr_data = torch.from_numpy(np.load(r_sdr)).float()

    # 拼成 (2, H, W) 送 SD2XY
    left_sd = torch.stack([left_sdl_data, left_sdr_data], dim=0)
    # right_sd = torch.stack([right_sdl_data, right_sdr_data], dim=0)

    sdlcx, sdlcy = SD2XY(left_sd)
    # sdrcx, sdrcy = SD2XY(right_sd)

    # Poisson blending
    sdlc_gpu = poisson_blending(sdlcx, sdlcy, iteration=15)
    # sdrc_gpu = poisson_blending(sdrcx, sdrcy, iteration=15)

    # 转 CPU numpy
    sdlc = sdlc_gpu.cpu().numpy()
    # sdrc = sdrc_gpu.cpu().numpy()

    # 保存
    np.save(os.path.join(save_left_dir, lf), sdlc)
    # np.save(os.path.join(save_right_dir, rf), sdrc)

    # 显示
    cv2.imshow("sdlc", ((sdlc + 50) * 2).astype(np.uint8))
    # cv2.imshow("sdrc", ((sdrc + 50) * 2).astype(np.uint8))
    cv2.waitKey(1)
