import os
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap
import torch.nn as nn
import scipy.ndimage
import torch.nn.functional as F
from scipy.signal import wiener
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def conv_and_threshold(input_tensor, kernel_size, threshold):
    input_tensor_1 = input_tensor.unsqueeze(0).unsqueeze(0)
    input_tensor = torch.abs(input_tensor_1)
    conv_kernel = torch.ones((kernel_size, kernel_size), dtype=torch.float32).to(input_tensor.device)
    conv_layer = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(kernel_size, kernel_size), stride=1,
                           padding=kernel_size // 2, bias=False)
    with torch.no_grad():
        conv_layer.weight = nn.Parameter(conv_kernel.unsqueeze(0).unsqueeze(0))
    conv_output = conv_layer(input_tensor)
    mask = conv_output > threshold
    result_tensor = torch.where(mask, input_tensor_1, torch.tensor(0.0, device=input_tensor_1.device))

    result_tensor = result_tensor.squeeze(0).squeeze(0)

    return result_tensor

def conv_and_threshold_2(input_tensor, kernel_size, threshold):
    input_tensor_1 = input_tensor.unsqueeze(0).unsqueeze(0)
    input_tensor = torch.abs(input_tensor_1)
    conv_kernel = torch.ones((kernel_size, kernel_size), dtype=torch.float32).to(input_tensor.device)
    conv_kernel[kernel_size // 2, kernel_size // 2] = 0
    conv_layer = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(kernel_size, kernel_size), stride=1,
                           padding=kernel_size // 2, bias=False)
    with torch.no_grad():
        conv_layer.weight = nn.Parameter(conv_kernel.unsqueeze(0).unsqueeze(0))
    conv_output = conv_layer(input_tensor)
    mask = conv_output > threshold
    result_tensor = torch.where(mask, input_tensor_1, torch.tensor(0.0, device=input_tensor_1.device))
    result_tensor = result_tensor.squeeze(0).squeeze(0)
    return result_tensor

def td_denoise(A_tensor, td_param):
    A_1 = (torch.abs(A_tensor) >= 1).float() * A_tensor
    sub_tensor = A_1.unsqueeze(0).unsqueeze(0)
    kernel = torch.ones((1, 1, 3, 3), dtype=torch.float32).to(A_tensor.device) / 9.0
    smoothed_matrix = F.conv2d(sub_tensor, kernel, padding=1)
    smoothed_matrix = smoothed_matrix.squeeze(0).squeeze(0)
    sub_tensor = sub_tensor.squeeze(0).squeeze(0)
    mask_1 = torch.abs(smoothed_matrix) >= 0.5
    TD_10 = sub_tensor * mask_1
    TD_3 = conv_and_threshold(TD_10, 3, 5)
    TD_4 = conv_and_threshold_2(TD_3, 3, 5)
    label_array = torch.full(A_tensor.shape, -1).to(A_tensor.device)
    label_array[TD_4 != 0] = 1
    label_array[A_tensor == 0] = 0
    return TD_4, label_array

def local_var_test(input_tensor, kernel_size):
    if input_tensor.dim() == 2:
        tensor = input_tensor.unsqueeze(0).unsqueeze(0)
    elif input_tensor.dim() == 3:
        tensor = input_tensor.unsqueeze(0)
    kernel = torch.ones(1, 1, kernel_size, kernel_size, device=tensor.device) / 9
    local_mean = F.conv2d(tensor, kernel, padding=1)
    local_mean_square = F.conv2d(tensor ** 2, kernel, padding=1)
    local_variance = local_mean_square - local_mean ** 2
    result_tensor = local_variance
    result_tensor = result_tensor.squeeze(0).squeeze(0)
    return result_tensor

def variance_to_threshold(variance, min_thr=3, max_thr=8):
    norm_variance = torch.clamp(variance, min=0, max=1)
    threshold_range = max_thr - min_thr
    dynamic_threshold = min_thr + (norm_variance) * threshold_range
    return dynamic_threshold

def calculate_metrics(mode_1, mode_2):
    TP = ((mode_1 == 1) & (mode_2 == 1)).sum().item()
    TN = ((mode_1 == -1) & (mode_2 == -1)).sum().item()
    FP = ((mode_1 == -1) & (mode_2 != -1)).sum().item()
    FN = ((mode_1 == 1) & (mode_2 != 1)).sum().item()
    return TP, TN, FP, FN

def local_var(input_tensor, kernel_size, th_mean, th_var):
    if input_tensor.dim() == 2:
        tensor = input_tensor.unsqueeze(0).unsqueeze(0)
    elif input_tensor.dim() == 3:
        tensor = input_tensor.unsqueeze(0)
    kernel = torch.ones(1, 1, kernel_size, kernel_size) / 9
    kernel = kernel.to(tensor.device)
    local_mean = F.conv2d(tensor, kernel, padding=1)
    local_mean_square = F.conv2d(tensor ** 2, kernel, padding=1)
    local_variance = local_mean_square - local_mean ** 2
    mask = (local_variance > th_var)
    result_tensor = torch.where(mask, tensor, torch.tensor(0.0, device=tensor.device))
    result_tensor = result_tensor.squeeze(0).squeeze(0)
    return result_tensor

def variance_to_threshold_td(variance, min_thr=3, max_thr=8):
    norm_variance = torch.clamp(variance, min=0, max=2)
    threshold_range = max_thr - min_thr
    dynamic_threshold = min_thr + (norm_variance) * threshold_range / 2
    return dynamic_threshold

def td_adaptive_filter(A, min_thr=3, max_thr=8, kernel_size=3):
    local = local_var_test(A, kernel_size)
    dynamic_threshold = variance_to_threshold_td(local, min_thr, max_thr)
    TD_10 = (torch.abs(A) >= 1).float() * A
    TD_3 = conv_and_threshold(TD_10, kernel_size, dynamic_threshold)
    TD_4 = conv_and_threshold_2(TD_3, kernel_size, 5)
    A_tensor = A
    label_array = torch.full(A_tensor.shape, -1, device=A_tensor.device)
    label_array[TD_4 != 0] = 1
    label_array[A_tensor == 0] = 0
    return TD_4

def batch_local_var(input_batch, kernel_size):
    if input_batch.dim() == 3:
        tensor = input_batch.unsqueeze(1)  # (B, 1, H, W)
    else:
        tensor = input_batch
    kernel = torch.ones(1, 1, kernel_size, kernel_size,
                       device=tensor.device, dtype=tensor.dtype) / (kernel_size**2)
    local_mean = F.conv2d(tensor, kernel, padding=kernel_size//2, groups=tensor.size(1))
    local_mean_square = F.conv2d(tensor**2, kernel, padding=kernel_size//2, groups=tensor.size(1))
    return local_mean_square - local_mean**2

def batch_variance_to_threshold(variance_batch, min_thr, max_thr):
    norm_variance = torch.clamp(variance_batch, min=0, max=1)
    return min_thr + norm_variance * (max_thr - min_thr)

def batch_conv_threshold(input_batch, kernel_size, threshold_batch):
    if input_batch.dim() == 3:
        inputtensor = input_batch.unsqueeze(1)  # (B, 1, H, W)
    else:
        inputtensor = input_batch

    B, C, H, W = inputtensor.shape
    abs_input = torch.abs(inputtensor)
    kernel = torch.ones(1, 1, kernel_size, kernel_size,
                       device=inputtensor.device, dtype=inputtensor.dtype)
    conv_output = F.conv2d(abs_input, kernel, padding=kernel_size//2, groups=C)
    mask = conv_output > threshold_batch
    return torch.where(mask, inputtensor, 0)

def batch_conv_threshold_2(input_batch, kernel_size, threshold):
    if input_batch.dim() == 3:
        inputtensor = input_batch.unsqueeze(1)  # (B, 1, H, W)
    else:
        inputtensor = input_batch

    B, C, H, W = inputtensor.shape
    abs_input = torch.abs(inputtensor)
    kernel = torch.ones(kernel_size, kernel_size,
                       device=inputtensor.device, dtype=inputtensor.dtype)
    kernel[kernel_size//2, kernel_size//2] = 0
    kernel = kernel.view(1, 1, kernel_size, kernel_size)
    conv_output = F.conv2d(abs_input, kernel, padding=kernel_size//2, groups=C)
    if isinstance(threshold, (int, float)):
        threshold_tensor = torch.tensor(threshold, device=inputtensor.device)
    else:
        threshold_tensor = threshold.view(B, C, 1, 1)

    mask = conv_output > threshold_tensor
    return torch.where(mask, inputtensor, 0)


def batch_sd_adaptive_filter(A_batch, min_thr=3, max_thr=8, kernel_size=3):
    local = batch_local_var(A_batch, kernel_size)
    dynamic_threshold = batch_variance_to_threshold(local, min_thr, max_thr)
    TD_10 = (torch.abs(A_batch) >= 1).float() * A_batch
    TD_3 = batch_conv_threshold(TD_10, kernel_size, dynamic_threshold)
    TD_4 = batch_conv_threshold_2(TD_3, kernel_size, 5)
    TD_4 = TD_4.squeeze(1)
    label_array = torch.full_like(A_batch, -1, device=A_batch.device)
    label_array[TD_4 != 0] = 1
    label_array[A_batch == 0] = 0
    return TD_4

def sd_denoise_4dir_adp_test(A_tensor, B_tensor, sd_params):
    result = torch.zeros(160, 320, device=A_tensor.device)
    var_fil_ksize = sd_params["var_fil_ksize"]
    var_th = sd_params["var_th"]
    adapt_th_min = sd_params["adapt_th_min"]
    adapt_th_max = sd_params["adapt_th_max"]
    result[:, ::2] = A_tensor.clone()
    result[:, 1::2] = B_tensor.clone()
    result_copy = local_var(result, var_fil_ksize, 0.5, var_th)
    result_1 = result_copy.clone()
    result_1[0::2, 0::2] = -result_1[0::2, 0::2]
    result_2 = result_copy.clone()
    result_2[0::2, 1::2] = -result_2[0::2, 1::2]
    result_3 = result_copy.clone()
    result_3[0::2, ...] = -result_3[0::2, ...]
    result_4 = result_copy.clone()
    result_4[..., 0::2] = -result_4[..., 0::2]
    result_1_pos = result_1 * (result_1 >= 1).float()
    result_1_neg = result_1 * (result_1 <= -1).float()
    result_2_pos = result_2 * (result_2 >= 1).float()
    result_2_neg = result_2 * (result_2 <= -1).float()
    result_3_pos = result_3 * (result_3 >= 1).float()
    result_3_neg = result_3 * (result_3 <= -1).float()
    result_4_pos = result_4 * (result_4 >= 1).float()
    result_4_neg = result_4 * (result_4 <= -1).float()
    t2 = time.time()
    inputs = torch.stack([
        result_1_pos, result_2_pos, result_3_pos, result_4_pos,
        result_1_neg, result_2_neg, result_3_neg, result_4_neg
    ], dim=0)

    batch_output = batch_sd_adaptive_filter(inputs, adapt_th_min, adapt_th_max)
    (ad_1_pos, ad_2_pos, ad_3_pos, ad_4_pos,
     ad_1_neg, ad_2_neg, ad_3_neg, ad_4_neg) = batch_output.unbind(0)
    ad_sum = ad_1_pos + ad_2_pos + ad_3_pos + ad_4_pos - ad_1_neg - ad_2_neg - ad_3_neg - ad_4_neg
    final_result = (ad_sum > 0).float() * result
    final_l = final_result[:, ::2]
    final_r = final_result[:, 1::2]
    label_array_l = torch.full(final_l.shape, -1, device=final_l.device)
    label_array_l[final_l != 0] = 1
    label_array_l[A_tensor == 0] = 0
    label_array_r = torch.full(final_r.shape, -1, device=final_l.device)
    label_array_r[final_r != 0] = 1
    label_array_r[B_tensor == 0] = 0

    return final_l, label_array_l, final_r, label_array_r


def td_new(sdl_1_denoised, sdr_1_denoised, sdl_2_denoised, sdr_2_denoised, td_tensor, td_params):
    td_raw = td_tensor.float()
    td_orin = td_raw.clone()

    var_fil_ksize = td_params["var_fil_ksize"]
    var_th = td_params["var_th"]

    adapt_th_min = td_params["adapt_th_min"]
    adapt_th_max = td_params["adapt_th_max"]

    t0 = time.time()
    td_orin = local_var(td_orin, var_fil_ksize, 0.5, var_th)
    t1 = time.time()
    tdl_orin = torch.empty_like(td_orin, device=td_orin.device)

    zero_tensor_1 = torch.zeros(80, 160).to(td_orin.device)
    zero_tensor_1[0:79, ...] = td_orin[2::2, :].clone()

    tdl_orin[::2, :] = td_orin[1::2, :] - td_orin[::2, :]
    tdl_orin[1::2, :] = -(zero_tensor_1 - td_orin[1::2, :])
    tdr_orin = torch.empty_like(td_orin, device=td_orin.device)

    orin = torch.empty_like(td_orin, device=td_orin.device)
    orin[::2, :-1] = td_orin[::2, 1:]
    orin[1::2, ] = td_orin[1::2, ]
    orin[::2, -1] = 0

    zero_tensor_2 = torch.zeros(80, 160).to(td_orin.device)
    zero_tensor_2[0:79, ...] = orin[2::2, :].clone()

    tdr_orin[::2, :] = orin[1::2, :] - orin[::2, :]
    tdr_orin[1::2, :] = zero_tensor_2 - orin[1::2, :]

    t2 = time.time()

    sdl_1_cali = sdl_1_denoised
    sdr_1_cali = sdr_1_denoised
    sdl_2_cali = sdl_2_denoised
    sdr_2_cali = sdr_2_denoised

    tensor1 = sdl_1_cali
    tensor2 = sdl_2_cali

    polarities_diff = torch.sign(tensor1) != torch.sign(tensor2)
    polarities_same_and_zero = (torch.sign(tensor1) == torch.sign(tensor2)) & (tensor1 == 0) & (tensor2 == 0)
    polarities_same_and_nonzero = (torch.sign(tensor1) == torch.sign(tensor2)) & (tensor1 != 0) & (tensor2 != 0)

    ref_tensor = (tensor2 - tensor1) * polarities_diff.float()
    cmp_tensor = tdl_orin * polarities_diff.float()
    ref_sign = torch.sign(ref_tensor)
    cmp_sign = torch.sign(cmp_tensor)
    mask = (ref_sign == cmp_sign)
    result_tensor = torch.where(mask, cmp_tensor, torch.zeros_like(cmp_tensor, device=td_orin.device))
    polarity_flag = torch.zeros_like(td_orin, dtype=torch.bool , device=td_orin.device)
    polarity_flag[::2, :] = ((td_orin[1::2, :]) == torch.sign(td_orin[::2, :]))
    polarity_flag[1::2, :] = torch.sign(zero_tensor_1) == torch.sign(td_orin[1::2, :])
    final_l = polarity_flag.float() * tdl_orin * polarities_same_and_zero.float() + result_tensor + tdl_orin * polarities_same_and_nonzero.float()

    t3 = time.time()

    tensor1 = sdr_1_cali
    tensor2 = sdr_2_cali

    polarities_diff = torch.sign(tensor1) != torch.sign(tensor2)
    polarities_same_and_zero = (torch.sign(tensor1) == torch.sign(tensor2)) & (tensor1 == 0) & (tensor2 == 0)
    polarities_same_and_nonzero = (torch.sign(tensor1) == torch.sign(tensor2)) & (tensor1 != 0) & (tensor2 != 0)
    ref_tensor = (tensor2 - tensor1) * polarities_diff.float()
    cmp_tensor = tdr_orin * polarities_diff.float()
    ref_sign = torch.sign(ref_tensor)
    cmp_sign = torch.sign(cmp_tensor)
    mask = (ref_sign == cmp_sign)
    result_tensor = torch.where(mask, cmp_tensor, torch.zeros_like(cmp_tensor))
    polarity_flag = torch.zeros_like(orin, dtype=torch.bool, device=td_orin.device)
    polarity_flag[::2, :] = ((orin[1::2, :]) == torch.sign(orin[::2, :]))
    polarity_flag[1::2, :] = torch.sign(zero_tensor_2) == torch.sign(orin[1::2, :])
    final_r = polarity_flag.float() * tdr_orin * polarities_same_and_zero.float() + result_tensor + tdr_orin * polarities_same_and_nonzero.float()

    t4 = time.time()

    tsd_lu_orin = final_l[::2, :]
    tsd_ll_orin = final_l[1::2, :]
    tsd_ru_orin = final_r[::2, :]
    tsd_rl_orin = final_r[1::2, :]

    tsd_lu_to_td = torch.zeros(160, 160).to(td_tensor.device)
    tsd_ll_to_td = torch.zeros(160, 160).to(td_tensor.device)
    tsd_ru_to_td = torch.zeros(160, 160).to(td_tensor.device)
    tsd_rl_to_td = torch.zeros(160, 160).to(td_tensor.device)

    tsd_lu_to_td[::2, :] = tsd_lu_orin
    tsd_lu_to_td[1::2, :] = tsd_lu_orin
    tsd_ll_to_td[1::2, :] = tsd_ll_orin
    tsd_ll_to_td[2::2, :] = tsd_ll_orin[0:79, ...]
    tsd_ru_to_td[::2, 1:] = tsd_ru_orin[..., 0:159]
    tsd_ru_to_td[1::2, :] = tsd_ru_orin
    tsd_rl_to_td[2::2, 1:] = tsd_rl_orin[0:79, 0:159]
    tsd_rl_to_td[1::2, :] = tsd_rl_orin
    t5 = time.time()

    rusult = ((torch.abs(tsd_lu_to_td) > 0).float() + (torch.abs(tsd_ll_to_td) > 0).float() + (
                torch.abs(tsd_ru_to_td) > 0).float() + (torch.abs(tsd_rl_to_td) > 0).float()) * td_orin

    rusult = td_adaptive_filter(rusult, min_thr=adapt_th_min, max_thr=adapt_th_max, kernel_size=3)
    t6 = time.time()

    A_tensor = td_raw
    label_array = torch.full(A_tensor.shape, -1, device=A_tensor.device)
    label_array[rusult != 0] = 1  # LSD_reconstructed
    label_array[A_tensor == 0] = 0

    return rusult, label_array

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def proces_scene(root_path, td_param, sd_param, save_path):
    sdl_dir = os.path.join(save_path, 'sdl')
    sdr_dir = os.path.join(save_path, 'sdr')
    td_dir = os.path.join(save_path, 'td')
    os.makedirs(sdl_dir, exist_ok=True)
    os.makedirs(sdr_dir, exist_ok=True)
    os.makedirs(td_dir, exist_ok=True)

    for subdir, _, files in os.walk(root_path):
        de_files = [file for file in files if file.endswith('.npy')]
        if not de_files:
            continue
        print(f"Processing {subdir}")

        de_files.sort()

        last_lsd = None
        last_rsd = None

        for de_file in de_files:
            de_file_path = os.path.join(subdir, de_file)
            de_array = np.load(de_file_path)
            de_array_tensor = torch.from_numpy(de_array).to(device)
            de_array_tensor = de_array_tensor.permute(1, 2, 0)  # [H, W, 3]

            # 分通道调用降噪函数
            LSD_reconstructed, label_l, RSD_reconstructed, label_r = sd_denoise_4dir_adp_test(
                de_array_tensor[..., 1], de_array_tensor[..., 2], sd_param)

            if last_lsd is None or last_rsd is None:
                TD_reconstructed, label_td = td_denoise(de_array_tensor[..., 0], td_param)
            else:
                TD_reconstructed, label_td = td_new(
                    last_lsd, last_rsd, LSD_reconstructed, RSD_reconstructed,
                    de_array_tensor[..., 0], td_param)

            last_lsd = LSD_reconstructed
            last_rsd = RSD_reconstructed

            # 保存降噪后的结果
            sdl_save_path = os.path.join(sdl_dir, f"sdl_{de_file}")
            sdr_save_path = os.path.join(sdr_dir, f"sdr_{de_file}")
            td_save_path = os.path.join(td_dir, f"td_{de_file}")

            np.save(sdl_save_path, LSD_reconstructed.cpu().numpy())
            np.save(sdr_save_path, RSD_reconstructed.cpu().numpy())
            np.save(td_save_path, TD_reconstructed.cpu().numpy())

            print(f"Saved sdl to {sdl_save_path}")
            print(f"Saved sdr to {sdr_save_path}")
            print(f"Saved td to {td_save_path}")


if __name__ == '__main__':
    root_path = f"/home/leshannx/Documents/Tianmou_Evaluation/xinxinli/raw_rod_cam0"  # 你的输入文件夹
    save_path = f"/home/leshannx/Documents/Tianmou_Evaluation/xinxinli/denoised_output_cam0"     # 保存降噪结果的文件夹

    td_param = {
        'var_fil_ksize': 3,
        'var_th': 1,
        'adapt_th_min': 3,
        'adapt_th_max': 8,
    }
    sd_param = {
        'var_fil_ksize': 3,
        'var_th': 1,
        'adapt_th_min': 3,
        'adapt_th_max': 9,
    }
    proces_scene(root_path, td_param, sd_param, save_path)

    # Two light levels：bright/lowlight
    # mutiscene has 6 scenes：scene1/scene2/scene3/scene4/scene5/scene6
    # std has 7 scenes：scene1/scene2/scene3/scene4/scene5/scene6/scene7
