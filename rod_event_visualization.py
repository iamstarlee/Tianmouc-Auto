import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import cv2  # 需要安装 opencv-python

def SD2XY(sd_raw: torch.Tensor):
    '''
    input: [h,w,2]/[2,h,w]/[n,2,h,w] torch tensor (CPU)
    output: sdx, sdy 同维度变换后tensor
    坐标变换规则参照http://www.tianmouc.cn:40000/tianmoucv/introduction.html
    '''
    if len(sd_raw.shape) == 3:
        assert (sd_raw.shape[2] == 2 or sd_raw.shape[0] == 2)
        if sd_raw.shape[2] == 2:
            sd = sd_raw.permute(2, 0, 1).unsqueeze(0)  # [h,w,c]->[1,c,h,w]
        else:
            sd = sd_raw.unsqueeze(0)
    else:
        assert (len(sd_raw.shape) == 4 and sd_raw.shape[1] == 2)
        sd = sd_raw

    b, c, h, w = sd.shape

    # 按规则切片，步长2隔行取
    sdul = sd[:, 0:1, 0::2, :]  # 上行偶数行
    sdll = sd[:, 0:1, 1::2, :]  # 下行奇数行
    sdur = sd[:, 1:2, 0::2, :]
    sdlr = sd[:, 1:2, 1::2, :]

    # resize到目标尺寸(h, 2*w)，用cv2.resize实现双线性插值
    target_size = (w * 2, h)  # cv2尺寸是 (width, height)

    def resize_tensor(tensor):
        # tensor: [b,c,H,W], 这里只处理单batch单通道，取[0,0]
        np_img = tensor[0, 0].numpy()
        resized = cv2.resize(np_img, target_size, interpolation=cv2.INTER_LINEAR)
        return torch.from_numpy(resized).unsqueeze(0).unsqueeze(0).float()

    sdul_r = resize_tensor(sdul)
    sdll_r = resize_tensor(sdll)
    sdur_r = resize_tensor(sdur)
    sdlr_r = resize_tensor(sdlr)

    # 计算 sdx, sdy
    sdx = ((sdul_r + sdll_r) / 1.414 - (sdur_r + sdlr_r) / 1.414) / 2
    sdy = ((sdur_r - sdlr_r) / 1.414 + (sdul_r - sdll_r) / 1.414) / 2

    # 根据输入形状返回结果形状，兼容输入形状为3维或4维
    if len(sd_raw.shape) == 3:
        return sdx.squeeze(0).squeeze(0), sdy.squeeze(0).squeeze(0)
    else:
        return sdx.squeeze(1), sdy.squeeze(1)


import numpy as np
import cv2

def event_visualization(diff_data, thresh=1.0, gain=1.0):
    H, W = diff_data.shape

    # --- 背景白色 ---
    rgb = np.ones((H, W, 3), dtype=np.float32) * 255  # BGR = 白色背景

    # --- 极性掩码 ---
    pos_mask = diff_data > thresh
    neg_mask = diff_data < -thresh
    diff_scaled = np.abs(diff_data) * gain
    diff_scaled = np.clip(diff_scaled, 0, 255)

    # --- 正极性：深蓝调 (BGR) ---
    #   蓝色主导：B 高，G 中，R 低
    base_blue = np.array([180, 120, 80], dtype=np.float32)  # 深蓝基色
    white = np.array([255, 255, 255], dtype=np.float32)

    rgb[pos_mask] = (
        white - (white - base_blue) * (diff_scaled[pos_mask][:, None] / 255.0)
    )

    # --- 负极性：黑色 ---
    rgb[neg_mask] = 255 - diff_scaled[neg_mask][:, None]

    # --- 转成 RGB (给 matplotlib 用，如果 OpenCV 显示就不需要) ---
    # rgb = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_BGR2RGB)

    return rgb.astype(np.uint8)

import numpy as np

def event_visualization_td(diff_data, thresh=1.0, gain=1.0):
    H, W = diff_data.shape

    # --- 背景白色 ---
    rgb = np.ones((H, W, 3), dtype=np.float32) * 255  

    # --- 极性掩码 ---
    pos_mask = diff_data > thresh
    neg_mask = diff_data < -thresh
    diff_scaled = np.abs(diff_data) * gain
    diff_scaled = np.clip(diff_scaled, 0, 255)

    # --- 正极性：红色 (BGR: (80, 80, 200)) ---
    base_red = np.array([80, 80, 200], dtype=np.float32)
    white = np.array([255, 255, 255], dtype=np.float32)

    rgb[pos_mask] = (
        white - (white - base_red) * (diff_scaled[pos_mask][:, None] / 255.0)
    )

    # --- 负极性：蓝色 (BGR: (200, 80, 80)) ---
    base_blue = np.array([200, 80, 80], dtype=np.float32)

    rgb[neg_mask] = (
        white - (white - base_blue) * (diff_scaled[neg_mask][:, None] / 255.0)
    )

    return rgb.astype(np.uint8)



def visualize_2x2_with_td_and_sd(i, input_array, denoised_paths, thresh=1.0, gain=1.0, save_path=None, save_npy_path=None):
    import matplotlib.pyplot as plt
    import torch
    import numpy as np

    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    H, W = input_array.shape[1], input_array.shape[2]

    # 第一行：td通道，直接差异比较
    td_raw = input_array[0]
    td_denoised = np.load(denoised_paths['td'])
    diff_raw = cv2.resize(td_raw, (320, 160))
    diff_denoised = cv2.resize(td_denoised, (320, 160))

    vis_raw = event_visualization(diff_raw, thresh, gain)
    vis_denoised = event_visualization(diff_denoised, thresh, gain)

    axes[0, 0].imshow(cv2.cvtColor(vis_raw, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('td Raw')
    axes[0, 1].imshow(cv2.cvtColor(vis_denoised, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title('td Denoised')
    for ax in axes[0]:
        ax.axis('off')

    # 第二行：sdl + sdr 通道，用 SD2XY 处理后比较
    sdl_raw = input_array[1]
    sdr_raw = input_array[2]
    sdl_denoised = np.load(denoised_paths['sdl'])
    sdr_denoised = np.load(denoised_paths['sdr'])

    sd_raw_np = np.stack([sdl_raw, sdr_raw], axis=2)  # (H, W, 2)
    sd_denoised_np = np.stack([sdl_denoised, sdr_denoised], axis=2)
    sd_raw = torch.from_numpy(sd_raw_np).float()
    sd_denoised = torch.from_numpy(sd_denoised_np).float()

    sdx_raw, sdy_raw = SD2XY(sd_raw)
    sdx_denoised, sdy_denoised = SD2XY(sd_denoised)

    diff_raw = sdx_raw.numpy() + sdy_raw.numpy()
    diff_denoised = sdx_denoised.numpy() + sdy_denoised.numpy()

    vis_raw = event_visualization(diff_raw, thresh, gain)
    sd_vis_denoised = event_visualization(diff_denoised, thresh, gain)
    td_vis_denoised = event_visualization(td_raw, thresh, gain)

    axes[1, 0].imshow(cv2.cvtColor(vis_raw, cv2.COLOR_BGR2RGB))
    axes[1, 0].set_title('sdl+sdr Raw SD2XY')
    axes[1, 1].imshow(cv2.cvtColor(sd_vis_denoised, cv2.COLOR_BGR2RGB))
    axes[1, 1].set_title('sdl+sdr Denoised SD2XY')
    for ax in axes[1]:
        ax.axis('off')

    plt.tight_layout()

    if save_path is not None and i % 200 == 0:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close(fig)

    # 保存 diff_denoised 数组为 npy 文件（这里保存 sdl+sdr 叠加的差异）
    cv2.imshow("vis_sd", sd_vis_denoised)
    cv2.imshow("vis_td", td_vis_denoised)
    cv2.waitKey(1)
    if save_npy_path is not None:
        np.save(save_npy_path, diff_raw)

# def visualize_2x2_with_td_and_sd(
#     i,
#     input_array,
#     denoised_paths,
#     thresh=1.0,
#     gain=1.0,
#     save_dir=None  # 新增：统一保存路径
# ):
#     import matplotlib.pyplot as plt
#     import torch
#     import numpy as np
#     import cv2
#     import os

#     fig, axes = plt.subplots(2, 2, figsize=(8, 8))
#     H, W = input_array.shape[1], input_array.shape[2]

#     # 第一行：td通道
#     td_raw = input_array[0]
#     td_denoised = np.load(denoised_paths['td'])
#     diff_raw = cv2.resize(td_raw, (320, 160))
#     diff_denoised = cv2.resize(td_denoised, (320, 160))

#     vis_raw = event_visualization(diff_raw, thresh, gain)
#     vis_denoised = event_visualization(diff_denoised, thresh, gain)

#     axes[0, 0].imshow(vis_raw)
#     axes[0, 0].set_title('td Raw')
#     axes[0, 1].imshow(vis_denoised)
#     axes[0, 1].set_title('td Denoised')
#     for ax in axes[0]:
#         ax.axis('off')

#     # 第二行：sdl + sdr 通道
#     sdl_raw = input_array[1]
#     sdr_raw = input_array[2]
#     sdl_denoised = np.load(denoised_paths['sdl'])
#     sdr_denoised = np.load(denoised_paths['sdr'])

#     sd_raw_np = np.stack([sdl_raw, sdr_raw], axis=2)
#     sd_denoised_np = np.stack([sdl_denoised, sdr_denoised], axis=2)
#     sd_raw = torch.from_numpy(sd_raw_np).float()
#     sd_denoised = torch.from_numpy(sd_denoised_np).float()

#     # 计算 Ix, Iy
#     sdx_raw, sdy_raw = SD2XY(sd_raw)
#     sdx_denoised, sdy_denoised = SD2XY(sd_denoised)

#     diff_raw = sdx_raw.numpy() + sdy_raw.numpy()
#     diff_denoised = sdx_denoised.numpy() + sdy_denoised.numpy()

#     vis_raw = event_visualization(diff_raw, thresh, gain)
#     vis_denoised = event_visualization(diff_denoised, thresh, gain)

#     axes[1, 0].imshow(vis_raw)
#     axes[1, 0].set_title('sdl+sdr Raw SD2XY')
#     axes[1, 1].imshow(vis_denoised)
#     axes[1, 1].set_title('sdl+sdr Denoised SD2XY')
#     for ax in axes[1]:
#         ax.axis('off')

#     plt.tight_layout()

#     # 如果指定保存目录，就创建 Ix / Iy 子目录
#     ix_path, iy_path = None, None
#     if save_dir is not None:
#         ix_dir = os.path.join(save_dir, "Ix")
#         iy_dir = os.path.join(save_dir, "Iy")
#         os.makedirs(ix_dir, exist_ok=True)
#         os.makedirs(iy_dir, exist_ok=True)

#         ix_path = os.path.join(ix_dir, f"Ix_{i:06d}.npy")
#         iy_path = os.path.join(iy_dir, f"Iy_{i:06d}.npy")

#         np.save(ix_path, sdx_denoised.numpy())
#         np.save(iy_path, sdy_denoised.numpy())

#     # 可视化窗口
#     cv2.imshow("vis", vis_denoised)
#     cv2.waitKey(1)

#     # 返回 Ix/Iy 路径
#     return {
#         "ix_path": ix_path,
#         "iy_path": iy_path
#     }




def main(rod_dir, denoised_dir, out_dir, thresh=1.0, gain=1.0):
    os.makedirs(out_dir, exist_ok=True)
    rod_files = sorted([f for f in os.listdir(rod_dir) if f.endswith('.npy')])

    frame_step = 1  # 每隔100帧处理一次

    for i in range(0, len(rod_files), frame_step):
        rod_file = rod_files[i]
        rod_path = os.path.join(rod_dir, rod_file)
        input_array = np.load(rod_path)

        denoised_paths = {}
        for ch in ['td', 'sdl', 'sdr']:
            denoised_file = f"{ch}_{rod_file}"
            denoised_path = os.path.join(denoised_dir, ch, denoised_file)
            if not os.path.exists(denoised_path):
                print(f"Warning: {denoised_path} 不存在，跳过此帧")
                break
            denoised_paths[ch] = denoised_path
        else:
            print(f"Processing {rod_file} ...")
            save_path = os.path.join(out_dir, rod_file.replace('.npy', '.png'))
            save_npy_path = os.path.join(out_dir, f"{i}_{rod_file.replace('.npy', '_diff_denoised.npy')}")
            visualize_2x2_with_td_and_sd(i, input_array, denoised_paths, thresh, gain, save_path=save_path, save_npy_path=save_npy_path)
            # visualize_2x2_with_td_and_sd(i, input_array, denoised_paths, thresh, gain, out_dir)

if __name__ == "__main__":
    rod_dir = f"/home/leshannx/Documents/Tianmou_Evaluation/xinxinli/raw_rod_cam0"
    denoised_dir = f"/home/leshannx/Documents/Tianmou_Evaluation/xinxinli/denoised_output_cam0"
    out_dir = f"/home/leshannx/Documents/Tianmou_Evaluation/xinxinli/Ixy"
    main(rod_dir, denoised_dir, out_dir, thresh=1.0, gain=10.0)

    # rod_dir = "rod_output_cam0"
    # denoised_dir = "denoised_output_cam0"
    # out_dir = "save_ixy"
    # main(rod_dir, denoised_dir, out_dir, thresh=1.0, gain=10.0)
