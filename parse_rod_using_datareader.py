#!/usr/bin/env python3
"""
使用TianmoucDataReader解析ROD数据
这种方式可以自动处理CONE和ROD的时间同步
支持单目和双目数据
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib
from tianmoucv.data import TianmoucDataReader
import cv2
import shutil

# 设置中文字体支持，避免中文显示警告
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

def apply_elliptical_roi(img, radius_ratio=0.4, offset_x=0, offset_y=0, x_scale=0.5):
    """
    在图像中心生成一个椭圆形 ROI 掩膜，非ROI区域填充ROI边缘颜色。
    :param img: 输入图像 (H,W) 或 (H,W,3)
    :param radius_ratio: 半径相对于图像最短边的比例 (0~1)，作为y方向半径
    :param offset_x: 圆心在水平方向的偏移（+右，-左）
    :param offset_y: 圆心在竖直方向的偏移（+下，-上）
    :param x_scale: x方向半径相对于y方向的比例（默认0.5）
    :return: 掩膜后的图像, 掩膜
    """
    h, w = img.shape[:2]
    center = (w // 2 + offset_x, h // 2 + offset_y)
    radius_y = int(min(h, w) * radius_ratio)
    radius_x = int(radius_y * x_scale)

    # 创建掩膜 (ROI = 1, 外部 = 0)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(mask, center, (radius_x, radius_y), 0, 0, 360, 255, -1)

    if img.dtype == np.uint8:
        roi = img.copy()
        # 获取ROI边缘像素颜色
        edge_pixels = cv2.bitwise_and(img, img, mask=mask)
        edge_color = cv2.mean(edge_pixels, mask=mask)[0]  # 灰度平均
        # 非ROI区域填充边缘颜色
        result = img.copy()
        result[mask == 0] = int(edge_color)
    else:
        roi = img.copy().astype(np.float32)   # 确保 float32
        mask = mask.astype(np.uint8)          # 确保 uint8
        mask_3c = np.repeat(mask[:, :, None], 3, axis=2)

        # 边缘颜色（mask 内像素的平均值）
        edge_color = cv2.mean(img, mask=mask)[:3]  # 结果是 float

        result = img.copy().astype(np.float32)
        for c in range(3):
            result[:, :, c][mask == 0] = edge_color[c]

    return result, mask



def parse_rod_with_datareader(data_path, output_dir, N=1, camera_idx=0, parse_rod=True, parse_cone=True):
    """
    使用TianmoucDataReader解析ROD和/或CONE RGB数据
    
    参数:
        data_path: 包含cone和rod子目录的数据路径
        output_dir: 输出目录
        N: 每个样本包含N+1帧COP (默认为1)
        camera_idx: 相机编号，0为左相机，1为右相机
        parse_rod: 是否解析ROD数据（默认True）
        parse_cone: 是否解析CONE RGB数据（默认True）
    """
    
    # 创建输出目录
    camera_suffix = f"_cam{camera_idx}"
    camera_output_dir = output_dir + camera_suffix
    os.makedirs(camera_output_dir, exist_ok=True)
    
    # 初始化数据读取器
    dataset = TianmoucDataReader(data_path, N=N, camera_idx=camera_idx)
    print(f"相机{camera_idx}: 数据集包含 {len(dataset)} 个样本")
    
    total_rod_frames = 0
    
    # 遍历所有样本
    for sample_idx in range(len(dataset)):
        sample = dataset[sample_idx]
        
        print(f"\n相机{camera_idx} 样本 {sample_idx}:")
        
        # 处理ROD数据
        if parse_rod:
            # 获取ROD数据 (rawDiff)
            # 形状: [3, num_frames, 160, 160]
            # 3个通道分别是: TD, SDL, SDR
            rod_data = sample['rawDiff']
            
            # 获取数据比例和元信息
            data_ratio = sample.get('dataRatio', 25)  # 默认25
            meta = sample.get('meta', {})
            
            print(f"  ROD数据形状: {rod_data.shape}")
            print(f"  数据比例: {data_ratio}")
            print(f"  包含 {rod_data.shape[1]} 帧ROD数据")
            
            # 保存每一帧ROD数据
            for frame_idx in range(rod_data.shape[1]):
                # 提取单帧数据 [3, 160, 160]
                frame_data = rod_data[:, frame_idx, :, :].numpy()
                
                # 生成文件名
                global_frame_idx = total_rod_frames + frame_idx
                output_filename = f"rod_cam{camera_idx}_sample{sample_idx:03d}_frame{frame_idx:04d}_global{global_frame_idx:06d}.npy"
                output_path = os.path.join(camera_output_dir, output_filename)
                
                # 保存数据
                np.save(output_path, frame_data)
            
            total_rod_frames += rod_data.shape[1]
        
        # 处理CONE相机的RGB图像
        if parse_cone:
            # 检查并保存所有可用的CONE帧 (F0, F1, F2, ...)
            cone_frame_count = 0
            for i in range(10):  # 最多检查10个CONE帧
                cone_key = f'F{i}'
                if cone_key in sample:
                    cone_data = sample[cone_key].numpy()  # 通常是 [H, W, 3] 或 [3, H, W]
                    
                    # 确保数据是 [H, W, 3] 格式
                    if cone_data.shape[0] == 3 and len(cone_data.shape) == 3:
                        # 从 [3, H, W] 转换为 [H, W, 3]
                        cone_data = cone_data.transpose(1, 2, 0)
                    
                    # 确保数据在0-255范围内
                    if cone_data.max() <= 1.0:
                        cone_data = (cone_data * 255).astype(np.uint8)
                    else:
                        cone_data = cone_data.astype(np.uint8)
                    
                    # 保存为PNG图像
                    cone_filename = f"cone_cam{camera_idx}_sample{sample_idx:03d}_{cone_key}.png"
                    cone_path = os.path.join(camera_output_dir, cone_filename)
                    # Image.fromarray(cone_data).save(cone_path)
                    
                    # 也可以保存原始npy数据
                    cone_npy_filename = f"cone_cam{camera_idx}_sample{sample_idx:03d}_{cone_key}.npy"
                    cone_npy_path = os.path.join(camera_output_dir, cone_npy_filename)
                    np.save(cone_npy_path, sample[cone_key].numpy())
                    
                    cone_frame_count += 1
                    
                    # 如果有HDR版本，也保存
                    hdr_key = f'{cone_key}_HDR'
                    if hdr_key in sample:
                        hdr_data = sample[hdr_key].numpy()
                        if hdr_data.shape[0] == 3 and len(hdr_data.shape) == 3:
                            hdr_data = hdr_data.transpose(1, 2, 0)
                        
                        # HDR数据可能需要特殊处理
                        hdr_data = np.clip(hdr_data, 0, 1)
                        hdr_data = (hdr_data * 255).astype(np.uint8)
                        
                        hdr_filename = f"cone_cam{camera_idx}_sample{sample_idx:03d}_{cone_key}_HDR.png"
                        hdr_path = os.path.join(camera_output_dir, hdr_filename)
                        Image.fromarray(hdr_data).save(hdr_path)
            
            if cone_frame_count > 0:
                print(f"  保存了 {cone_frame_count} 帧CONE RGB图像")
    
    print(f"\n相机{camera_idx} 完成！")
    if parse_rod:
        print(f"  共保存 {total_rod_frames} 帧ROD数据")
    print(f"  输出目录: {camera_output_dir}")
    
    return total_rod_frames


def extract_rod_only_simple(data_path, output_dir, camera_idx=0, N=1, parse_rod=True, parse_cone=True):
    """
    简化版：提取ROD和/或CONE数据，使用TianmoucDataSampleParser
    """
    import os
    from tianmoucv.data import TianmoucDataReader
    from tianmoucv.data.tianmoucDataSampleParser import TianmoucDataSampleParser
    from PIL import Image
    
    camera_suffix = f"_cam{camera_idx}"
    camera_output_dir = output_dir + camera_suffix
    os.makedirs(camera_output_dir, exist_ok=True)
    
    # 读取数据，传递 N 参数
    dataset = TianmoucDataReader(data_path, N=N, camera_idx=camera_idx)
    
    total_frames = 0
    
    for idx in range(len(dataset)):
        sample = dataset[idx]
        parser = TianmoucDataSampleParser(sample)
        
        # 获取数据比例
        data_rate = parser.get_data_rate()
        
        print(f"相机{camera_idx} 样本 {idx}:")
        
        # 处理ROD数据
        if parse_rod:
            # 获取原始差分数据（未上采样）
            rod_data = parser.get_tsd(ifUpSampled=False)  # [3, frames, 160, 160]
            
            if rod_data is not None:
                num_frames = rod_data.shape[1]
                print(f"  {num_frames} 帧ROD数据")
                
                # 保存每一帧
                for frame_idx in range(num_frames):
                    frame = rod_data[:, frame_idx, :, :].numpy()
                    filename = f"rod_cam{camera_idx}_idx{idx:03d}_frame{frame_idx:04d}.npy"
                    filepath = os.path.join(camera_output_dir, filename)
                    img_hwc = np.transpose(frame, (1, 2, 0))  # (H, W, C)
                    frame, mask = apply_elliptical_roi(img_hwc, radius_ratio=2, offset_x=-1, offset_y=-1)
                    cv2.imshow("rod", cv2.resize(frame.astype(np.uint8), (320, 160)))
                    cv2.waitKey(1)
                    frame = np.transpose(frame, (2, 0, 1))  # (C, H, W)
                    np.save(filepath, frame)
                    total_frames += 1
        
        # 处理CONE RGB图像
        if parse_cone:
            cone_count = 0
            for i in range(parser.N + 1):  # parser.N + 1 是CONE帧的数量
                cone_key = f'F{i}'
                if cone_key in sample:
                    cone_data = sample[cone_key].numpy()
                    
                    # 确保数据是 [H, W, 3] 格式
                    if cone_data.shape[0] == 3 and len(cone_data.shape) == 3:
                        cone_data = cone_data.transpose(1, 2, 0)
                    
                    # （可选）如果数据是 0~1，可以保留 float，不再转 uint8
                    # 如果你一定要保存成 uint8，可以保留下面几行：
                    if cone_data.max() <= 1.0:
                        cone_data = (cone_data * 255).astype(np.uint8)
                    else:
                        cone_data = cone_data.astype(np.uint8)

                    # 保存为 NPY 文件
                    cone_filename = f"cone_cam{camera_idx}_idx{idx:03d}_{cone_key}.npy"
                    cone_path = os.path.join(camera_output_dir, cone_filename)
                    cone_data = cv2.resize(cone_data, (160, 160))
                    cone_data, mask = apply_elliptical_roi(cone_data, radius_ratio=2, offset_x=-1, offset_y=-1)
                    cv2.imshow("cone", cv2.resize(cone_data, (320, 160)))
                    cv2.waitKey(1)
                    np.save(cone_path, cone_data)

                    cone_count += 1
            
            if cone_count > 0:
                print(f"  保存了 {cone_count} 帧CONE RGB图像")
    
    print(f"相机{camera_idx} 共提取 {total_frames} 帧ROD数据")
    return total_frames


def parse_stereo_rod_data(data_path, output_dir, N=1):
    """
    解析双目ROD数据
    """
    print("开始处理双目ROD数据...")
    
    # 处理左相机（camera_idx=0）
    print("\n=== 处理左相机数据 ===")
    left_frames = parse_rod_with_datareader(data_path, output_dir, N=N, camera_idx=0)
    
    # 处理右相机（camera_idx=1）
    print("\n=== 处理右相机数据 ===")
    right_frames = parse_rod_with_datareader(data_path, output_dir, N=N, camera_idx=1)
    
    # print(f"\n双目数据处理完成！")
    # print(f"左相机: {left_frames} 帧")
    # print(f"右相机: {right_frames} 帧")
    # print(f"总计: {left_frames + right_frames} 帧")
    
    return left_frames, right_frames


def extract_stereo_rod_simple(data_path, output_dir, N=1, parse_rod=True, parse_cone=True):
    """
    简化版双目数据提取
    """
    if parse_rod and parse_cone:
        print("开始提取双目ROD和CONE RGB数据...")
    elif parse_rod:
        print("开始提取双目ROD数据...")
    elif parse_cone:
        print("开始提取双目CONE RGB数据...")
    
    # 处理左相机（camera_idx=0）
    print("\n=== 提取左相机数据 ===")
    left_frames = extract_rod_only_simple(data_path, output_dir, camera_idx=0, N=N, 
                                         parse_rod=parse_rod, parse_cone=parse_cone)
    
    # 处理右相机（camera_idx=1）
    print("\n=== 提取右相机数据 ===")
    right_frames = extract_rod_only_simple(data_path, output_dir, camera_idx=1, N=N, 
                                          parse_rod=parse_rod, parse_cone=parse_cone)
    
    print(f"\n双目数据提取完成！")
    print(f"左相机: {left_frames} 帧")
    print(f"右相机: {right_frames} 帧")
    print(f"总计: {left_frames + right_frames} 帧")
    
    return left_frames, right_frames


def visualize_rod_sequence(output_dir, camera_idx=0, start_frame=0, num_frames=10):
    """
    可视化一系列ROD帧，观察时间变化
    """
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from IPython.display import HTML
    
    # 查找指定相机的npy文件
    camera_suffix = f"_cam{camera_idx}"
    camera_output_dir = output_dir + camera_suffix
    
    if not os.path.exists(camera_output_dir):
        print(f"未找到相机{camera_idx}的输出目录: {camera_output_dir}")
        return
    
    npy_files = sorted([f for f in os.listdir(camera_output_dir) if f.endswith('.npy') and 'rod' in f])
    
    if not npy_files:
        print(f"未找到相机{camera_idx}的ROD数据文件")
        return
    
    # 选择要显示的帧
    selected_files = npy_files[start_frame:start_frame+num_frames]
    print(f"相机{camera_idx} 选择了 {len(selected_files)} 帧进行可视化")
    # 创建图形
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    titles = ['时间差分(TD)', '空间差分左(SDL)', '空间差分右(SDR)']
    
    # 初始化图像
    ims = []
    for i in range(3):
        data = np.load(os.path.join(camera_output_dir, selected_files[0]))
        im = axes[i].imshow(data[i], cmap='gray', animated=True)
        axes[i].set_title(titles[i])
        axes[i].axis('off')
        ims.append(im)
    
    # 更新函数
    def update(frame_idx):
        data = np.load(os.path.join(camera_output_dir, selected_files[frame_idx]))
        for i in range(3):
            ims[i].set_data(data[i])
            ims[i].set_clim(vmin=data[i].min(), vmax=data[i].max())
        fig.suptitle(f'相机{camera_idx} - 帧: {selected_files[frame_idx]}')
        return ims
    
    # 创建动画
    ani = animation.FuncAnimation(fig, update, frames=len(selected_files), 
                                 interval=100, blit=True)
    
    plt.tight_layout()
    return ani


def visualize_stereo_comparison(output_dir, start_frame=0, num_frames=5):
    """
    同时可视化左右相机的ROD数据进行对比
    """
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    
    # 查找左右相机的npy文件
    left_dir = output_dir + "_cam0"
    right_dir = output_dir + "_cam1"
    
    if not os.path.exists(left_dir) or not os.path.exists(right_dir):
        print("未找到左右相机的输出目录")
        return
    
    left_files = sorted([f for f in os.listdir(left_dir) if f.endswith('.npy') and 'rod' in f])
    right_files = sorted([f for f in os.listdir(right_dir) if f.endswith('.npy') and 'rod' in f])
    
    if not left_files or not right_files:
        print("未找到左右相机的ROD数据文件")
        return
    
    # 选择要显示的帧
    selected_left = left_files[start_frame:start_frame+num_frames]
    selected_right = right_files[start_frame:start_frame+num_frames]
    
    # 创建图形 (2行3列：左相机TD/SDL/SDR，右相机TD/SDL/SDR)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    titles = ['时间差分(TD)', '空间差分左(SDL)', '空间差分右(SDR)']
    
    # 初始化图像
    ims = []
    for i in range(3):
        # 左相机
        left_data = np.load(os.path.join(left_dir, selected_left[0]))
        im_left = axes[0, i].imshow(left_data[i], cmap='gray', animated=True)
        axes[0, i].set_title(f'左相机 - {titles[i]}')
        axes[0, i].axis('off')
        ims.append(im_left)
        
        # 右相机
        right_data = np.load(os.path.join(right_dir, selected_right[0]))
        im_right = axes[1, i].imshow(right_data[i], cmap='gray', animated=True)
        axes[1, i].set_title(f'右相机 - {titles[i]}')
        axes[1, i].axis('off')
        ims.append(im_right)
    
    # 更新函数
    def update(frame_idx):
        left_data = np.load(os.path.join(left_dir, selected_left[frame_idx]))
        right_data = np.load(os.path.join(right_dir, selected_right[frame_idx]))
        
        for i in range(3):
            # 更新左相机
            ims[i*2].set_data(left_data[i])
            ims[i*2].set_clim(vmin=left_data[i].min(), vmax=left_data[i].max())
            # 更新右相机
            ims[i*2+1].set_data(right_data[i])
            ims[i*2+1].set_clim(vmin=right_data[i].min(), vmax=right_data[i].max())
        
        fig.suptitle(f'双目对比 - 帧: {frame_idx}')
        return ims
    
    # 创建动画
    ani = animation.FuncAnimation(fig, update, frames=len(selected_left), 
                                 interval=200, blit=True)
    
    plt.tight_layout()
    return ani


if __name__ == "__main__":
    # 双目数据路径
    data_path = f"/home/leshannx/Documents/Tianmou_Evaluation/xinxinli"  # 包含cone和rod子目录的双目数据


    # data_path = '/home/leshannx/Documents/Tianmou_Evaluation/xinxinli'
    output_dir = "xinxinli"  # 输出目录改为cone_rgb_output
    # output_dir = "cone_output"  # 输出目录改为cone_rgb_output
    
    # 只解析CONE RGB图像，不解析ROD数据
    # print("=== 提取CONE RGB图像 ===")

    # --- 每次运行前清空目录 ---
    output_dir = f"xinxinli/raw_cone"  # 输出目录改为cone_rgb_output
    extract_stereo_rod_simple(data_path, output_dir, N=1, parse_rod=False, parse_cone=True)
    
    # 如果需要同时解析ROD和CONE数据：
    # extract_stereo_rod_simple(data_path, output_dir, N=1, parse_rod=True, parse_cone=True)
    
    # 如果只需要ROD数据：
    output_dir = f"xinxinli/raw_rod"
    extract_stereo_rod_simple(data_path, output_dir, N=1, parse_rod=True, parse_cone=False)
    
    # 可视化左相机前10帧
    # ani_left = visualize_rod_sequence(output_dir, camera_idx=0, start_frame=0, num_frames=86)
    
    # # 可视化右相机前10帧
    # ani_right = visualize_rod_sequence(output_dir, camera_idx=1, start_frame=0, num_frames=9900)
    
    # # 双目对比可视化前5帧
    # ani_stereo = visualize_stereo_comparison(output_dir, start_frame=0, num_frames=500)
    
    # plt.show()
