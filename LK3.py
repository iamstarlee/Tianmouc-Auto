import numpy as np
import cv2
import time
import re
import os
from rod_event_visualization import event_visualization

# Poisson_denoised_out, raw_cone_cam0, Ixy, denoised_output_cam0/td#
# ============ 参数区域 ============
data = 900
image_folder = f'xinxinli/denoised_output_cam0/td'
video_name = 'rgbLK.mp4'
fps = 120  # 视频帧率

trajectory_len = 100000
detect_interval = 5
frame_idx = 0

# Lucas-Kanade 光流参数
lk_params = dict(winSize=(21, 21),
                 maxLevel=1,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# ---- 选择特征点检测方法 ----
# options: "HARRIS", "FAST", "ORB"
detector_type = "HARRIS"

# Harris / Shi-Tomasi 参数
harris_params = dict(maxCorners=100,
                     qualityLevel=0.03,
                     minDistance=10,
                     useHarrisDetector=True,
                     k=0.04)

# FAST 参数
fast_params = dict(threshold=30,
                   nonmaxSuppression=True,
                   type=cv2.FAST_FEATURE_DETECTOR_TYPE_9_16)

# ORB 参数
orb_params = dict(nfeatures=100,
                  scaleFactor=1.2,
                  nlevels=8,
                  edgeThreshold=31,
                  firstLevel=0,
                  WTA_K=2,
                  scoreType=cv2.ORB_HARRIS_SCORE,  # 或 cv2.ORB_FAST_SCORE
                  patchSize=31,
                  fastThreshold=30)

# ============ 文件读取函数 ============
def sorted_npy_files(folder):
    files = [f for f in os.listdir(folder) if f.endswith('.npy')]
    def key_fn(fname):
        b = os.path.basename(fname)
        m = re.search(r'idx(\d+)_frame(\d+)', b)
        if m:
            return (int(m.group(1)), int(m.group(2)), b)
        nums = re.findall(r'(\d+)', b)
        if len(nums) >= 2:
            return (int(nums[-2]), int(nums[-1]), b)
        if len(nums) == 1:
            return (int(nums[0]), -1, b)
        return (float('inf'), float('inf'), b)
    files.sort(key=key_fn)
    return [os.path.join(folder, f) for f in files]

files = sorted_npy_files(image_folder)  # 返回排序好的文件名（不带路径）
image_files = files  # 拼完整路径

# 视频输出
video_writer = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), fps, (320, 160))

trajectories = []

# ============ 结果保存文件 ============
# 从 image_folder 提取最后一级作为图像形式
image_type = os.path.basename(image_folder)
# 结果保存文件
result_filename = "OFres/" + f"{detector_type}_multi_{data}_{image_type}.txt"

# 每次运行前清空文件并写表头
# with open(result_filename, "w") as f:
#     f.write("Frame,AvgReprojError,AvgTrajectoryLength\n")

# 如果文件不存在，写入表头
# if not os.path.exists(result_filename):
#     with open(result_filename, "w") as f:
#         f.write("Frame,AvgReprojError,AvgTrajectoryLength\n")


end = 35

if "cone" not in files[0]:
    end *= 25
# ============ 主循环 ============
for i in range(1, end, 1):
    start = time.time()

    # 读取 npy 数据
    prev_data = np.load(image_files[i-1])
    frame_data = np.load(image_files[i])

    prev_data_or = cv2.resize(prev_data, (320, 160))
    frame_data_or = cv2.resize(frame_data, (320, 160))

    prev_data = ((prev_data_or + 50) * 2).clip(0, 255)
    frame_data = ((frame_data_or + 50) * 2).clip(0, 255)

    # 转灰度
    if prev_data.ndim == 3 and prev_data.shape[2] == 3:
        prev_gray = cv2.cvtColor(prev_data_or.astype(np.uint8), cv2.COLOR_BGR2GRAY)
        img_viz = frame_data_or.astype(np.uint8)
    else:
        prev_gray = prev_data.astype(np.uint8)
        img_viz = event_visualization(prev_data_or * 10)

    if frame_data.ndim == 3 and frame_data.shape[2] == 3:
        frame_gray = cv2.cvtColor(frame_data_or.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    else:
        frame_gray = frame_data.astype(np.uint8)

    # ============ LK 光流 ============
    reproj_errors = []  # 重投影误差
    if len(trajectories) > 0:
        img0, img1 = prev_gray, frame_gray
        p0 = np.float32([trajectory[-1] for trajectory in trajectories]).reshape(-1, 1, 2)
        p1, _st, _err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
        p0r, _st, _err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)

        d = abs(p0 - p0r).reshape(-1, 2).max(-1)
        good = d < 1

        reproj_errors = np.linalg.norm(p1.reshape(-1, 2) - p0.reshape(-1, 2), axis=1)

        new_trajectories = []
        for trajectory, (x, y), good_flag, dist in zip(trajectories, p1.reshape(-1, 2), good, reproj_errors):
            if not good_flag:
                continue
            trajectory.append((x, y))
            if len(trajectory) > trajectory_len:
                del trajectory[0]
            new_trajectories.append(trajectory)

            cv2.circle(img_viz, (int(x), int(y)), 3, (0, 0, 255), -1)

        trajectories = new_trajectories
        cv2.polylines(img_viz, [np.int32(traj) for traj in trajectories], False, (30, 144, 255), 1)

    # ============ 特征点检测 ============
    if frame_idx % detect_interval == 0:
        h, w = frame_gray.shape
        mask = np.zeros((h, w), dtype=np.uint8)
        center = (w // 2 - 1, h // 2 -1)
        radius = int(min(h, w) * 0.41)
        cv2.circle(mask, center, radius, 255, -1)

        if detector_type.upper() == "HARRIS":
            p = cv2.goodFeaturesToTrack(frame_gray, **harris_params, mask=mask)

        elif detector_type.upper() == "FAST":
            fast = cv2.FastFeatureDetector_create(**fast_params)
            keypoints = fast.detect(frame_gray, mask=mask)
            p = np.array([kp.pt for kp in keypoints], dtype=np.float32).reshape(-1, 1, 2) if keypoints else None

        elif detector_type.upper() == "ORB":
            orb = cv2.ORB_create(**orb_params)
            keypoints = orb.detect(frame_gray, mask=mask)
            p = np.array([kp.pt for kp in keypoints], dtype=np.float32).reshape(-1, 1, 2) if keypoints else None

        else:
            raise ValueError(f"未知 detector_type: {detector_type}")

        if p is not None:
            min_dist = 15.0
            existing_points = np.array([traj[-1] for traj in trajectories], dtype=np.float32) if trajectories else np.empty((0, 2))

            filtered_points = []
            for (x, y) in np.float32(p).reshape(-1, 2):
                if existing_points.shape[0] > 0:
                    dists = np.linalg.norm(existing_points - np.array([x, y]), axis=1)
                    if np.min(dists) < min_dist:
                        continue
                filtered_points.append((x, y))

            for x, y in filtered_points:
                trajectories.append([(x, y)])

    # ============ 统计信息 ============
    # 只有在存在轨迹且有重投影误差时才统计
    if len(trajectories) == 0 or len(reproj_errors) == 0:
        print(f"Frame {i}: no valid trajectories, skipped.")
        prev_gray = frame_gray
        frame_idx += 1
        continue

    # 过滤掉长度 < min_traj_len 的轨迹
    if "cone" in files[0] and data > 300:

        # min_traj_len = 10
        # valid_trajectories = [traj for traj in trajectories if len(traj) >= min_traj_len]

        # 如果有效轨迹数量太少，直接跳过
        min_traj_count = 30
    else:
        min_traj_count = 1

    if len(trajectories) < min_traj_count:
        print(f"Frame {i}: valid trajectories < {min_traj_count}, skipped.")
        prev_gray = frame_gray
        frame_idx += 1
        continue

    # 计算平均重投影误差和轨迹长度
    avg_reproj_error = np.mean(reproj_errors)
    traj_lengths = [len(traj) for traj in trajectories]
    avg_traj_len = np.mean(traj_lengths)

    print(f"Frame {i}: avg reprojection error = {avg_reproj_error:.2f}, avg trajectory length = {avg_traj_len:.2f}")

    # 写入txt文件
    # with open(result_filename, "a") as f:
    #     f.write(f"{i},{avg_reproj_error:.4f},{avg_traj_len:.2f}\n")



    frame_idx += 1
    prev_gray = frame_gray

    end = time.time()
    fps = 1 / (end-start)

    # cv2.imwrite(f"of_out/{i}.png", img_viz)
    cv2.imshow('Optical Flow', img_viz)
    video_writer.write(img_viz)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_writer.release()

