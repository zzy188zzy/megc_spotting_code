import os
import dlib  # 人脸识别的库 Dlib
import numpy as np  # 数据处理的库 numpy
import cv2  # 图像处理的库 OpenCv
from tools.filter import temporal_ideal_filter
from tools.emd import do_emd
from tqdm import tqdm
import pandas as pd
import math


font = cv2.FONT_HERSHEY_SIMPLEX
landmark0 = []

def crop_picture(img_rd, size, vid_name, img_name, total_landmark):
    # global landmarks
    # print(vid_name, img_name)

    # img_gray = cv2.cvtColor(img_rd, cv2.COLOR_RGB2GRAY)
    # # 人脸数
    # faces = detector(img_gray, 0)
    # # print(faces)
    # # 标68个点
    # for i in range(len(faces)):
    #     # 取特征点坐标
    #     landmarks = np.matrix([[p.x, p.y] for p in predictor(img_rd, faces[i]).parts()])
    # 两个眼角的位置
    # print(landmarks.shape)
    # landmark_path = f"/data2/zyzhang/dataset/CASME_cube/openface_crop_align/s{vid_name.split('_')[0]}/{vid_name}/{vid_name}.npy"
    # total_landmark = np.load(landmark_path)
    # img_idx = int(img_name.split("_")[-1].replace(".bmp", "")) - 1
    # landmarks = total_landmark[img_idx]
    # print(landmarks.shape)
    # exit()
    # if len(faces) == 0:
    #     img_idx = int(img_name.split("_")[-1].replace(".bmp", "")) - 1
    #     landmark_path = f"/data2/zyzhang/dataset/CASME_cube/openface_crop_align/s{vid_name.split('_')[0]}/{vid_name}/{vid_name}.npy"
    #     total_landmark = np.load(landmark_path)
    #     temp_land = []
    #     for k in range(68):
    #         temp_land.append([total_landmark[img_idx, k, 0], total_landmark[img_idx, k, 1]])
    #     landmarks = np.matrix(temp_land)
    # print(vid_name, img_name, "not found. Load pre landmark")

    img_idx = int(img_name.split("_")[-1].replace(".bmp", "")) - 1
    temp_land = []
    for k in range(68):
        temp_land.append([total_landmark[img_idx, k, 0], total_landmark[img_idx, k, 1]])
    landmarks = np.matrix(temp_land)
    # img_temp = img_rd.copy()
    # for idx in range(68):
    #     # print(landmarks[idx][0, 0], landmarks[idx][0, 1])
    #     cv2.circle(img_temp, (landmarks[idx][0, 0], landmarks[idx][0, 1]), 3, (0, 255, 0), -1)
    # cv2.imwrite("temp_img.jpg", img_temp)

    left = landmarks[39]
    right = landmarks[42]
    width_eye = int((right[0, 0] - left[0, 0]) / 2)
    # print(right.shape)
    # print(right[0, 0])

    center = [int((right[0, 0] + left[0, 0]) / 2), int((right[0, 1] + left[0, 1]) / 2)]
    cv2.rectangle(img_rd, (center[0] - int(4.5 * width_eye), center[1] - int(3.5 * width_eye)),
                  (center[0] + int(4.5 * width_eye), center[1] + int(5.5 * width_eye)),
                  (0, 0, 255), 2)
    b = center[1] + int(5 * width_eye)
    d = center[0] + int(4 * width_eye)
    a = max((center[1] - int(3 * width_eye)), 0)
    c = max(center[0] - int(4 * width_eye), 0)
    img_crop = img_rd[a:b, c:d]
    img_crop_samesize = cv2.resize(img_crop, (size, size))
    return landmarks, img_crop_samesize, a, b, c, d


def get_roi_bound(low, high, round, landmark0):
    roi1_points = landmark0[low:high]
    roi1_high = roi1_points[:, 0].argmax(axis=0)
    roi1_low = roi1_points[:, 0].argmin(axis=0)
    roi1_left = roi1_points[:, 1].argmin(axis=0)
    roil_right = roi1_points[:, 1].argmax(axis=0)
    roil_h = roi1_points[roi1_high, 0]
    roi1_lo = roi1_points[roi1_low, 0]
    roi1_le = roi1_points[roi1_left, 1]
    roil_r = roi1_points[roil_right, 1]
    roil_h_ex = (roil_h + round)[0, 0]
    roi1_lo_ex = (roi1_lo - round)[0, 0]
    roi1_le_ex = (roi1_le - round)[0, 0]
    roil_r_ex = (roil_r + round)[0, 0]
    return (roil_h_ex), (roi1_lo_ex), (roi1_le_ex), (roil_r_ex)


def get_roi(flow, percent):
    r1, theta1 = cv2.cartToPolar(flow[:, :, 0], flow[:, :, 1], angleInDegrees=True)
    r1 = np.ravel(r1)

    x1 = np.ravel(flow[:, :, 0])
    y1 = np.ravel(flow[:, :, 1])

    arg = np.argsort(r1)  # 代表了r1这个矩阵内元素的从小到大顺序
    num = int(len(r1) * (1 - percent))
    x_new = 0
    y_new = 0

    for i in range(num, len(arg)):  # 想取相对比较大的
        a = arg[i]
        x_new += x1[a]
        y_new += y1[a]
    x = x_new / (len(arg) - num)
    y = y_new / (len(arg) - num)

    return x, y


# 返回图像的68个标定点
def tu_landmarks(gray, img_rd, landmark0, frame_shang, frame_left, w, h, img_size, fa_model):
    # faces = detector(gray, 0)
    # if (len(faces) == 0):
    #     landmark0[:, 0] = (landmark0[:, 0] - frame_left) * (img_size / w)
    #     landmark0[:, 1] = (landmark0[:, 1] - frame_shang) * (img_size / h)
    #     landmarkss = landmark0
    # else:
    #     landmarkss = np.matrix([[p.x, p.y] for p in predictor(img_rd, faces[0]).parts()])

    landmarkss = np.array(fa_model.get_landmarks(img_rd)).reshape(68, 2)
    # print(landmarkss.shape)
    temp_land = []
    for idx in range(68):
        temp_land.append([landmarkss[idx, 0], landmarkss[idx, 1]])
    landmarkss = np.matrix(temp_land, dtype=np.int64)
    return landmarkss


# 对给定的每个视频帧之间的光流。进行求平方和和开根号的计算，并画出动作线
def draw_line(flow_total):
    flow_total = np.array(flow_total)

    flow_total = np.sum(flow_total ** 2, axis=1)
    flow_total = np.sqrt(flow_total)

    return flow_total


def fenxi(flow_total, imf_sum1, yuzhi1, yuzhi2):  # 使用寻找峰的方法
    flow_total = np.array(flow_total)
    low = np.min(flow_total)  # 找到最小值
    flow_total = flow_total - low  # 从零开始
    flow_total_fenxi = []
    for j in range(len(flow_total)):  # 找到大于较小阈值
        if flow_total[j] >= yuzhi1:
            flow_total_fenxi.append(j)  # 大于较小阈值的帧的索引
    flow_total_pp = []
    if len(flow_total_fenxi) > 0:  # 对经过第一步筛选的，帧相邻的连在一起
        start = flow_total_fenxi[0]
        end = flow_total_fenxi[0]
        st = 0
        for i in range(len(flow_total_fenxi)):
            if flow_total_fenxi[i] >= end and flow_total_fenxi[i] - end < 3:
                end = flow_total_fenxi[i]
            else:
                flow_total_pp.append([start, end])
                start = flow_total_fenxi[i]
                end = flow_total_fenxi[i]
        flow_total_pp.append([start, end])
    flow_total_fenxi = []
    flow_total_pp = np.array(flow_total_pp)

    for i in range(len(flow_total_pp)):  # 第二次筛选
        start = flow_total_pp[i, 0]
        end = flow_total_pp[i, 1]

        for j in range(start, end):
            a = max(0, j - 30)
            b = min(len(flow_total) - 1, j + 30)  # 找到这个点的两边，左边右边各30，注意不能超过滑动窗口的碧娜姐
            low = np.min(flow_total[a:b])  # 左右区间都找最小的
            low1 = np.min(imf_sum1[a:b])  # 左右区间都找最小的
            if flow_total[j] - low > yuzhi2:
                flow_total_fenxi.append(j)
    flow_total_pp2 = []
    if len(flow_total_fenxi) > 0:
        start = flow_total_fenxi[0]
        end = flow_total_fenxi[0]
        st = 0
        for i in range(len(flow_total_fenxi)):
            if flow_total_fenxi[i] >= end and flow_total_fenxi[i] - end < 3:
                end = flow_total_fenxi[i]
            else:
                flow_total_pp2.append([start, end])
                start = flow_total_fenxi[i]
                end = flow_total_fenxi[i]

        flow_total_pp2.append([start, end])

    return np.array(flow_total_pp2)


def expend(flow1_total_fenxi, flow1_total_edm):
    for i in range(len(flow1_total_fenxi)):
        start = flow1_total_fenxi[i, 0]
        end = flow1_total_fenxi[i, 1]
        a1 = max(0, start - 30)
        b1 = min(len(flow1_total_edm) - 1, start + 30)
        a2 = max(0, end - 30)
        b2 = min(len(flow1_total_edm) - 1, end + 30)
        if end > start:  # 因为有可能end=start
            high = np.max(flow1_total_edm[start:end])
        else:
            high = flow1_total_edm[start]

        st_low = np.min(flow1_total_edm[a1:b1])
        st_arglow = np.argmin(flow1_total_edm[a1:b1]) + a1  # start的左右中最小的索引
        en_low = np.min(flow1_total_edm[a2:b2])  # end的左右中最小的索引
        en_arglow = np.argmin(flow1_total_edm[a2:b2]) + a2
        if st_arglow < start:
            for j in range(start - 1, -1, -1):
                if flow1_total_edm[j] - st_low < 0.33 * (high - st_low):
                    start = j
                    break
                if flow1_total_edm[j] > flow1_total_edm[j + 1]:
                    start = j + 2
                    break
        else:
            left = max(start - 10, 0)
            aa = np.argmin(flow1_total_edm[left:start + 1]) + left  # 代表了start左侧十个中值最小的索引
            if flow1_total_edm[start] - flow1_total_edm[aa] > 0.3:
                start = aa + 1
        if en_arglow > end:
            for j in range(end + 1, en_arglow):
                if flow1_total_edm[j] - en_low < 0.33 * (high - en_low):
                    end = j
                    break
                if flow1_total_edm[j] > flow1_total_edm[j - 1]:
                    end = j - 2
                    break
        else:
            right = min(end + 10, len(flow1_total_edm) - 1)
            aa = np.argmin(flow1_total_edm[end:right + 1]) + end  # 代表了end右侧十个中值最小的索引
            if flow1_total_edm[end] - flow1_total_edm[aa] > 0.3:
                end = aa - 1  # 用最小值的索引进行替换

        flow1_total_fenxi[i, 0] = start
        flow1_total_fenxi[i, 1] = end
    return flow1_total_fenxi


def process(flow1_total, yuzhi1, yuzhi2, position, xuhao, k, a, totalflow):
    fs = 1
    c = 0.2
    # c2=0.1
    yuzhi1 = yuzhi1 + c
    yuzhi2 = yuzhi2 + c
    flow1_total = draw_line(flow1_total)  # 作用是将光流特征转换为幅值的形式
    flow1_total = np.array(flow1_total)

    position = position + str(xuhao) + "----"  #

    threshold_filt = 2
    flow1_total_edm1 = temporal_ideal_filter(flow1_total[a:-a], 1, threshold_filt, 30)  # 滤波
    hh = len(flow1_total_edm1) + 2

    flow1_total_edm2, imf_sum1 = do_emd(flow1_total[a:-a], flow1_total_edm1, position, str(k - hh), fs)

    flow1_total_fenxi = fenxi(flow1_total_edm1, imf_sum1, yuzhi1, yuzhi2)  # 得到了分析结果
    flow1_total_fenxi = expend(flow1_total_fenxi, flow1_total_edm1)  # 向两边扩展

    flow1_total_fenxi = flow1_total_fenxi + (k - hh) + a
    for i in range(len(flow1_total_fenxi)):
        totalflow.append(flow1_total_fenxi[i])

    return totalflow


def nms2(totalflow, threshold):
    totalflow = np.array(totalflow)
    hh = [[0, 0]]
    for i in range(len(totalflow)):
        new = 1
        if i == 0:
            hh = np.vstack((hh, [[totalflow[i, 0], totalflow[i, 1]]]))
            continue
        for j in range(1, len(hh)):
            if totalflow[i, 0] > hh[j, 1] or totalflow[i, 1] < hh[j, 0]:  # 两个间隔完全不相交
                iou = 0
            else:
                ma = max(totalflow[i, 0], hh[j, 0])
                mi = min(totalflow[i, 1], hh[j, 1])
                wid = mi - ma
                iou = max(wid / (hh[j, 1] - hh[j, 0]), wid / (totalflow[i, 1] - totalflow[i, 0]))
            # 通过iou决定是不是要添加
            if iou > threshold:  # SAMM0.34  CASME 0.29   #如果重复率比较高就
                new = 0
                hh[j, 1] = max(hh[j, 1], totalflow[i, 1])
                hh[j, 0] = min(hh[j, 0], totalflow[i, 0])
        if new == 1:
            hh = np.vstack((hh, [[totalflow[i, 0], totalflow[i, 1]]]))
    return hh