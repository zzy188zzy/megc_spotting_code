from cas_utils import *
import os
import numpy as np  # 数据处理的库 numpy
import cv2  # 图像处理的库 OpenCv
from tqdm import tqdm
import pandas as pd
import face_alignment
import json


def cal_iou(seg1, seg2):
    iou_value = (min(seg1[1], seg2[1]) - max(seg1[0], seg2[0])) / (max(seg1[1], seg2[1]) - min(seg1[0], seg2[0]))
    return iou_value


def draw_roiline_redetect(path1, path2, fs,
                          eye_param, eye_param_2,
                          eye_param_little, eye_param_little_2,
                          mouth_param, mouth_param_2,
                          mouth_param_little, mouth_param_little_2,
                          nose_param, nose_param_2,
                          nms_param, fa_model, st, ed, pre_dis
                          ):  # 与16相比再增加两个位置眼睑部位
    path = path1
    fileList1 = os.listdir(path)  # 图片路径
    fileList1.sort()
    fileList = []
    l = 0
    for i in fileList1:
        if (l % fs == 0):
            fileList.append(i)
        l = l + 1
    label_vio = np.array([[0, 0]])

    landmark_path = f"/data2/zyzhang/dataset/CASME_feature/CAS_crop_align/{path2}/{path2}.npy"
    total_landmark = np.load(landmark_path)
    st = max(st - pre_dis, 0)
    ed = min(st + 200, len(fileList))
    fileList = fileList[st:ed]
    start = 1
    end = len(fileList)
    k = 0
    for i in fileList:
        # for i in fileList:
        k = k + 1
        if (k >= start):
            if (k == start):
                flow1_total = [[0, 0]]  # 是存储了不同位置帧之间的光流
                flow1_total1 = [[0, 0]]
                flow1_total2 = [[0, 0]]
                flow1_total3 = [[0, 0]]
                flow2_total = [[0, 0]]
                flow3_total = [[0, 0]]
                flow3_total1 = [[0, 0]]
                flow3_total2 = [[0, 0]]
                flow3_total3 = [[0, 0]]
                flow4_total = [[0, 0]]
                flow4_total1 = [[0, 0]]
                flow4_total2 = [[0, 0]]
                flow4_total3 = [[0, 0]]
                flow4_total4 = [[0, 0]]
                flow4_total5 = [[0, 0]]
                flow5_total1 = [[0, 0]]
                flow5_total2 = [[0, 0]]
                flow2_total1 = [[0, 0]]
                flow6_total = [[0, 0]]
                flow7_total = [[0, 0]]

                img_rd = cv2.imread(path + "/" + i)  # D:/face_image_test/EP07_04/

                img_size = 256
                landmark0, img_rd, frame_shang, frame_xia, frame_left, frame_right = crop_picture(img_rd, img_size,
                                                                                                  path2, i,
                                                                                                  total_landmark)
                # 记录框的位置，上下左右在整个图片中的坐标，和68点的位置。img_rd是被裁减之后的面部位置，并resize到256*256

                gray = cv2.cvtColor(img_rd, cv2.COLOR_RGB2GRAY)  # 变成灰度图
                landmark0 = tu_landmarks(gray, img_rd, landmark0, frame_shang, frame_left, frame_xia - frame_shang,
                                         frame_right - frame_left, img_size, fa_model)  # 对人脸68个点的定位
                # 相对与新图片的68点的位置。
                # img_temp = img_rd.copy()
                # for idx in range(68):
                #     # print(idx)
                #     # print(landmark0[idx][0, 0], landmark0[idx][0, 1])
                #     cv2.circle(img_temp, (landmark0[idx][0, 0], landmark0[idx][0, 1]), 3, (0, 255, 0), -1)
                # cv2.imwrite("temp_img3.jpg", img_temp)
                # exit()

                round1 = 0
                roil_right, roi1_left, roi1_low, roi1_high = get_roi_bound(17, 22, 0, landmark0)  # 左眉毛的位置

                roi1_sma = []  # 存储了左眼的三个小的感兴趣区域，从里到外
                roi1_sma.append([landmark0[20, 1] - (roi1_low - 15), landmark0[20, 0] - (roi1_left - 5)])
                roi1_sma.append([landmark0[19, 1] - (roi1_low - 15), landmark0[19, 0] - (roi1_left - 5)])
                roi1_sma.append([landmark0[18, 1] - (roi1_low - 15), landmark0[18, 0] - (roi1_left - 5)])

                eyes_mid_round = int((landmark0[22, 0] - landmark0[21, 0]) * 2 / 5)
                eyes_mid_dis = landmark0[22, 0] - landmark0[21, 0] - eyes_mid_round * 2
                prevgray_roi1 = gray[max(0, roi1_low - 5): roi1_high + 15, roi1_left - 5: roil_right + eyes_mid_round]
                # top_left = (roi1_left - 5, roi1_low - 5)
                # bottom_right = (roil_right + eyes_mid_round, roi1_high + 15)
                # cv2.rectangle(img_rd, top_left, bottom_right, (0, 255, 0), 2)

                # 右眼以左眼为基准
                roi3_right, roi3_left, roi3_low, roi3_high = get_roi_bound(22, 27, 0, landmark0)
                # top_left = (roil_right + eyes_mid_round + eyes_mid_dis, roi3_low - 5)
                # bottom_right = (roi3_right + 5, roi3_high + 15)
                # cv2.rectangle(img_rd, top_left, bottom_right, (0, 255, 0), 2)
                roi3_sma = []  # 存储了右眼的三个小的感兴趣区域，从里到外
                roi3_sma.append([landmark0[23, 1] - (roi3_low - 15), landmark0[23, 0] - roi3_left])
                roi3_sma.append([landmark0[24, 1] - (roi3_low - 15), landmark0[24, 0] - roi3_left])
                roi3_sma.append([landmark0[25, 1] - (roi3_low - 15), landmark0[25, 0] - roi3_left])
                prevgray_roi3 = gray[max(0, roi3_low - 5): roi3_high + 15,
                                roil_right + eyes_mid_round + eyes_mid_dis: roi3_right + 5]

                # top_left = (roi3_left - 10, max(0,roi3_low - 5))
                # bottom_right = (roi3_right + 5, roi3_high + 10)
                # cv2.rectangle(img_rd, top_left, bottom_right, (0, 255, 0), 2)

                # 嘴巴处的四个
                roi4_right, roi4_left, roi4_low, roi4_high = get_roi_bound(48, 67, 0, landmark0)
                roi4_sma = []
                roi4_sma.append([landmark0[48, 1] - (roi4_low - 15), landmark0[48, 0] - (roi4_left - 20)])
                roi4_sma.append([landmark0[54, 1] - (roi4_low - 15), landmark0[54, 0] - (roi4_left - 20)])
                roi4_sma.append([landmark0[51, 1] - (roi4_low - 15), landmark0[51, 0] - (roi4_left - 20)])
                roi4_sma.append([landmark0[57, 1] - (roi4_low - 15), landmark0[57, 0] - (roi4_left - 20)])
                roi4_sma.append([landmark0[62, 1] - (roi4_low - 15), landmark0[62, 0] - (roi4_left - 20)])

                prevgray_roi4 = gray[(roi4_low - 15):roi4_high + 10, roi4_left - 20:roi4_right + 20]
                # top_left = (roi4_left - 20, roi4_low - 15)
                # bottom_right = (roi4_right + 20, roi4_high + 10)
                # cv2.rectangle(img_rd, top_left, bottom_right, (0, 255, 0), 2)
                # cv2.imwrite(f"imgs/{path2}.png", img_rd)
                # return

                # 鼻子两侧
                roi5_right, roi5_left, roi5_low, roi5_high = get_roi_bound(30, 36, 0, landmark0)
                roi5_sma = []
                roi5_sma.append([landmark0[31, 1] - (roi5_low - 20), landmark0[31, 0] - (roi5_left - 30)])
                roi5_sma.append([landmark0[35, 1] - (roi5_low - 20), landmark0[35, 0] - (roi5_left - 30)])

                prevgray_roi5 = gray[(roi5_low - 20):roi5_high + 5, roi5_left - 30:roi5_right + 30]

                roi2_right, roi2_left, roi2_low, roi2_high = get_roi_bound(29, 31, 13, landmark0)
                prevgray_roi2 = gray[roi2_low:roi2_high, roi2_left:roi2_right]

            else:
                if (True):
                    img_rd1 = cv2.imread(path + "/" + i)  # D:/face_image_test/EP07_04/
                    # print(path+i)
                    img_crop = img_rd1[frame_shang:frame_xia, frame_left:frame_right]  # 按照第一个图的框切割出一个脸

                    img_rd = cv2.resize(img_crop, (img_size, img_size))
                    gray = cv2.cvtColor(img_rd, cv2.COLOR_RGB2GRAY)
                    # 求全局的光流
                    gray_roi2 = gray[roi2_low:roi2_high, roi2_left:roi2_right]
                    # 使用Gunnar Farneback算法计算密集光流
                    flow2 = cv2.calcOpticalFlowFarneback(prevgray_roi2, gray_roi2, None, 0.5, 3, 15, 5, 7, 1.5, 0)
                    flow2 = np.array(flow2)

                    # him2, x1, y1 = get_roi_him(flow2[15:-10, 5:-5, :])
                    x1, y1 = get_roi(flow2[15:-10, 5:-5, :], 0.7)
                    # print("全局运动为{}and{}".format(x1,y1))
                    flow2_total1.append([x1, y1])
                    # 左眼
                    gray_roi1 = gray[max(0, roi1_low - 5): roi1_high + 15, roi1_left - 5: roil_right + eyes_mid_round]
                    # 使用Gunnar Farneback算法计算密集光流
                    flow1 = cv2.calcOpticalFlowFarneback(prevgray_roi1, gray_roi1, None, 0.5, 3, 15, 5, 7, 1.5,
                                                         0)  # 计算整个左眉毛处的光流

                    flow1[:, :, 0] = flow1[:, :, 0]
                    flow1[:, :, 1] = flow1[:, :, 1]
                    round1 = 10
                    roi1_sma = np.array(roi1_sma)
                    a, b = get_roi(flow1[round1:-round1, round1:-round1, :], 0.2)  # 去掉光流特征矩阵周边round大小的部分，求均值
                    a1, b1 = get_roi(  # 一个感兴趣区域处的平均光流
                        flow1[roi1_sma[0, 0] - 10:roi1_sma[0, 0] + 10, roi1_sma[0, 1] - 10:roi1_sma[0, 1] + 10, :],
                        0.2)
                    a2, b2 = get_roi(
                        flow1[roi1_sma[1, 0] - 10:roi1_sma[1, 0] + 10, roi1_sma[1, 1] - 10:roi1_sma[1, 1] + 10, :],
                        0.2)
                    a3, b3 = get_roi(
                        flow1[roi1_sma[2, 0] - 10:roi1_sma[2, 0] + 10, roi1_sma[2, 1] - 10:roi1_sma[2, 1] + 10, :],
                        0.2)
                    flow1_total1.append([a1 - x1, b1 - y1])  # 局部区域减去全局光流
                    flow1_total2.append([a2 - x1, b2 - y1])
                    flow1_total3.append([a3 - x1, b3 - y1])
                    flow1_total.append([a - x1, b - y1])

                    gray_roi3 = gray[max(0, roi3_low - 5): roi3_high + 15,
                                roil_right + eyes_mid_round + eyes_mid_dis: roi3_right + 5]
                    # 使用Gunnar Farneback算法计算密集光流
                    flow3 = cv2.calcOpticalFlowFarneback(prevgray_roi3, gray_roi3, None, 0.5, 3, 15, 5, 7, 1.5, 0)
                    flow3[:, :, 0] = flow3[:, :, 0]
                    flow3[:, :, 1] = flow3[:, :, 1]
                    round1 = 10

                    roi3_sma = np.array(roi3_sma)
                    a, b = get_roi(flow3[round1:-round1, round1:-round1, :], 0.3)
                    a1, b1 = get_roi(
                        flow3[roi3_sma[0, 0] - 10:roi3_sma[0, 0] + 10, roi3_sma[0, 1] - 10:roi3_sma[0, 1] + 10, :],
                        0.3)
                    a2, b2 = get_roi(
                        flow3[roi3_sma[1, 0] - 10:roi3_sma[1, 0] + 10, roi3_sma[1, 1] - 10:roi3_sma[1, 1] + 10, :],
                        0.3)
                    a3, b3 = get_roi(
                        flow3[roi3_sma[2, 0] - 10:roi3_sma[2, 0] + 10, roi3_sma[2, 1] - 10:roi3_sma[2, 1] + 10, :],
                        0.3)

                    flow3_total1.append([a1 - x1, b1 - y1])
                    flow3_total2.append([a2 - x1, b2 - y1])
                    flow3_total3.append([a3 - x1, b3 - y1])
                    flow3_total.append([a - x1, b - y1])

                    gray_roi4 = gray[(roi4_low - 15):roi4_high + 10, roi4_left - 20:roi4_right + 20]

                    flow4 = cv2.calcOpticalFlowFarneback(prevgray_roi4, gray_roi4, None, 0.5, 3, 15, 5, 7, 1.5, 0)
                    flow4[:, :, 0] = flow4[:, :, 0]
                    flow4[:, :, 1] = flow4[:, :, 1]
                    round1 = 10
                    roi4_sma = np.array(roi4_sma)
                    # print(roi1_sma)
                    a, b = get_roi(flow4[round1:-round1, round1:-round1, :], 0.3)
                    a1, b1 = get_roi(
                        flow4[roi4_sma[0, 0] - 10:roi4_sma[0, 0] + 10, roi4_sma[0, 1] - 10:roi4_sma[0, 1] + 20, :],
                        0.2)
                    a2, b2 = get_roi(
                        flow4[roi4_sma[1, 0] - 10:roi4_sma[1, 0] + 10, roi4_sma[1, 1] - 20:roi4_sma[1, 1] + 10, :],
                        0.2)
                    a3, b3 = get_roi(
                        flow4[roi4_sma[2, 0] - 10:roi4_sma[2, 0] + 10, roi4_sma[2, 1] - 10:roi4_sma[2, 1] + 10, :],
                        0.2)
                    a4, b4 = get_roi(
                        flow4[roi4_sma[3, 0] - 10:roi4_sma[3, 0] + 10, roi4_sma[3, 1] - 10:roi4_sma[3, 1] + 10, :],
                        0.2)
                    a5, b5 = get_roi(
                        flow4[roi4_sma[4, 0] - 10:roi4_sma[4, 0] + 10, roi4_sma[4, 1] - 10:roi4_sma[4, 1] + 10, :],
                        0.2)

                    flow4_total1.append([a1 - x1, b1 - y1])
                    flow4_total2.append([a2 - x1, b2 - y1])
                    flow4_total3.append([a3 - x1, b3 - y1])
                    flow4_total4.append([a4 - x1, b4 - y1])
                    flow4_total5.append([a5 - x1, b5 - y1])
                    flow4_total.append([a - x1, b - y1])

                    gray_roi5 = gray[(roi5_low - 20):roi5_high + 5, roi5_left - 30:roi5_right + 30]
                    # # 使用Gunnar Farneback算法计算密集光流
                    flow5 = cv2.calcOpticalFlowFarneback(prevgray_roi5, gray_roi5, None, 0.5, 3, 15, 5, 7, 1.5, 0)

                    round1 = 10
                    roi5_sma = np.array(roi5_sma)

                    a1, b1 = get_roi(
                        flow5[roi5_sma[0, 0] - 20:roi5_sma[0, 0] + 5, roi5_sma[0, 1] - 20:roi5_sma[0, 1] + 10, :],
                        0.2)
                    a2, b2 = get_roi(
                        flow5[roi5_sma[1, 0] - 20:roi5_sma[1, 0] + 5, roi5_sma[1, 1] - 10:roi5_sma[1, 1] + 20, :],
                        0.2)

                    flow5_total1.append([a1 - x1, b1 - y1])
                    flow5_total2.append([a2 - x1, b2 - y1])
                    round1 = 5

        if (k == end):
            hh = end - start + 1

            totalflow = []
            totalflowmic = []
            totalflowmac = []
            a = 1
            totalflow = process(flow1_total, eye_param, eye_param_2, "left_eye", 0, k, a, totalflow)
            totalflow = process(flow1_total1, eye_param_little, eye_param_little_2, "left_eye", 1, k, a, totalflow)
            totalflow = process(flow1_total2, eye_param_little, eye_param_little_2, "left_eye", 2, k, a, totalflow)
            totalflow = process(flow1_total3, eye_param_little, eye_param_little_2, "left_eye", 3, k, a, totalflow)

            totalflow = process(flow3_total, eye_param, eye_param_2, "right_eye", 0, k, a, totalflow)
            totalflow = process(flow3_total1, eye_param_little, eye_param_little_2, "right_eye", 1, k, a, totalflow)
            totalflow = process(flow3_total2, eye_param_little, eye_param_little_2, "right_eye", 2, k, a, totalflow)
            totalflow = process(flow3_total3, eye_param_little, eye_param_little_2, "right_eye", 3, k, a, totalflow)

            totalflow = process(flow4_total, mouth_param, mouth_param_2, "mouth", 0, k, a, totalflow)
            totalflow = process(flow4_total1, mouth_param_little, mouth_param_little_2, "mouth", 1, k, a, totalflow)
            totalflow = process(flow4_total2, mouth_param_little, mouth_param_little_2, "mouth", 2, k, a, totalflow)
            totalflow = process(flow4_total3, mouth_param_little, mouth_param_little_2, "mouth", 3, k, a, totalflow)
            totalflow = process(flow4_total4, mouth_param_little, mouth_param_little_2, "mouth", 4, k, a, totalflow)
            totalflow = process(flow4_total5, mouth_param_little, mouth_param_little_2, "mouth", 5, k, a, totalflow)

            totalflow = process(flow5_total1, nose_param, nose_param_2, "nose", 1, k, a, totalflow)
            totalflow = process(flow5_total2, nose_param, nose_param_2, "nose", 2, k, a, totalflow)

            totalflow = np.array(nms2(totalflow, nms_param))  # 把所有通道融合起来
            totalflow = np.array(nms2(totalflow, nms_param))
            totalflow_1 = totalflow - (k - hh)
            move = 100
            for i in range(len(totalflow_1)):
                if (totalflow_1[i, 0] < 100 and totalflow_1[i, 1] > 100):
                    if (totalflow_1[i, 1] < 150):
                        move = totalflow_1[i, 1] + 20
                    elif (totalflow_1[i, 0] > 50):
                        move = totalflow_1[i, 0] - 20
                    else:
                        a = min(189, totalflow_1[i, 1])
                        move = a + 10

            label_vio = np.vstack((label_vio, totalflow))
            break
    # print("全部：")

    label_video_update = []  # 去除一些太短的片段
    label_video_update1 = []
    for i in range(len(label_vio)):
        if label_vio[i, 1] - label_vio[i, 0] >= 12 and label_vio[i, 1] - label_vio[i, 0] <= 200:
            label_video_update.append([label_vio[i, 0], label_vio[i, 1]])
    label_video_update.sort()
    label_video_update = np.array(nms2(label_video_update, nms_param))
    label_video_update = np.array(nms2(label_video_update, nms_param))
    for i in range(len(label_video_update)):
        if (label_video_update[i, 1] != 0):
            label_video_update1.append([label_video_update[i, 0], label_video_update[i, 1]])

    label_video_update1 = np.array(label_video_update1) + st
    return label_video_update1


def redetect(eye_param, eye_param_2,
             eye_param_little, eye_param_little_2,
             mouth_param, mouth_param_2,
             mouth_param_little, mouth_param_little_2,
             nose_param, nose_param_2, nms_param, iou_1, path):
    df = pd.read_csv(path)
    fa_model = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_HALF_D, device='cuda', face_detector='sfd',
                                            flip_input=False)
    drop_list = []
    new_df = df.copy()
    for idx in tqdm(range(len(df))):
        vid = df["vid"][idx]
        onset = df["onset"][idx]
        offset = df["offset"][idx]
        pre_dis = 10
        pp1 = draw_roiline_redetect(f"/data2/zyzhang/dataset/CASME_feature/CAS_crop_align/{vid}/{vid}_aligned", vid, 1,
                                    eye_param, eye_param_2,
                                    eye_param_little, eye_param_little_2,
                                    mouth_param, mouth_param_2,
                                    mouth_param_little, mouth_param_little_2,
                                    nose_param, nose_param_2, nms_param, fa_model, onset, offset, pre_dis)
        pp1 = pp1.tolist()
        pre_dis = 50
        pp2 = draw_roiline_redetect(f"/data2/zyzhang/dataset/CASME_feature/CAS_crop_align/{vid}/{vid}_aligned", vid, 1,
                                    eye_param, eye_param_2,
                                    eye_param_little, eye_param_little_2,
                                    mouth_param, mouth_param_2,
                                    mouth_param_little, mouth_param_little_2,
                                    nose_param, nose_param_2, nms_param, fa_model, onset, offset, pre_dis)
        pp2 = pp2.tolist() + pp1
        pre_dis = 100
        pp3 = draw_roiline_redetect(f"/data2/zyzhang/dataset/CASME_feature/CAS_crop_align/{vid}/{vid}_aligned", vid, 1,
                                    eye_param, eye_param_2,
                                    eye_param_little, eye_param_little_2,
                                    mouth_param, mouth_param_2,
                                    mouth_param_little, mouth_param_little_2,
                                    nose_param, nose_param_2, nms_param, fa_model, onset, offset, pre_dis)
        pp3 = pp3.tolist() + pp2

        if len(pp3) == 0:
            print(vid, onset, offset)
            drop_list.append(idx)
            continue
        pp3.sort()
        flag = False
        seg_list = []
        temp_list = []
        for seg in pp3:
            temp_list.append([int(seg[0]), int(seg[1])])
        new_dic = {
            "name": vid,
            "org_segment": [int(onset), int(offset)],
            "segments": temp_list
        }
        with open('pp.json', 'a') as f:
            json.dump(new_dic, f, indent=4)
        for first_seg in pp3:
            iou = round(cal_iou(first_seg, [onset, offset]), 2)
            if iou >= iou_1:
                # Try different iou_1 and select the one that performs best on the test set
                print(first_seg, onset, offset)
                flag = True
                seg_list.append(first_seg)

        # if flag:
        # print("=========================")
        # print(seg_list)
        # cnt = len(seg_list) + 1
        # sum_x = onset
        # sum_y = offset
        # for seg in seg_list:
        #     sum_x += seg[0]
        #     sum_y += seg[1]
        # sum_x /= cnt
        # sum_y /= cnt
        # new_df.at[idx, 'onset'] = int(sum_x)
        # new_df.at[idx, 'offset'] = int(sum_y)
        # print(onset, offset)
        # print("True", vid, onset, offset)

        if not flag:
            # print("=========================")
            # print(pp)
            # print("False", vid, onset, offset)
            drop_list.append(idx)
    print(drop_list)
    pre_cas_df = new_df.drop(index=drop_list).reset_index(drop=True)
    pre_cas_df.to_csv("csv_files/cas_pred_drop_test.csv", index=False)
