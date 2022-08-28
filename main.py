from io import StringIO
from pathlib import Path
import streamlit as st
import time
#from detect import detect
import os
import sys
import argparse
from PIL import Image
from demo import detector,detect
import torch

import imutils
import argparse
import time
from pathlib import Path

import cv2
import torch
from utils.general import increment_path
from utils.BaseDetector import *

from AIDetector_pytorch import Detector

from torch import from_numpy, jit


from streamlit_webrtc import webrtc_streamer
import av
import cv2


def get_subdirs(b='.'):
    '''
        Returns all sub-directories in a specific Path
    '''
    result = []
    for d in os.listdir(b):
        bd = os.path.join(b, d)
        if os.path.isdir(bd):
            result.append(bd)
    return result


def get_detection_folder():
    '''
        Returns the latest folder in a runs\detect
    '''
    return max(get_subdirs(os.path.join('runs', 'detect')), key=os.path.getmtime)


class det:
    def __init__(self):
        self.det = Detector()  # 一个yolo
        self.flag = 0
        self.net = jit.load(r'.\action_detect\checkPoint\openpose.jit')
        self.action_net = jit.load(r'.\action_detect\checkPoint\action5.jit')


class VideoProcessor:

    def recv(self, frame):#输入是来自网络摄像头的图像帧
        img = frame.to_ndarray(format="bgr24")
        print("frame",frame)
        print("img",img)
        img = detector(img) #Detector().feedCap(img)#cv2.cvtColor(, cv2.COLOR_GRAY2BGR)
        self.flag = self.flag+1

        return av.VideoFrame.from_ndarray(img, format="bgr24")#输出将显示在屏幕上





if __name__ == '__main__':

    st.title('课堂行为识别')

    parser = argparse.ArgumentParser()
    # 选用训练的权重，可用根目录下的yolov5s.pt，也可用runs/train/exp/weights/best.pt
    parser.add_argument('--weights', type=str, default='models/yolov5s.pt', help='model.pt path(s)')
    # 检测数据，可以是图片/视频路径，也可以是'0'(电脑自带摄像头),也可以是rtsp等视频流
    parser.add_argument('--source', type=str, default='D:\XXXXX\smart_classroom\data/vedio/t.mp4',
                        help='source')  # file/folder, 0 for webcam
    # 网络输入图片大小
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    # 置信度阈值，检测到的对象属于特定类（狗，猫，香蕉，汽车等）的概率
    parser.add_argument('--conf-thres', type=float, default=0.45, help='object confidence threshold')
    # 做nms的iou阈值
    parser.add_argument('--iou-thres', type=float, default=0.4, help='IOU threshold for NMS')
    # 检测的设备，cpu；0(表示一个gpu设备cuda:0)；0,1,2,3(多个gpu设备)。值为空时，训练时默认使用计算机自带的显卡或CPU
    parser.add_argument('--device', default='gpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    # 是否展示检测之后的图片/视频，默认False
    parser.add_argument('--view-img', action='store_true', help='display results')
    # 是否将检测的框坐标以txt文件形式保存，默认False
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    # 是否将检测的labels以txt文件形式保存，默认False
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    # 设置只保留某一部分类别，如0或者0 2 3
    parser.add_argument('--classes', nargs='+', default=0, type=int,
                        help='filter by class: --class 0, or --class 0 2 3')
    # 进行nms是否也去除不同类别之间的框，默认False
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    # 推理的时候进行多尺度，翻转等操作(TTA)推理
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    # 如果为True，则对所有模型进行strip_optimizer操作，去除pt文件中的优化器等信息，默认为False
    parser.add_argument('--update', action='store_true', help='update all models')
    # 检测结果所存放的路径，默认为runs/detect
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    # 检测结果所在文件夹的名称，默认为exp
    parser.add_argument('--name', default='exp', help='save results to project/name')
    # 若现有的project/name存在，则不进行递增
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    opt = parser.parse_args()
    print(opt)


    source = ("图片检测", "视频检测")
    source_index = st.sidebar.selectbox("选择输入", range(
        len(source)), format_func=lambda x: source[x])

    if source_index == 0:
        uploaded_file = st.sidebar.file_uploader(
            "上传图片", type=['png', 'jpeg', 'jpg'])
        if uploaded_file is not None:
            is_valid = True
            with st.spinner(text='资源加载中...'):
                st.sidebar.image(uploaded_file)
                picture = Image.open(uploaded_file)
                picture = picture.save(f'data/images/{uploaded_file.name}')
                opt.source = f'data/images/{uploaded_file.name}'
            flag = False
        else:
            is_valid = False
    else:
        uploaded_file = st.sidebar.file_uploader("上传视频", type=['mp4'])
        if uploaded_file is not None:
            is_valid = True
            with st.spinner(text='资源加载中...'):
                st.sidebar.video(uploaded_file)
                with open(os.path.join("data", "videos", uploaded_file.name), "wb") as f:
                    f.write(uploaded_file.getbuffer())
                opt.source = f'data/videos/{uploaded_file.name}'
            flag = True
        else:
            is_valid = False

    if True:#is_valid:
        # print('valid')
        # if st.button('开始检测'):

            with torch.no_grad():
                #while True:
                webrtc_streamer(key="example", video_processor_factory=VideoProcessor)
                #detect(opt, flag)
                st.balloons()

            # if source_index == 0:
            #     with st.spinner(text='Preparing Images'):
            #         for img in os.listdir(get_detection_folder()):
            #             st.image(str(Path(f'{get_detection_folder()}') / img))
            #         st.balloons()
            # else:
            #     with st.spinner(text='Preparing Video'):
            #         for vid in os.listdir(get_detection_folder()):
            #             _, follow = os.path.splitext(str(vid))
            #             if follow=='.mp4':
            #                 video_file = open(str(Path(f'{get_detection_folder()}') / vid) ,'rb')
            #                 video_bytes = video_file.read()
            #                 print(video_file)
            #                 st.video(video_bytes)
            #                 # print(str(Path(f'{get_detection_folder()}') / vid))
            #                 # st.video(str(Path(f'{get_detection_folder()}') / vid))
            #                 st.balloons()
