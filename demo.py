import imutils
import argparse
import numpy
import time
from pathlib import Path
import os

import cv2
import torch
from utils.general import increment_path
from utils.BaseDetector import *

from AIDetector_pytorch import Detector

from torch import from_numpy, jit
import runOpenpose


import runOpenpose


def detector(im,flag = 0):
    #print(im.shape)
    print("im",im)
    im= numpy.array(im)
    det=Detector()
    net = jit.load(r'.\action_detect\checkPoint\openpose.jit')
    action_net = jit.load(r'.\action_detect\checkPoint\action5.jit')
    result = det.feedCap(im)
    result = result['frame']  # bgr格式
    #print(result.shape)
    print("result",result)
    result = imutils.resize(result, height=768)
    result = runOpenpose.run_demo(net, action_net, result, 256, False, [], True)
    #cv2.imshow('demo',result)
    #cv2.waitKey(1000)
    return result

def detect(opt,video_flag,save_img=False):
    global ip
    source, weights, view_img, save_txt, imgsz = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://'))
    # Directories

    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
    path=source

    name = 'demo'

    det = Detector()#一个yolo
    cap = cv2.VideoCapture(source)#从文件or摄像机读视频 https://blog.csdn.net/u010368556/article/details/79186992
    fps = int(cap.get(5))#获取参数 5：帧速率
    print('fps:', fps)
    t = int(1000/fps)
    videoWriter = None
    vid_path, vid_writer = None, None
    flag=0
    print("加载摔倒检测的模型开始")
    net = jit.load(r'.\action_detect\checkPoint\openpose.jit')
    action_net = jit.load(r'.\action_detect\checkPoint\action5.jit')
    print("加载摔倒检测的模型结束")
    print("检测中...")

    while True:

        # try:
        _, im = cap.read()#将视频帧读取到cv::Mat矩阵中
        if im is None:
            break


        result = det.feedCap(im)
        result = result['frame']#bgr格式
        result = imutils.resize(result, height=768)
        print(result.__len__())
        print(result.shape)#(256, 455, 3)
        from_numpy(result)
        if True:#flag % 6 <= 2:
            result = runOpenpose.run_demo(net, action_net, result, 256, False, [], True)
        flag = flag +1

        if videoWriter is None:
            fourcc = cv2.VideoWriter_fourcc(
                'm', 'p', '4', 'v')  # opencv3.0
            # videoWriter = cv2.VideoWriter(
            # 'result.mp4', fourcc, fps, (result.shape[1], result.shape[0]))


        #cv2.imshow(name, result)

        if webcam:  # batch_size >= 1
            p = path[t]
        else:
            p = path
        p = Path(p)
        sourse_name, _ = os.path.splitext(p.name)#img
        filename = str(sourse_name)+str(time.strftime('%Y%m%d%H%M%S', time.localtime(time.time())))+".jpg"
        save_path = str(save_dir / filename)  # img.jpg
        imageName = str(time.strftime('%Y%m%d%H%M%S', time.localtime(time.time()))) + ".jpg"
        cv2.imwrite(save_path , result)

    if video_flag:
            vid_name = str(sourse_name)+str(time.strftime('%Y%m%d%H%M%S', time.localtime(time.time())))+".mp4"
            save_path=str(save_dir/vid_name)
            if isinstance(vid_writer, cv2.VideoWriter):
                vid_writer.release()  # release previous video writer
            fourcc = 'mp4v'  # output video codec
            fps = cap.get(cv2.CAP_PROP_FPS)
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            # vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))

            list = []
            print(save_dir)
            for root, dirs, files in os.walk(save_dir):
                for file in files:
                    list.append(file)  # 获取目录下文件名列表
                    print(file)

            # VideoWriter是cv2库提供的视频保存方法，将合成的视频保存到该路径中
            # 'MJPG'意思是支持jpg格式图片
            # fps = 5代表视频的帧频为5，如果图片不多，帧频最好设置的小一点
            # (1280,720)是生成的视频像素1280*720，一般要与所使用的图片像素大小一致，否则生成的视频无法播放
            # 定义保存视频目录名称和压缩格式，像素为1280*720
            #print(save_path)
            video = cv2.VideoWriter(save_path, 0x00000021, fps,
                                    (1280,720),True)

            print("视频合成中...")
            for i in range(1, len(list)):
                # 读取图片
                img = cv2.imread(str(save_dir/list[i - 1]))
                # resize方法是cv2库提供的更改像素大小的方法
                # 将图片转换为1280*720像素大小
                img = cv2.resize(img, (1280,720))
                # 写入视频
                video.write(img)
                #print(i)
            print("视频合成完毕")


        # if video_flag==False:
        # cv2.imwrite(save_path, result)
        # else:  # 'video' or 'stream'
        #     if vid_path != save_path:  # new video
        #         vid_path = save_path
        #         if isinstance(vid_writer, cv2.VideoWriter):
        #             vid_writer.release()  # release previous video writer
        #         if cap:  # video
        #             fps = cap.get(cv2.CAP_PROP_FPS)
        #             w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        #             h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        #         else:  # stream
        #             fps, w, h = 30, result.shape[1], result.shape[0]
        #             save_path += '.mp4'
        #         vid_writer = cv2.VideoWriter(
        #             # save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        #             # 参考 https://xugaoxiang.com/2021/08/20/opencv-h264-videowrite
        #             save_path, cv2.VideoWriter_fourcc(*'avc1'), fps, (w, h))
        #
        #     vid_writer.write(result)
        #    cv2.waitKey(t)


if __name__ == '__main__':
    print(torch.cuda.is_available())
    parser = argparse.ArgumentParser()
    # 选用训练的权重，可用根目录下的yolov5s.pt，也可用runs/train/exp/weights/best.pt
    parser.add_argument('--weights', type=str, default='models/yolov5s.pt', help='model.pt path(s)')
    # 检测数据，可以是图片/视频路径，也可以是'0'(电脑自带摄像头),也可以是rtsp等视频流
    parser.add_argument('--source', type=str, default='D:\XXXXX\smart_classroom\data/video/7_2.mp4',
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

    with torch.no_grad():
        detect(opt,True)
        # cap = cv2.VideoCapture(opt.source)
        # _, im = cap.read()
        # detector(im)

