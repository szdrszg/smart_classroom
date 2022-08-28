
import numpy as np
from torch import from_numpy, argmax

DEVICE = "gpu"
d={0:'sit', 1:'answer', 2:'raise_hand', 3:'write', 4:'sleep'}

def action_detect(net,pose,crown_proportion):
    # img = cv2.cvtColor(pose.img_pose,cv2.IMREAD_GRAYSCALE)
    action_num=5

    maxHeight = pose.keypoints.max()
    minHeight = pose.keypoints.min()

    img = pose.img_pose.reshape(-1)
    img = img / 255  # 把数据转成[0,1]之间的数据

    img = np.float32(img)

    img = from_numpy(img[None,:]).cpu()

    predect = net(img)#tensor([[9.8220e-01, 1.6849e-02, 9.5495e-04]] 三个possible
    print('predect')
    print(predect)

    action_id = int(argmax(predect,dim=1).cpu().detach().item())#取出tensor中最大值所对应的索引，此时最大值为9.8220e-01，其对应的位置索引值为0
    print('action_id')
    print(action_id)#0

    possible_rate = predect[:,action_id]# + 0.4*(crown_proportion-1) #9.8220e-01,
    print('possibel_rate')
    print(possible_rate)

    possible_rate = possible_rate.detach().numpy()[0]

    if possible_rate > 0.55:
    # if maxHeight-minHeight < 50:
        pose.pose_action = d.get(action_id)
        pose.action_possible = possible_rate
    #     pose.pose_action = 'sit'
    #     if possible_rate > 1:
    #         possible_rate = 1
    #     pose.action_fall = possible_rate
    #     pose.action_normal = 1-possible_rate
    # else:
    #     pose.pose_action = 'normal'
    #     if possible_rate >= 0.5:
    #         pose.action_fall = 1-possible_rate
    #         pose.action_normal = possible_rate
    #     else:
    #         pose.action_fall = possible_rate
    #         pose.action_normal = 1 - possible_rate

    return pose