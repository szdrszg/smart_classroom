def mat_inter(box1, box2):
    # 判断两个矩形是否相交
    # box=(xA,yA,xB,yB)
    x01, y01, x02, y02 = box1
    x11, y11, x12, y12 = box2

    lx = abs((x01 + x02) / 2 - (x11 + x12) / 2)
    ly = abs((y01 + y02) / 2 - (y11 + y12) / 2)
    sax = abs(x01 - x02)
    sbx = abs(x11 - x12)
    say = abs(y01 - y02)
    sby = abs(y11 - y12)
    if lx <= (sax + sbx) / 2 and ly <= (say + sby) / 2:
        return True
    else:
        return False


def solve_coincide(box1, box2):
    # box=(xA,yA,xB,yB)
    # 计算两个矩形框的重合度
    #if mat_inter(box1, box2) == True:
    x01, y01, x02, y02 = box1
    x11, y11, x12, y12 = box2

    #print(abs(min(x01,x02)-min(x12,x11)),abs(max(x01,x02)-max(x01,x12)),abs(min(y02,y01)-min(y12,y11)),abs(max(y01,y02)-max(y11,y12)))
    #yolo框和openpose的框相距小于50或openpose框在yolo框里侧
    a1 = abs(min(x01,x02)-min(x12,x11)) <50 or min(x01,x02)-min(x12,x11)<0
    a2 = abs(max(x01,x02)-max(x01,x12)) <50 or max(x01,x02)-max(x01,x12)>0
    a3 = abs(min(y02,y01)-min(y12,y11)) <50 or min(y02,y01)-min(y12,y11)<0
    a4 = abs(max(y01,y02)-max(y11,y12)) <50 or max(y01,y02)-max(y11,y12)>0

    if a1 and a2 and a3 and a4:
        print("box1{}", box1)
        print("box2{}", box2)
        # col = min(x02, x12) - max(x01, x11)
        # row = min(y02, y12) - max(y01, y11)
        # intersection = col * row
        # area1 = (x02 - x01) * (y02 - y01)
        # area2 = (x12 - x11) * (y12 - y11)
        # coincide = intersection / (area1 + area2 - intersection)
        return True#coincide
    else:
        return False


def coincide(boxList, poseBox):
    for box in boxList:
        coincideValue = solve_coincide(box, poseBox)
        if coincideValue:
            return coincideValue
    return 0
