# encoding:utf-8
# !/usr/bin/env python
import cv2
import numpy as np
import cvtools

points_get = []      # 用于传回获取到的点坐标
def loadImagePoints(imageDir, width, height, debug=False):
    imageSize = (0, 0)
    imagePoints = []
    nimages = 0
    images = cvtools.loadImages(imageDir)
    for imageName in images:
        # 读取图片时转换为灰度图像
        im = cv2.imread(imageName, 0)
        if debug:
            print "正在处理图像:",imageName
        imageSize = (im.shape[1], im.shape[0])
        retval, corners = cv2.findChessboardCorners(im, (width, height))
        if retval:
            cv2.cornerSubPix(im, corners, (11, 11), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1))
            nimages = nimages + 1
            if debug:
                im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
                cv2.drawChessboardCorners(im, (width, height), corners, retval)
                cv2.namedWindow(imageName, cv2.WINDOW_NORMAL)
                cv2.imshow(imageName, im)
#                print corners.reshape(-1,2)
                cv2.waitKey()
            imagePoints.append(corners.reshape(-1, 2))
    cv2.destroyAllWindows()
    print "imageSize is", imageSize;
    return imagePoints, nimages, imageSize
def runcalibrate(width, height, imagePoints, nimages, imageSize, flags=None):
    if flags is None:
        flags = cv2.CALIB_FIX_ASPECT_RATIO + cv2.CALIB_ZERO_TANGENT_DIST + cv2.CALIB_SAME_FOCAL_LENGTH + cv2.CALIB_FIX_K3 + cv2.CALIB_FIX_K4 + cv2.CALIB_FIX_K5
    objectPoints = []
    patternPoints = cvtools.createPatternPoints(width, height)
    for i in range(0, nimages):
        objectPoints.append(patternPoints)
    return cv2.calibrateCamera(objectPoints, imagePoints, imageSize, flags=flags)
def calibrateCamera(imageDir, width, height, debug=False, flags=None):
    """
        用imageDir下的图片进行相机标定
        width：棋牌的宽
        height：棋牌的高
        debug：是否显示调试信息
        flags：参考opencv的calibrateCamera函数
        """
    imagePoints, nimages, imageSize = loadImagePoints(imageDir, width, height, debug)
    return runcalibrate(width, height, imagePoints, nimages, imageSize, flags)
def stereoCalibrate(imageDir, K, D, width, height, debug=False, flags=cv2.CALIB_FIX_INTRINSIC):
    """
        width：棋牌的宽
        height：棋牌的高
        debug：输出调试信息
        flags：参考opencv的stereoCalibrate函数
        """
    imagePoints, nimages, imageSize = loadImagePoints(imageDir, width, height, debug)
    objectPoints = []
    patternPoints = cvtools.createPatternPoints(width, height)
    imagePoints1 = []
    imagePoints2 = []
    for i in range(0, len(imagePoints)):
        if i % 2 == 0:
            imagePoints1.append(imagePoints[i])
        else:
            imagePoints2.append(imagePoints[i])
    for i in range(0, nimages / 2):
        objectPoints.append(patternPoints)
    return cv2.stereoCalibrate(objectPoints, imagePoints1, imagePoints2, imageSize,K,D,K,D,criteria=(cv2.TERM_CRITERIA_COUNT+cv2.TERM_CRITERIA_EPS, 100, 1e-5),flags=flags)

def parallaxPoints(P2, projCornersLeft, projCornersRight,imageHeight):
    points4D = np.zeros((3, projCornersLeft.shape[1]))#存放三维点
    for i in range(0, projCornersLeft.shape[1]):
        #Z
        points4D[2, i] = -P2[0, 3] / (projCornersLeft[0, i] - projCornersRight[0, i])
        #X
        points4D[0, i] = points4D[2, i] / P2[0, 0] * (projCornersLeft[0, i] - P2[0, 2])
        #Y
        # points4D[1,i] = points4D[2,i] / P2[0,0] * (imageHeight- projCornersRight[1,i] - P2[1,2])
        points4D[1, i] = points4D[2, i] / P2[0, 0] * (projCornersRight[1, i] - P2[1, 2])
    return points4D

def parallaxDis(projCornersLeft, projCornersRight):
    dis = 0;
    print "匹配点视差"
    for i in range(0, projCornersLeft.shape[1]):
        print projCornersLeft[0, i] - projCornersRight[0, i]
        dis += projCornersLeft[0, i] - projCornersRight[0, i]
    dis = dis / projCornersLeft.shape[1]
    return dis

def onMouseEvent(event, x, y, flags, param):
#        global startPointx
#        global startPointy
        global flagDraw
        # 鼠标右键按下响应
        if(event==2):
            print "Position is", "[" + str(x) + "," + str(y) + "]"      # 获取到的点坐标
            points = [y, x]           # 标出选取点
            global points_get
            points_get.append([x, y])
            # point_get = [x, y]

def drawline(im, star, end, color):
    cv2.line(im, star, end, [0, 255, 0])

def get8Max(point):
    """
    得到输入点的8领域
    :param point: 中心点坐标
    :return:存储中心点的8领域坐标的list
    """
    leftMax = []
    for x in range(point[0] - 1, point[0] + 1):
        for y in range(point[1] - 1, point[1] + 1):
            leftMax.append([x, y])

    return leftMax

def computeDistance(P1, P2,im, leftPoints, rightPoints):
    """
    根据输入的点得坐标，根据三角化法和视差法分别求解两点之间的实际距离
    :param P1: 相机参数
    :param P2: 相机参数
    :im:左视图
    :param leftPoint: 两个点坐标，在左视图选取的点得坐标
    :param rightPoint: 与左视图选取的点在右视图中匹配到的点的坐标
    :return: triangulatePoints4D,Dis_t,triangulateRealDis，parallaxPoints4D,Dis_p,parallaxRealDis
    :返回值：三角化法求得的空间点坐标、空间平均距离、真实平均距离；视差法求得的空间点坐标、空间平均距离、真实平均距离
    """
    # 计算真实距离用的
    Dis_P1P2 = -75 / P2[0, 3] * P2[0, 0]

    # 左边点1的领域
    leftPoint1 = leftPoints[0]
    # print "leftPoint1:", leftPoint1
    leftMax1 = get8Max(leftPoint1)
    # 左边点2的领域
    leftPoint2 = leftPoints[1]
    leftMax2 = get8Max(leftPoint2)

    # 右边点1的领域
    rightPoint1 = rightPoints[0]
    rightMax1 = get8Max(rightPoint1)
    # 右边点2的领域
    rightPoint2 = rightPoints[1]
    rightMax2 = get8Max(rightPoint2)

    # 一共有81对左边点坐标组合，用于计算距离
    projLeftPoints = []
    for l1 in leftMax1:
        for l2 in leftMax2:
            projLeftPoints.append([l1, l2])
    # 有81对右边点坐标组合，每个位置的点对都是上面左边对应位置的点对的匹配点
    projRightPoints = []
    for r1 in rightMax1:
        for r2 in rightMax2:
            projRightPoints.append([r1, r2])

    # 计算所有点对的距离
    sum_t = 0.0  # 三角化法求得的距离之和
    realSum_t = 0.0
    sum_p = 0.0  # 视差法求得的距离之和
    realSum_p = 0.0

    allDis_t = [] # 三角化法求得的距离
    allRealDis_t = []
    allDis_p = [] # 视差法求得的距离
    allRealDis_p = []
    for i in range(0, len(projLeftPoints)):

        templeft = np.asarray(projLeftPoints[i], dtype=np.float32).T
        tempright = np.asarray(projRightPoints[i], dtype=np.float32).T
        # 三角化法
        triangulatePoints4D = cv2.triangulatePoints(P1, P2, templeft, tempright)
        # 空间坐标
        triangulatePoints4D = triangulatePoints4D / triangulatePoints4D[3]
        # 当前点对的空间距离
        tempDis_t = np.linalg.norm(triangulatePoints4D[:, 0] - triangulatePoints4D[:, 1])
        # 把空间距离保存
        allDis_t.append(tempDis_t)
        sum_t = sum_t + tempDis_t
        # 计算实际距离并保存
        allRealDis_t.append(tempDis_t * Dis_P1P2)
        realSum_t = realSum_t + tempDis_t * Dis_P1P2

        # 视差法
        parallaxPoints4D = parallaxPoints(P2, templeft, tempright, im.shape[0])
        # 空间距离
        tempDis_p = np.linalg.norm(parallaxPoints4D[:, 0] - parallaxPoints4D[:, 1])
        # 保存计算结果
        allDis_p.append(tempDis_p)
        sum_p = sum_p + tempDis_p
        allRealDis_p.append(tempDis_p * Dis_P1P2)
        realSum_p = realSum_p + tempDis_p * Dis_P1P2

    # 计算空间距离的平均值
    Dis_t = sum_t / len(allDis_t)  # 三角化法
    Dis_P = sum_p / len(allDis_p)  # 视差法

    # 计算真实距离平均值
    realDis_t = realSum_t / len(allRealDis_t)  # 三角化法
    realDis_p = realSum_p / len(allRealDis_p)  # 视差法

    return Dis_t, Dis_P, realDis_t, realDis_p





