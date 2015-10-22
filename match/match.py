# encoding:utf-8
# !/usr/bin/env python

import cv
import cv2
import sys
import mycv
import numpy as np
import cvtools
import os
import find

if __name__ == '__main__':
    # if len(sys.argv) < 4:
    #     print "使用方法：match 立体图像左视图目录 立体相机右视图目录 相机参数目录 "
    #     sys.exit(0)
    # imgLeftPath111 = sys.argv[1]
    # imgRightPath111 = sys.argv[2]
    # inputDir111 = sys.argv[3]

    # imgLeft = cv2.imread(imgLeftPath, cv2.IMREAD_GRAYSCALE)
    # imgRight = cv2.imread(imgRightPath, cv2.IMREAD_GRAYSCALE)
    # imageSize = (imgLeft.shape[1], imgLeft.shape[0])
    # K1 = np.loadtxt(os.path.join(inputDir, "K1.txt"))
    # K2 = np.loadtxt(os.path.join(inputDir, "K2.txt"))
    # D1 = np.loadtxt(os.path.join(inputDir, "D1.txt"))
    # D2 = np.loadtxt(os.path.join(inputDir, "D1.txt"))
    # F = np.loadtxt(os.path.join(inputDir, "F.txt"))
    # R = np.loadtxt(os.path.join(inputDir, "R.txt"))
    # T = np.loadtxt(os.path.join(inputDir, "T.txt"))


    path = '/Users/tangling/python_L/match/'
    imgLeft = cv2.imread(path + "images/test/0left.png", cv2.IMREAD_GRAYSCALE)
    imgRight = cv2.imread(path + "images/test/0right.png", cv2.IMREAD_GRAYSCALE)
    # imgLeft = cv2.imread(imgLeftPath, cv2.IMREAD_GRAYSCALE)
    # imgRight = cv2.imread(imgRightPath, cv2.IMREAD_GRAYSCALE)
    imageSize = (imgLeft.shape[1], imgLeft.shape[0])
    K1 = np.loadtxt(path + 'xj/K1.txt')
    K2 = np.loadtxt(path + 'xj/K2.txt')
    D1 = np.loadtxt(path + "xj/D1.txt")
    D2 = np.loadtxt(path + "xj/D1.txt")
    F = np.loadtxt(path + "xj/F.txt")
    R = np.loadtxt(path + "xj/R.txt")
    T = np.loadtxt(path + "xj/T.txt")
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(K1, D1, K2, D2, imageSize, R, T)
    mapl1, mapl2 = cv2.initUndistortRectifyMap(K1, D1, R1, P1, imageSize, cv2.CV_16SC2)
    mapr1, mapr2 = cv2.initUndistortRectifyMap(K2, D2, R2, P2, imageSize, cv2.CV_16SC2)
    left = cv2.remap(imgLeft, mapl1, mapl2, cv2.INTER_LINEAR)
    cv2.namedWindow("left", cv2.WINDOW_NORMAL)
    cv2.imshow("left", left)
    # print imgLeftPath
    right = cv2.remap(imgRight, mapr1, mapr2, cv2.INTER_LINEAR)
    cv2.namedWindow("right", cv2.WINDOW_NORMAL)
    cv2.imshow("right", right)
    # print imgRightPath
    projPointsRight = []
    while True:
        cv.SetMouseCallback("left", mycv.onMouseEvent)
        cv.SetMouseCallback("right", mycv.onMouseEvent)
        c = cv.WaitKey()
        # cv2.destroyAllWindows()
        cv2.line(left, (mycv.points_get[0][0], mycv.points_get[0][1]), (mycv.points_get[1][0], mycv.points_get[1][1]), 255, 2)
        cv2.imshow('left', left)
        print 'mycv.point_get:', mycv.points_get
        for point in mycv.points_get:
            result = find.findPoint(left, right, point)
            find.markPoint(left, 'left', point, 255)
            find.markPoint(right, 'right', result[0][1], 255)
            projPointsRight.append(result[0][1])
        cv2.line(right, (projPointsRight[0][0], projPointsRight[0][1]), (projPointsRight[1][0], projPointsRight[1][1]), 255, 2)
        if c == 27 or c == ord('q'):
            break

    print "测试"

    Dis_P1P2 = -75 / P2[0, 3] * P2[0, 0]
    print "按照相机位置算出来的比例",Dis_P1P2

    # ptLeft_1= mycv.points_get[0]
    # ptLeft_2= mycv.points_get[1]
    #
    # ptRight_1= [483,534]
    # ptRight_2= [566,529]
    #
    # projPointsLeft=[]
    # projPointsRight=[]
    # projPointsLeft.append(ptLeft_1);projPointsRight.append(ptRight_1)
    # projPointsLeft.append(ptLeft_2);projPointsRight.append(ptRight_2)

    projPointsLeft = mycv.points_get

    # projPointsLeft = np.asarray(projPointsLeft, dtype=np.float32).T
    # projPointsRight = np.asarray(projPointsRight, dtype=np.float32).T
#    parallaxDis = mycv.parallaxDis(projPointsLeft,projPointsRight)
#    print "每幅图平均视差",parallaxDis
#     print "三角化法"
#     triangulatePoints4D = cv2.triangulatePoints(P1, P2, projPointsLeft, projPointsRight)
#     triangulatePoints4D = triangulatePoints4D / triangulatePoints4D[3]
#     Dis_t = np.linalg.norm(triangulatePoints4D[:, 0] - triangulatePoints4D[:, 1])
#     print "triangulate算法 空间坐标", triangulatePoints4D
#     print "triangulate算法 空间距离", Dis_t
#     print"triangulate算法 实际距离", Dis_t * Dis_P1P2
#     # parallax
#     print "视差法"
#     parallaxPoints4D = mycv.parallaxPoints(P2, projPointsLeft, projPointsRight, imgLeft.shape[0])
#     Dis_p = np.linalg.norm(parallaxPoints4D[:, 0] - parallaxPoints4D[:, 1])
#     print "视差法 空间坐标", parallaxPoints4D
#     print "parallaxPoints算法 空间距离", Dis_p
#     print "parallaxPoints算法 实际距离", Dis_p * Dis_P1P2
    Dis_t, Dis_P, realDis_t, realDis_p = mycv.computeDistance(P1, P2, imgLeft, projPointsLeft, projPointsRight)

    print "Dis_t:", Dis_t
    print "realDis_t:", realDis_t
    print "Dis_p:", Dis_P
    print "realDis_p:", realDis_p

    cv2.destroyAllWindows()



