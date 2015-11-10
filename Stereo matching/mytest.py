# encoding:utf-8

import mycvtools
import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal

# 传回鼠标点击得到的点坐标
pointGot = []

"""
根据输入的点坐标，左右视图，搜索半径，返回指定个数的候选匹配点
x，y:源点的坐标
r:搜索窗口大小
leftImage,rightImage:左右视图
num:返回候选点个数
"""
def getIndex(x, y, left, right, r, num = 15):

    # 根据传入的图像初始化第一次小窗口搜索的视图
    bleft = mycvtools.ZeroBorderImage(left, r)
    bright = mycvtools.ZeroBorderImage(right, r)
    # 计算SSD
    wighth = bleft.getRegion(x, y)
    ssd = np.array(bright.search(wighth, y))
    # 计算峰值点索引，即是峰值点的x坐标
    index = signal.find_peaks_cwt(-ssd, np.arange(1,50))
    # 获得最小的num个索引
    minIndex = np.argsort(ssd[index])[:num]
    index = np.array(index)[minIndex]
    #
    # 第二次扩大窗口在第一次搜索的结果中二次匹配
    # 初始化第二次扩大窗口搜索的视图
    bleft2 = mycvtools.ZeroBorderImage(left, r + r)
    bright2 = mycvtools.ZeroBorderImage(right, r + r)
    wighth2 = bleft2.getRegion(x, y)
    # 第二次匹配结果
    rightPoint = bright2.searchWithIndex(wighth2, index, y)

    # 所有ssd,返回最小的num个索引
    return ssd,rightPoint,index
# 画点和线
def drawLineInRight(right,points):
    cv2.namedWindow("draw:right", cv2.WINDOW_NORMAL)
    cright = right.copy()
    for point in points:
        cv2.circle(cright, (point[0], point[1]), 6, (0, 0, 255), -1)
    cv2.line(cright, (points[0][0], points[0][1]), (points[1][0],points[1][1]), (0, 0, 255), 5)
    cv2.imshow("draw:right", cright)
    return cright

def drawLineInLeft(left,points):
    cv2.namedWindow("draw:left", cv2.WINDOW_NORMAL)
    cleft = left.copy()
    for point in points:
        cv2.circle(cleft, (point[0], point[1]), 6, (0, 0, 255), -1)
    # cv2.circle(cleft, (x, y), 6, (0, 0, 255), -1)
    cv2.line(cleft, (points[0][0], points[0][1]), (points[1][0],points[1][1]), (0, 0, 255), 5)
    cv2.imshow("draw:left", cleft)
    return cleft
# 鼠标事件
def onMouseDoubleClick(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        # print x
        # print y

        global pointGot
        pointGot.append([x, y])
