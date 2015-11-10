# encoding:utf-8
import cv
import cv2
import sys
import mycv
import mytest
import mycvtools
import os
import numpy as np
import matplotlib.pyplot as plt



# 读图像
imgLeft = cv2.imread("test/0left.png",cv2.IMREAD_GRAYSCALE)
imgRight = cv2.imread("test/0right.png",cv2.IMREAD_GRAYSCALE)
imageSize = (imgLeft.shape[1], imgLeft.shape[0])
# 读相机参数
K1 = np.loadtxt("xj/K1.txt")
K2 = np.loadtxt("xj/K2.txt")
D1 = np.loadtxt("xj/D1.txt")
D2 = np.loadtxt("xj/D1.txt")
F = np.loadtxt("xj/F.txt")
R = np.loadtxt("xj/R.txt")
T = np.loadtxt("xj/T.txt")

# 图像矫正
R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(K1, D1, K2, D2, imageSize, R, T)
mapl1, mapl2 = cv2.initUndistortRectifyMap(K1, D1, R1, P1, imageSize, cv2.CV_16SC2)
mapr1, mapr2 = cv2.initUndistortRectifyMap(K2, D2, R2, P2, imageSize, cv2.CV_16SC2)
# 矫正后的图像
left = cv2.remap(imgLeft, mapl1, mapl2, cv2.INTER_LINEAR)
right = cv2.remap(imgRight, mapr1, mapr2, cv2.INTER_LINEAR)
# 初始r设置为64
r = 64
# 初始化显示窗口
cv2.namedWindow("right", cv2.WINDOW_NORMAL)
cv2.namedWindow("left", cv2.WINDOW_NORMAL)
# 设置鼠标响应事件
cv2.setMouseCallback("left", mytest.onMouseDoubleClick)
# 显示图像
cv2.imshow("right", right)
cv2.imshow("left", left)

# 按ESC退出
while (cv2.waitKey() != 27):
    continue
plt.close("all")
cv2.destroyAllWindows()

# print mytest.pointGot
# 鼠标点击获得的点，只有前两次点击有效
leftPoints = []
if len(mytest.pointGot) >= 2:
    for i in range(2):
        leftPoints.append(mytest.pointGot[i])
    print "你一共选择了%d个点，只有前两个点有效：" % len(mytest.pointGot),
    print leftPoints
elif len(mytest.pointGot) < 2:
    print "请选两个点！"
    exit()
# print leftPoints
# 画出选出两个点，并连线
cleft = mytest.drawLineInLeft(left,leftPoints)
# 选出的两个点的匹配点
rightPoints = []
# 保存对应点的ssd
ssds = []
# 15个候选点的索引
indexs = []
for p in leftPoints:
    tempSSD,tempRP ,tempIndex= mytest.getIndex(p[0], p[1], left, right, r)
    ssds.append(tempSSD)
    rightPoints.append(tempRP)
    indexs.append(tempIndex)
# print rightPoints
cright = mytest.drawLineInRight(right, rightPoints)

print "测试："
Dis_t, Dis_P, realDis_t, realDis_p = mycv.computeDistance(P1, P2, left, leftPoints, rightPoints)
print "三角化法："
print "空间距离：",Dis_t
print "实际距离：",realDis_t
print "视差法："
print "空间距离：",Dis_P
print "实际距离：",realDis_p

# 绘制左右视图
plt.figure(1)
ax1 = plt.subplot(211)
ax2 = plt.subplot(212)
plt.sca(ax1)
plt.title("left")
plt.imshow(cleft, "gray")
plt.sca(ax2)
plt.title("right")
plt.imshow(cright, "gray")

plt.xlabel("The Distance between two points is: " + str(realDis_t) + " mm")
plt.title("right")
# plt.show()
cv2.imshow("draw:right", cright)

# 绘制ssd曲线
plt.figure(2)
plt.title("SSD")
ax3 = plt.subplot(211)
ax4 = plt.subplot(212)
plt.sca(ax3)
plt.title("left_point1:" + str(object=rightPoints[0]))
line1, = plt.plot(ssds[0])
plt.plot(indexs[0], ssds[0][indexs[0]], "ro")
line1.set_label("SSD")
plt.sca(ax4)
plt.title("left_point2:" + str(object=rightPoints[1]))
line2, = plt.plot(ssds[1])
plt.plot(indexs[1], ssds[1][indexs[1]], "ro")
line2.set_label("SSD")
plt.legend()
plt.show()

cv2.waitKey()
cv2.destroyAllWindows()
plt.close("all")