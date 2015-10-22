# coding=utf-8

import cv2
import numpy as np
import test

# path = '/Users/tangling/python_L/match/'
# imgLeft = cv2.imread(path + 'images/kt/0left.png', cv2.IMREAD_GRAYSCALE)
# imgRight = cv2.imread(path + 'images/kt/0right.png', cv2.IMREAD_GRAYSCALE)
# imageSize = (imgLeft.shape[1], imgLeft.shape[0])
# K1 = np.loadtxt(path + 'xj/K1.txt')
# K2 = np.loadtxt(path + 'xj/K2.txt')
# D1 = np.loadtxt(path + "xj/D1.txt")
# D2 = np.loadtxt(path + "xj/D1.txt")
# F = np.loadtxt(path + "xj/F.txt")
# R = np.loadtxt(path + "xj/R.txt")
# T = np.loadtxt(path + "xj/T.txt")
# R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(K1, D1, K2, D2, imageSize, R, T)
# mapl1, mapl2 = cv2.initUndistortRectifyMap(K1, D1, R1, P1, imageSize, cv2.CV_16SC2)
# mapr1, mapr2 = cv2.initUndistortRectifyMap(K2, D2, R2, P2, imageSize, cv2.CV_16SC2)
# left = cv2.remap(imgLeft, mapl1, mapl2, cv2.INTER_LINEAR)
# cv2.namedWindow("left", cv2.WINDOW_NORMAL)
# cv2.imshow("left", left)
#
# right = cv2.remap(imgRight, mapr1, mapr2, cv2.INTER_LINEAR)
# cv2.namedWindow("right", cv2.WINDOW_NORMAL)
# cv2.imshow("right", right)
# cv2.waitKey()
# cv2.destroyAllWindows()

def markPoint(im, imname, point, color):
	"""
	将找到的点和选取的点标示出来
	:param im: 图片
	:param imname: 图片名称
	:param point:  点坐标
	:param color:  颜色或灰度值
	:return: 无
	"""
	# print 'markPoint print point', point
	x = point[0]
	y = point[1]
	point = [y, x]       # 标点需要反转坐标
	# 改变point周围十字架的像素值
	for i in range(-3, 4):
		im[point[0] + i, point[1]] = color
	for j in range(-3, 4):
		im[point[0], point[1] + j] = color
	cv2.namedWindow(imname, cv2.WINDOW_NORMAL)
	cv2.imshow(imname, im)
	cv2.waitKey()
	# cv2.destroyAllWindows()

def createMatrix(im, point, size):
	"""
	根据输入的point坐标和size，构建一个以point为中心，size*size大小的矩阵
	:param im: 图像
	:param point: 点坐标
	:param size: 创建矩阵大小
	:return: 创建的矩阵
	"""
	coordinates = []      # 坐标数组
	grayArray = []        # 灰度值数组

	r = (size - 1) / 2
	# 构建坐标数组
	for i in range(-r, r + 1):
		for j in range(-r, r + 1):
			coordinates.append([point[0] + j, point[1] + i])

	# 构建坐标数组对应的灰度值数组,需要反转坐标
	for coordinate in coordinates:
		grayArray.append(im[coordinate[1], coordinate[0]])

	# 把灰度值数组转换成矩阵
	tempArray = np.array(grayArray)
	grayMatrix = tempArray.reshape(size, size)
	return grayMatrix

def findPoint(left, right, point):
	"""
	在右视图中搜索左视图中选取点的匹配点，Y坐标不变
	:param left: 左边图像
	:param right: 右边图像
	:param point: 需要匹配的点坐标
	:return:匹配到的点坐标
	"""
	# print 'findPoint print:point = ', points

	size = 13      # 初始的匹配矩阵大小
	result = []    # 结果数组
	result2 = []
	leftMatrix = createMatrix(left, point, size)   # 构建左边图像的矩阵
	# print '\nleftMatrix\n', leftMatrix

	for x in range(0, point[0]):
		rightPoint = [x, point[1]]
		rightMatrix = createMatrix(right, rightPoint, size)     # 构建右边图像的矩阵
		similap = np.linalg.norm(leftMatrix - rightMatrix)      # 计算矩阵欧式距离或相似度
		# similap = test.euclidSimilar(leftMatrix, rightMatrix)    # 皮尔逊系数
		# similap = test.cosSimilar(leftArray, rightArray)       # 余弦相似度
		# if x == 271:
		# 	print '271 similap', similap
		# 	print '\n271 Matrix\n', rightMatrix
		result.append([similap, rightPoint])           # 构建结果数组，格式：相似度，点坐标

	result.sort()    # 对结果进行按相似度排序

	# 选取第一次选取结果的前十个点，增大搜索窗口进行第二次选取
	p = result[0:10]
	# print p
	leftMatrix2 = createMatrix(left, point, 4 * size + 1)    # 第二次匹配矩阵大小为第一次匹配的矩阵大小的二倍
	for i in p:
		rightPoint = [i[1][0], point[1]]
		rightMatrix = createMatrix(right, rightPoint, 4 * size + 1)
		similap = np.linalg.norm(leftMatrix2 - rightMatrix)
		result2.append([similap, rightPoint])
	result2.sort()
	# print "result2:", result2

	return result2      # 返回第二次匹配中最可能的结果
	#  return result


# points = [[1072, 599], [1123, 590]]

# result = findPoint(left, right, point)
# for i in range(0, 10):
# 	print result[i]
# rightMatrix = createMatrix(right, result[0][1], 11)
# # print '\nrightMatrix:\n', rightMatrix
# markPoint(left, 'left', point, 255)
# markPoint(right, 'right', result[0][1], 255)

# for point in points:
# 	result = findPoint(left, right, point)
# 	markPoint(left, 'left', point, 255)
# 	markPoint(right, 'right', result[0][1], 255)


cv2.destroyAllWindows()

