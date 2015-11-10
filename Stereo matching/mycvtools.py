#encoding:utf-8
#!/usr/bin/env python
import numpy as np
import os
import cv2

def isImage(fileName):
    """
    判断fileName是否是图像文件（支持BMP、JPG和PNG）
    """
    fileName=fileName.upper()
    if fileName.endswith("BMP") or fileName.endswith("JPG") or fileName.endswith("PNG"):
        return True
    else:
        return False
def loadImages(imageDir):
    """
    返回imageDir下的图像文件列表
    """
    images=[]
    for f in os.listdir(imageDir):
        if isImage(f):
            images.append(os.path.join(imageDir,f))
    return images;
def createPatternPoints(width, height, squareSize = 26):
    """
    产生棋牌角点坐标，算法是我抄的，与for循环等价并且更简洁
    """
    patternPoints = np.zeros((width * height, 3), np.float32)
    patternPoints[:, :2] = np.indices((width, height)).T.reshape(-1, 2)  # 读不懂就把这句执行一遍
    patternPoints *= squareSize
    return patternPoints


class ZeroBorderImage:
    def __init__(self, img, r):
        # 为图像生成宽度为r的边界，防止在边界构造像素矩阵时超界
        self.img = np.float32(cv2.copyMakeBorder(img, r, r, r, r, cv2.BORDER_CONSTANT, value=(0, 0, 0)))
        # 获取图像的宽度和高度
        self.w = img.shape[1]
        self.h = img.shape[0]
        # 构造矩阵的的半径
        self.r = r

    def getRegion(self, x, y):
        """
        获取图像中以（x，y）为中心，r为半径的领域矩阵
        """
        if x < 0 or y < 0 or x > self.w - 1 or y > self.w - 1:
            raise ValueError, "x的范围必须为[0,%d],y的范围必须是[0,%d]"%(self.w,self.h)
        return self.img[y:y + self.r * 2 + 1, x:x + self.r * 2 + 1]

    def search(self, w, y):
        ssd = []
        for x in range(0, self.w):
            region = self.getRegion(x, y)
            d = np.sqrt(np.sum((w - region)**2))
            ssd.append(d)
        return ssd

    """
    在给定的index中搜索，用于第二次扩大窗口搜索
    """
    def searchWithIndex(self, w, index, y):
        ssd = []
        for x in index:
            region = self.getRegion(x,y)
            d = np.sqrt(np.sum((w - region)**2))
            ssd.append([d,x])
        ssd.sort
        # print ssd
        point = [ssd[0][1],y]
        # 返回SSD最小的点
        return point
