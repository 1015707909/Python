#encoding:utf-8
#!/usr/bin/env python
import numpy as np
import os
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
def createPatternPoints(width,height,squareSize=22.25):
    """
    产生棋牌角点坐标，算法是我抄的，与for循环等价并且更简洁
    """
    patternPoints = np.zeros( (width*height, 3), np.float32 )
    patternPoints[:,:2] = np.indices((width,height)).T.reshape(-1, 2) #读不懂就把这句执行一遍 
    patternPoints *= squareSize
    return patternPoints

