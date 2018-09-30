#-*- coding: utf-8 -*-
from __future__ import division
import cv2
import numpy as np
import os
import pdb
import argparse
from matplotlib import pyplot as plt

def Splice(img1, img2):
    img1_Gauss = cv2.GaussianBlur(img1,(5,5),0)
    img2_Gauss = cv2.GaussianBlur(img2,(5,5),0)
    #实例化SIFT
    sift = cv2.xfeatures2d.SIFT_create()
    #得到两幅图像的特征点
    kp1, des1 = sift.detectAndCompute(img1_Gauss, None)
    kp2, des2 = sift.detectAndCompute(img2_Gauss, None)
    img_KeypointsDraw=cv2.drawKeypoints(img1,kp1,np.array([]),(255,0,0),cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS) #绘制关键点
    #plt.imshow(img_KeypointsDraw), plt.show()
    #cv2.imwrite('./Pics/KeyPoints.png',img_KeypointsDraw)

    #实例化匹配器
    bf = cv2.BFMatcher()
    #匹配特征点，采用1NN（1近邻）匹配
    matches = bf.knnMatch(des1, des2, k=1)

    #画出并保存匹配结果(仅在实验中使用)
    #img_MatchesDraw = cv2.drawMatchesKnn(img1, kp1, img2, kp2, matches, None, flags=2)
    #plt.imshow(img_MatchesDraw), plt.show()
    #cv2.imwrite('./Pics/Matches1NN.png',img_MatchesDraw)


    #重新匹配特征点，并采用1NN/2NN<0.7的方式筛选出好的匹配对
    matches = bf.knnMatch(des1, des2, k=2)
    #good_matches用于保存好的匹配对
    good_matches = []
    for m, n in matches:
        if m.distance < 0.70 * n.distance:
            good_matches.append(m)

    #画出并保存匹配结果（仅在实验中使用）
    #img_MatchesDraw = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, flags=2)
    #plt.imshow(img_MatchesDraw), plt.show()
    #cv2.imwrite('./Pics/Matches2NN.png',img_MatchesDraw)

    #得到良好匹配对的坐标对
    good_kp1=[]
    good_kp2=[]
    for m in good_matches:
        good_kp1.append(kp1[m.queryIdx].pt)
        good_kp2.append(kp2[m.trainIdx].pt)

    good_kp1=np.array(good_kp1)
    good_kp2=np.array(good_kp2)
    #用RANSAC算法得到最佳透视变换矩阵
    retval, mask=cv2.findHomography(good_kp2, good_kp1, cv2.RANSAC,confidence=0.997)

    #得到右边图片经过透视变换后四个角落点的坐标
    (x_upleft,y_upleft,one)=np.dot(np.array((0,0,1)), retval.T)
    (x_upleft,y_upleft)=(int(x_upleft/one),int(y_upleft/one))

    (x_bottomleft,y_bottomleft,one)=np.dot(np.array((0,img2.shape[0]-1,1)),retval.T)
    (x_bottomleft,y_bottomleft)=(int(x_bottomleft/one),int(y_bottomleft/one))

    (x_bottomright,y_bottomright,one)=np.dot(np.array((img2.shape[1]-1,img2.shape[0]-1,1)),retval.T)
    (x_bottomright,y_bottomright)=(int(x_bottomright/one),int(y_bottomright/one))

    (x_upright,y_upright,one)=np.dot(np.array((img2.shape[1]-1,0,1)),retval.T)
    (x_upright,y_upright)=(int(x_upright/one),int(y_upright/one))

    #得到两幅图像重叠区域的起始坐标
    x_end=img1.shape[1]-1
    y_end=img1.shape[0]-1
    x_beg=min(x_upleft,x_bottomleft)

    #规定拼接图像的大小
    rows_out=max(img1.shape[0],y_bottomleft+1,y_bottomright+1)
    cols_out=max(img1.shape[1],x_bottomright+1, x_upright+1)

    #将经过透视变换后的右边图片加入拼接图像
    img_out = cv2.warpPerspective(img2, retval, (cols_out,rows_out))

    #将左边图片中不与右边图片重叠的部分加入拼接图像
    img_out[:img1.shape[0],:x_beg]=img1[:img1.shape[0],:x_beg]


    #将两张图片重叠的部分经过加权平均合成，再加入拼接图像
    for y_iter in range(img1.shape[0]):
        x_line=int(((y_iter-y_bottomleft)*x_upleft+(y_upleft-y_iter)*x_bottomleft)/(y_upleft-y_bottomleft))
        for x_iter in range(x_beg, img1.shape[1]):
            a=x_end - x_iter
            b=x_end - x_line
            weight = (x_end - x_iter) / (x_end - x_line)
            if ~img_out[y_iter][x_iter].any():
                img_out[y_iter][x_iter]=img1[y_iter][x_iter]
            elif ~img1[y_iter][x_iter].any():
                img_out[y_iter][x_iter]=img_out[y_iter][x_iter]
            else:
                img_out[y_iter][x_iter]=weight*img1[y_iter][x_iter]+(1-weight)*img_out[y_iter][x_iter]

    #显示并保存拼接图像
    #plt.imshow(img_out), plt.show()
    return img_out

def Splice_All(img_list, img_dir):
    num = len(img_list)
    if num > 2:
        if num % 2 == 0:
            return Splice(
                Splice_All(img_list[:num//2], img_dir),
                Splice_All(img_list[num//2:], img_dir)
                )
        else:
            return Splice(
                Splice_All(img_list[:(num+1)//2], img_dir),
                Splice_All(img_list[(num+1)//2:], img_dir)
            )
    else:
        if num == 2:
            img0 = cv2.imread(os.path.join(img_dir, img_list[0]))
            img1 = cv2.imread(os.path.join(img_dir, img_list[1]))
            return Splice(img0, img1)
        else:
            return cv2.imread(os.path.join(img_dir, img_list[0]))

def is_img(file_name, format):
    return any([file_name.endswith('.'+format[i]) or file_name.endswith('.'+format[i].upper()) for i in range(len(format))])


def main():
    
    #参数设置
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir', default='./Pics/')
    parser.add_argument('--result_dir', default='./Result/')
    parser.add_argument('--result_name', default='result.jpg')
    parser.add_argument('--format', default='jpg png')
    config = parser.parse_args()
    
    format = config.format.split()
    img_dir = config.img_dir
    result_dir = config.result_dir
    result_name = config.result_name
    img_list = [each for each in sorted(os.listdir(img_dir)) if is_img(each,format)]

    #合成！
    img_out = Splice_All(img_list, img_dir)

    cv2.imwrite(os.path.join(result_dir,result_name),img_out)

if __name__ == '__main__':
    main()