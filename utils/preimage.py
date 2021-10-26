import numpy as np
import cv2

#一文字を画像に変える

def pre(image,a=0.8,b=1.2):
    w,h=image.shape
    #0.8~1,2の一様乱数
    rand=(b-a) * np.random.rand() + a
    #resize
    image=cv2.resize(image,(int(w*rand),int(h*rand)))
    st_h=int((rand-1)*w//2)
    
    st_w=int((rand-1)*h//2)
    #拡大の場合はcropする
    if rand>=1:
        image=image[st_h:st_h+w,st_w:st_w+h]
    #縮小
    else:
        image=cv2.copyMakeBorder(image, abs(st_h), abs(st_h), abs(st_w), abs(st_w),cv2.BORDER_CONSTANT,value=[255,255])
    image=cv2.resize(image,(30,30))
    return image


def pre_seq(image):
    h,w=image.shape
    sequence=[]
    for i in range(0,w-1,30):
        sequence.append(pre(image[:,i:i+30]))
    concated_image=cv2.hconcat(sequence)
    return concated_image