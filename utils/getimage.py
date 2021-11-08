import cv2
import pygame
from utils.transform import Phoneme2Kana_emp
import numpy as np


def get_image(width=20,height=20,font_size=10,text=""):
    pygame.init()
    font = pygame.font.Font("./utils/ipag00303/ipag.ttf", font_size)     
    surf = pygame.Surface((width, height))
    surf.fill((255,255,255))


    text_rect = font.render(
        text, True, (0,0,0))
    
    if len(text)==1:
        surf.blit(text_rect, [width//2-font_size//2, height//2-font_size//2])  
    else:
        assert False
        surf.blit(text_rect, [width//2-font_size, height//2-font_size//2])  
    
    image = pygame.surfarray.pixels3d(surf)
    image = image.swapaxes(0, 1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    return image

def get_text_images(texts,width=20,height=20,font_size=10):
    sequence=[]
    for text in texts:
        image=get_image(width=width,height=height,font_size=font_size,text=text)
        sequence.append(image)
    #横に画像をconcatしている
    concated_image=cv2.hconcat(sequence)
    return concated_image



#存在しない文字に濁点をつけるコード
def get_VoicedImage(text=""):
    pygame.init()
    font = pygame.font.Font("./utils/ipag00303/ipag.ttf", 15)     
    surf = pygame.Surface((30, 30))
    surf.fill((255,255,255))
    text_rect = font.render(
        text, True, (0,0,0))
    
    text_rect2 = font.render(
        "゛",True,(0,0,0)
    )
    
    surf.blit(text_rect, [8, 8]) 
    
    if text in ["あ","い","え","も","わ","を","ん","ら","る","れ","ろ"]:
        surf.blit(text_rect2, [18,8])
    
    if text in ["お","な","ぬ","ね","の","ま","み","む","め","や","ゆ","よ"]:
        surf.blit(text_rect2, [20,8])
    
    if text in ["に"]:
        surf.blit(text_rect2, [21,8])
        
    if text in ["り"]:
        surf.blit(text_rect2, [19,6])
    
    image = pygame.surfarray.pixels3d(surf)
    image = image.swapaxes(0, 1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    return image

#testセットに対してランダムに濁点をつけて画像を得る
def get_voced_images(texts,index,width=30,height=30,font_size=15):
    sequence=[]
    for i in range(len(texts)):
        text=texts[i]
        if i==index:
            image=get_VoicedImage(text=text)
        else:
            image=get_image(width=width,height=height,font_size=font_size,text=text)
        sequence.append(image)
    concated_image=cv2.hconcat(sequence)
    return concated_image

#一つの文字を太文字に変換する
def get_bold_image(width=20,height=20,font_size=10,text="",flg=False):
    pygame.init()
    font = pygame.font.Font("./utils/ipag00303/ipag.ttf", font_size)     
    if flg:
        font.bold=True
    surf = pygame.Surface((width, height))
    surf.fill((255,255,255))
    
    text_rect = font.render(
        text, True, (0,0,0))
    
    if len(text)==1:
        surf.blit(text_rect, [width//2-font_size//2, height//2-font_size//2])  
    else:
        assert False
        surf.blit(text_rect, [width//2-font_size, height//2-font_size//2])  
    
    image = pygame.surfarray.pixels3d(surf)
    image = image.swapaxes(0, 1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    return image

#textの太文字を得る関数
def get_bold_text_images(texts,flgs,width=20,height=20,font_size=10):
    sequence=[]
    for i in range(len(texts)):
        text=texts[i]
        flg=flgs[i]
        image=get_bold_image(width=width,height=height,font_size=font_size,text=text,flg=flg)
        sequence.append(image)
    concated_image=cv2.hconcat(sequence)
    return concated_image


#どの文字を太字にするのかのフラグを作る関数
def get_flg(basename,speaker,kanas):
    flgs=[False]*len(kanas)
    with open("phoneme/"+speaker+"/Emp/"+basename+".lab","r") as f:
        f=f.read()
        phoneme=f.split(" ")
        emp_kanas=Phoneme2Kana_emp(phoneme)
    assert kanas == [t for t in emp_kanas if t!="＊"]
    for i in range(len(emp_kanas)):
        if emp_kanas[i]=="＊":
            left=i
            break
    for i in range(len(emp_kanas)-1,-1,-1):
        if emp_kanas[i]=="＊":
            right=i
            break
    assert left!=-1 and right!=-1 and left!=right and left<right
    for i in range(left,right-1):
        flgs[i]=True
    return flgs

def openjtalk2julius(p3):
    if p3 in ['A','I','U',"E", "O"]:
        return p3.lower()
    if p3 == 'cl':
        return 'q'
    if p3 == 'pau':
        return 'sp'
    return p3