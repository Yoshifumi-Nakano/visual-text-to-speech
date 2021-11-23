import cv2
import pygame


def get_image(width=30,height=30,font_size=20,text=""):
    pygame.init()
    font = pygame.font.Font("./utils/ipag00303/ipag.ttf", font_size)     
    surf = pygame.Surface((width, height))
    surf.fill((255,255,255))

    text_rect = font.render(
        text, True, (0,0,0))
    
    if text!="sp":
        surf.blit(text_rect, [width//2-font_size//3.5, height//2-font_size//2])  

    
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
