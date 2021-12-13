import os
import random
import json
import numpy as np
import pyworld as pw
import cv2
import pygame 

def get_image(width=20,height=20,font_size=10,text="",font_path=""):
    pygame.init()
    font = pygame.font.Font(font_path, font_size)     
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

def get_text_images(texts,width=20,height=20,font_size=10,font_path=""):
    sequence=[]
    for text in texts:
        image=get_image(width=width,height=height,font_size=font_size,text=text,font_path=font_path)
        sequence.append(image)
    #横に画像をconcatしている
    concated_image=cv2.hconcat(sequence)
    return concated_image


paths = os.listdir("./preprocessed_data/JSUT/text_kana")
for path in paths:
    basename=path.split(".")[0][5:]

    with open("./preprocessed_data/JSUT/text_kana/"+path) as f:
        kanas=f.read()

    iamge_filename="JSUT-image-30-30-15-{}.jpg".format(basename)
    text_image=get_text_images(
        texts=[t for t in kanas.replace("{", "").replace("}", "").split()],
        width=30,
        height=30,
        font_size=15,
        font_path="./utils/Koruri-20210720/Koruri-Regular.ttf"
    )
    iamge_filename="JSUT-image-30-30-15-{}.jpg".format(basename)
    cv2.imwrite(os.path.join("./preprocessed_data/JSUT","image_kana_koruri",iamge_filename),text_image)

    text_image=get_text_images(
        texts=[t for t in kanas.replace("{", "").replace("}", "").split()],
        width=30,
        height=30,
        font_size=15,
        font_path="./utils/aiharalaisyo/Aiharahudemojikaisho_free304.ttf"
    )
    iamge_filename="JSUT-image-30-30-15-{}.jpg".format(basename)
    cv2.imwrite(os.path.join("./preprocessed_data/JSUT","image_kana_aihara",iamge_filename),text_image)