import cv2
import pygame

def get_visual_text(width=30,height=30,font_size=20,text="",font_path="./font_ttf/ipag.ttf"):
    pygame.init()
    font = pygame.font.Font(font_path, font_size)     
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

def get_visual_texts(texts,width=20,height=20,font_size=20,font_path="./font_ttf/ipag.ttf"):
    sequence=[]
    for text in texts:
        image=get_visual_text(width=width,height=height,font_size=font_size,text=text,font_path=font_path)
        sequence.append(image)
    concated_image=cv2.hconcat(sequence)
    return concated_image