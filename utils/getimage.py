import cv2
import pygame


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