import re
import argparse
from string import punctuation
import pygame
import cv2

import torch
import yaml
import numpy as np
from torch.utils.data import DataLoader
from g2p_en import G2p
from pypinyin import pinyin, Style

from utils.model import get_model, get_vocoder
from utils.tools import to_device, synth_samples
from utils.getimage import get_text_images
from utils.transform import Phoneme2Kana_inference
from dataset import TextDataset
from text import text_to_sequence, symbols
import pyopenjtalk
from prepare_tg_accent import pp_symbols
from convert_label import openjtalk2julius

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_image(width=30,height=30,font_size=15,text=""):
    pygame.init()
    font = pygame.font.Font("utils/ipag00303/ipag.ttf", font_size)     
    surf = pygame.Surface((width, height))
    surf.fill((255,255,255))

    text_rect = font.render(
        text, True, (0,0,0))
    
    surf.blit(text_rect, [width//2-font_size//2, height//2-font_size//2])  
    
    image = pygame.surfarray.pixels3d(surf)
    image = image.swapaxes(0, 1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    return image


def get_VoicedImage(text=""):
    pygame.init()
    font = pygame.font.Font("utils/ipag00303/ipag.ttf", 15)     
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

def get_text_VocedImage(texts):
    sequence=[]
    for text in texts:
        image=get_VoicedImage(text=text)
        sequence.append(image)
    #横に画像をconcatしている
    concated_image=cv2.hconcat(sequence)
    return concated_image

def synthesize(model, step, configs, vocoder, batchs, control_values):
    preprocess_config, model_config, train_config = configs
    pitch_control, energy_control, duration_control = control_values

    use_accent = preprocess_config["preprocessing"]["accent"]["use_accent"]
    use_image =  preprocess_config["preprocessing"]["image"]["use_image"]
    
    for batch in batchs:
        batch = to_device(batch, device,use_image,use_accent)
        accents = None
        if use_accent:
            accents = batch[-1]
            batch = batch[:-1]
        with torch.no_grad():
            batch = batch[:-1]
            # Forward
            output = model(
                *(batch[2:]),
                p_control=pitch_control,
                e_control=energy_control,
                d_control=duration_control,
                accents=accents
            )
            synth_samples(
                batch,
                output,
                vocoder,
                model_config,
                preprocess_config,
                train_config["path"]["result_path"],
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_step", type=int, required=True)
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="raw text to synthesize, for single-sentence mode only",
    )
    parser.add_argument(
        "--speaker_id",
        type=int,
        default=0,
        help="speaker ID for multi-speaker synthesis, for single-sentence mode only",
    )
    parser.add_argument(
        "-p",
        "--preprocess_config",
        type=str,
        required=True,
        help="path to preprocess.yaml",
    )
    parser.add_argument(
        "-m", "--model_config", type=str, required=True, help="path to model.yaml"
    )
    parser.add_argument(
        "-t", "--train_config", type=str, required=True, help="path to train.yaml"
    )
    parser.add_argument(
        "--pitch_control",
        type=float,
        default=1.0,
        help="control the pitch of the whole utterance, larger value for higher pitch",
    )
    parser.add_argument(
        "--energy_control",
        type=float,
        default=1.0,
        help="control the energy of the whole utterance, larger value for larger volume",
    )
    parser.add_argument(
        "--duration_control",
        type=float,
        default=1.0,
        help="control the speed of the whole utterance, larger value for slower speaking rate",
    )
    parser.add_argument(
        "--isVoced",
        type=bool,
        default=True,
    )
    args = parser.parse_args()

    # Read Config
    preprocess_config = yaml.load(
        open(args.preprocess_config, "r"), Loader=yaml.FullLoader
    )
    model_config = yaml.load(open(args.model_config, "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(args.train_config, "r"), Loader=yaml.FullLoader)
    configs = (preprocess_config, model_config, train_config)

    # Get model
    model = get_model(args, configs, device, train=False)

    # Load vocoder
    vocoder = get_vocoder(model_config, device)

    # Preprocess texts
    symbol_to_id = {s: i for i, s in enumerate(symbols)}
    accent_to_id = {'0':0, '[':1, ']':2, '#':3}

    # create batch list
    input_kanas=["あ","い","え","お","な","に","ぬ","ね","の","ま","み","む","め","も","わ","を","ん","ら","り","る","れ","ろ","や","ゆ","よ"]
    batchs=[]
    N= 5

    for input_kana in input_kanas:
        ids = raw_texts = [input_kana]
        speakers = np.array([args.speaker_id])
        image=[get_text_VocedImage([input_kana]*N)]
        # cv2.imwrite("output_ver6/result/test2.jpg",image[0])
        #image=[get_text_images([input_kana]*N,width=30,height=30,font_size=15)]
        texts = np.array([[input_kana]*N])
        text_lens = np.array([len(texts[0])])
        accents = None
        batchs.append((ids, raw_texts, speakers, texts, text_lens, max(text_lens),None,None,None,None,None,None,accents,image))
    
    control_values = args.pitch_control, args.energy_control, args.duration_control

    synthesize(model, args.restore_step, configs, vocoder, batchs, control_values)