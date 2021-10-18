# import argparse
# import os

# import torch
# from torch.nn.parallel.data_parallel import data_parallel
# import yaml
# import torch.nn as nn
# from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter
# from tqdm import tqdm

# from utils.model import get_model, get_vocoder, get_param_num
# from utils.tools import to_device, log, synth_one_sample, plot_alignment_to_numpy
# from model import FastSpeech2Loss
# from dataset import Dataset
from pykakasi import kakasi

kakasi_ = kakasi()
kakasi_.setMode('J', 'H')  # J(Kanji) to H(Hiragana)
kakasi_.setMode('H', 'H') # H(Hiragana) to None(noconversion)
kakasi_.setMode('K', 'H') # K(Katakana) to a(Hiragana)
conv = kakasi_.getConverter()
print(conv.do("上院議員は、私がデータをゆがめたと告発した。"))

# def main(args, configs):
#     text = "形態素解析"
#     # オブジェクトをインスタンス化
#     kakasi = kakasi()
#     # モードの設定：J(Kanji) to H(Hiragana)
#     kakasi.setMode('J', 'H') 

#     # 変換して出力
#     conv = kakasi.getConverter()
#     conv.do(text)  # => けいたいそかいせき


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--restore_step", type=int, default=0)

#     #正式名称と略称を付けられる(-pは--preprocess_configの略称)
#     parser.add_argument(
#         "-p",
#         "--preprocess_config",
#         type=str,
#         required=True,
#         help="path to preprocess.yaml",
#     )
#     parser.add_argument(
#         "-m", "--model_config", type=str, required=True, help="path to model.yaml"
#     )
#     parser.add_argument(
#         "-t", "--train_config", type=str, required=True, help="path to train.yaml"
#     )
#     args = parser.parse_args() #引数の解析

#     # Read Config
#     preprocess_config = yaml.load(
#         open(args.preprocess_config, "r"), Loader=yaml.FullLoader
#     )
#     model_config = yaml.load(open(args.model_config, "r"), Loader=yaml.FullLoader)
#     train_config = yaml.load(open(args.train_config, "r"), Loader=yaml.FullLoader)
#     configs = (preprocess_config, model_config, train_config)

#     main(args, configs)

