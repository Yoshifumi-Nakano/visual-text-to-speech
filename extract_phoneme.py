import argparse
import pathlib
from pathlib import Path
import re
import sys
import glob
import os
import pyopenjtalk
from tqdm import tqdm

from convert_label import read_lab


# full context label to accent label from ttslearn
def numeric_feature_by_regex(regex, s):
    match = re.search(regex, s)
    if match is None:
        return -50
    return int(match.group(1))
def pp_symbols(labels, drop_unvoiced_vowels=True):
    PP = []
    accent = []
    N = len(labels)

    for n in range(len(labels)):
        lab_curr = labels[n]


        p3 = re.search(r"\-(.*?)\+", lab_curr).group(1)

        if drop_unvoiced_vowels and p3 in "AEIOU":
            p3 = p3.lower()

        if p3 == 'sil':
            assert n== 0 or n == N-1
            if n == N-1:
                e3 = numeric_feature_by_regex(r"!(\d+)_", lab_curr)
                if e3 == 0:
                    PP.append("")
                elif e3 == 1:
                    PP.append("")
            continue
        elif p3 == "pau":
            PP.append("sp")
            accent.append('0')
            continue
        else:
            PP.append(p3)
        # アクセント型および位置情報（前方または後方）
        a1 = numeric_feature_by_regex(r"/A:([0-9\-]+)\+", lab_curr)
        a2 = numeric_feature_by_regex(r"\+(\d+)\+", lab_curr)
        a3 = numeric_feature_by_regex(r"\+(\d+)/", lab_curr)
        # アクセント句におけるモーラ数
        f1 = numeric_feature_by_regex(r"/F:(\d+)_", lab_curr)
        lab_next = labels[n + 1]
        a2_next = numeric_feature_by_regex(r"\+(\d+)\+", lab_next)
        # アクセント境界
        if a3 == 1 and a2_next == 1:
            accent.append("#")
        # ピッチの立ち下がり（アクセント核）
        elif a1 == 0 and a2_next == a2 + 1 and a2 != f1:
            accent.append("]")
        # ピッチの立ち上がり
        elif a2 == 1 and a2_next == 2:
            accent.append("[")
        else:
            accent.append('0')
    return PP, accent


if __name__ == '__main__':
    # %%
    transcript_files = glob.glob("/app/JECS/data/neutral/transcript_org.txt")

    # %%
    if not os.path.exists("phoneme/JECS"):
        os.makedirs("phoneme/JECS")

    for transcript in transcript_files:
        with open(transcript, mode='r') as f:
            lines = f.readlines()
        for line in lines:
            #空行は飛ばす
            if line!='\n':
                filename, text = line.split(':')
                if "JA" in filename:
                    text=[l for l in text if l!='＊' and l!=' ' and l!='\n']
                    phoneme=pyopenjtalk.g2p("".join(text))
                    fullcontext_labels=pyopenjtalk.extract_fullcontext("".join(text))
                    print(fullcontext_labels[0])
                    quit()

                    # with open('phoneme/JECS/' + filename + '.lab', mode='w') as f:
                    #     f.write(phoneme.strip('\n'))