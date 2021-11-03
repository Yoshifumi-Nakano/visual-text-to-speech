import argparse
import pathlib
from pathlib import Path
import re
import sys
import glob
import os
import pyopenjtalk
from tqdm import tqdm

from convert_label import openjtalk2julius

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
                if "JA" in filename and "EMPH" in filename:
                    #強調を取り除いて作った音素
                    text_no_emp=[l for l in text if l!='＊' and l!=' ' and l!='\n']
                    phoneme_no_emp=pyopenjtalk.g2p("".join(text_no_emp))

                    #強調で分けて音素
                    text_sequence=[l for l in text if l!=' ' and l!='\n']
                    text_sequence = "".join(text_sequence)
                    text_sequence=text_sequence.split("＊")
                    phoneme_with_emp=""
                    phoneme_with_empMark=""
                    for te in text_sequence:
                        if te=='':
                            phoneme_with_empMark+="＊ "
                            continue
                        if te[0]=="、" or te[0]=="・":
                            phoneme_with_emp+="pau "
                            phoneme_with_empMark+="pau "

                        phoneme_with_emp+=pyopenjtalk.g2p(te)+" "
                        phoneme_with_empMark+=pyopenjtalk.g2p(te)+" "

                        if te[-1]=="、" or te[-1]=="・":
                            phoneme_with_emp+="pau "
                            phoneme_with_empMark+="pau "

                        phoneme_with_empMark+="＊ "
                        
                            
                    phoneme_with_emp=phoneme_with_emp[:-1]
                    phoneme_with_empMark=phoneme_with_empMark[:-3]

                    #音素を変換
                    ph1=""
                    ph2=""
                    ph3=""
                    for phoneme in phoneme_no_emp:
                        ph1+=openjtalk2julius(phoneme)
                    for phoneme in phoneme_with_emp:
                        ph2+=openjtalk2julius(phoneme)
                    for phoneme in phoneme_with_empMark:
                        ph3+=openjtalk2julius(phoneme)

            
                    if ph1!=ph2:
                        print(text)
                        print(text_sequence)
                        print("phoneme_no_emp",ph1)
                        print("phoneme_with_emp",ph2)
                    with open('phoneme/JECS/Normal/' + filename + '.lab', mode='w') as f:
                        f.write(ph1.strip('\n'))
                    with open('phoneme/JECS/Emp/' + filename + '.lab', mode='w') as f:
                        f.write(ph3.strip('\n'))