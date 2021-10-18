import glob
import pandas as pd
import os
# %%
transcript_files = glob.glob("/home/sarulab/yoshifumi_nakano/FastSpeech2/FastSpeech2-JSUT/jsut_ver1.1/*/transcript_utf8.txt")
print(transcript_files)

if not os.path.exists("raw_data/JSUT/JSUT"):
    os.makedirs("raw_data/JSUT/JSUT")
for transcript in transcript_files:
    print(transcript)
    # with open(transcript, mode='r') as f:
    #     lines = f.readlines()
    # for line in lines:
        
    #     filename, text = line.split(':')
    #     with open('' + filename + '.lab', mode='w') as f:
    #         f.write(text.strip('\n'))