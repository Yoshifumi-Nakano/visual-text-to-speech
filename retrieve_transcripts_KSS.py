import glob
import pandas as pd
import os
# %%
transcript_files = "/app/KSS/KSS/transcript.v.1.4.txt"
# %%
with open(transcript_files, mode='r') as f:
    lines = f.readlines()
for line in lines:
    filename, text = line.split('|')[0],line.split('|')[1]
    print(filename, text)
    with open('raw_data/KSS/KSS/' + filename[2:8] + '.lab', mode='w') as f:
        f.write(text.strip('\n'))