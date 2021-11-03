import glob
import re
import os
# %%
transcript_files = glob.glob("/app/JECS/data/neutral/transcript_org.txt")
print(transcript_files)

# %%
if not os.path.exists("raw_data/JSUT/JSUT"):
    os.makedirs("raw_data/JSUT/JSUT")

for transcript in transcript_files:
    with open(transcript, mode='r') as f:
        lines = f.readlines()
    for line in lines:
        #空行は飛ばす
        if line!='\n':
            filename, text = line.split(':')
            if "JE_EMPH" in filename and "JA" in filename:
                text=text.strip(" ")
                with open('raw_data/JECS/JECS/' + filename + '.lab', mode='w') as f:
                    f.write(text.strip('\n'))
