# move LJSpeech transcription to raw_data/LJSpeech/LJSpeech
# remove data which has number 
import csv
import os
import shutil

# change transcript_file_path and wav_dir_path into your ljspeech path
transcript_file_path = "./ljspeech/LJSpeech-1.1/metadata.csv"
wav_dir_path = "./ljspeech/LJSpeech-1.1/wavs"

with open(transcript_file_path, mode='r') as f:
    reader = csv.reader(f,delimiter='\t')
    reader = [r for r in reader]

    if not os.path.exists("./raw_data/LJSpeech/LJSpeech/"):
        os.makedirs("./raw_data/LJSpeech/LJSpeech/")

    for r in reader:
        filename=r[0].split("|")[0]
        text=r[0].split("|")[1]

        number=any(t.isdigit() for t in text)

        if not number:
            shutil.move(os.path.join(wav_dir_path,filename+".wav"),"raw_data/LJSpeech/LJSpeech/")
            with open(os.path.join("raw_data/LJSpeech/LJSpeech/" + filename + ".lab"), mode='w') as f:
                f.write(text.strip('\n'))
        else:
            print(filename)
