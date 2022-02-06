import csv
import os

check=[]
with open("LJSpeech/ljspeech/ljspeech/LJSpeech-1.1/metadata.csv", mode='r') as f:
    reader = csv.reader(f,delimiter='\t')
    reader=[r for r in reader]
    for r in reader:
        filename=r[0].split("|")[0]
        text=r[0].split("|")[1]
        
        flg=any(t.isdigit() for t in text)

        if flg:
            os.remove("raw_data/LJSpeech/LJSpeech/"+filename+".wav")
            check.append(filename)
        else:
            with open('raw_data/LJSpeech/LJSpeech/' + filename + '.lab', mode='w') as f:
                f.write(text.strip('\n'))

print(check)