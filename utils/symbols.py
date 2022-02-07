import os

def get_symbols():
    filenames=["train.txt","val.txt","test.txt"]
    text_seq=[]
    for filename in filenames:
        with open(os.path.join("./preprocessed_data/LJSpeech", filename),"r",encoding="utf-8") as f:
            for line in f.readlines():
                _,__,text,___=line.strip("\n").split("|")
                text_seq+=[t for t in text.replace("{", "").replace("}", "").split()]
    text_seq=list(set(text_seq))
    text_seq.sort()

    symbol_to_id = {s: i+1 for i, s in enumerate(text_seq)}
    return symbol_to_id
