import json
import math
import os
import cv2
import random
import numpy as np
from torch.utils.data import Dataset

from utils.symbols import get_symbols
from utils.tools import pad_1D, pad_2D,pad_2D_gray_image

class Dataset(Dataset):
    def __init__(
        self, filename, preprocess_config, train_config, sort=False, drop_last=False
    ):
        #basic info
        self.dataset_name = preprocess_config["dataset"]
        self.preprocessed_path = preprocess_config["path"]["preprocessed_path"]
        self.batch_size = train_config["optimizer"]["batch_size"]
        self.symbol_to_id = get_symbols()
        self.sort = sort
        self.drop_last = drop_last

        #image information
        self.image_preprocess_width=preprocess_config["preprocessing"]["image"]["width"]
        self.image_preprocess_height=preprocess_config["preprocessing"]["image"]["height"]
        self.image_preprocess_fontsize=preprocess_config["preprocessing"]["image"]["font_size"]
        self.image_preprocess_stride=preprocess_config["preprocessing"]["image"]["stride"]

        #filename equals to "train.txt" in train phase
        self.basename, self.speaker, self.text, self.raw_text = self.process_meta(
            filename
        )

        #speaker id function
        with open(os.path.join(self.preprocessed_path, "speakers.json")) as f:
            self.speaker_map = json.load(f)

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        #basic info
        basename = self.basename[idx]           
        speaker = self.speaker[idx]             
        speaker_id = self.speaker_map[speaker]  
        raw_text = self.raw_text[idx]           
        text = np.array([self.symbol_to_id[t] for t in self.text[idx].replace("{", "").replace("}", "").split()])  
        
        #load mel
        mel_path = os.path.join(
            self.preprocessed_path,
            "mel",
            "{}-mel-{}.npy".format(speaker, basename),
        )
        mel = np.load(mel_path)

        #load pitch 
        pitch_path = os.path.join(
            self.preprocessed_path,
            "pitch",
            "{}-pitch-{}.npy".format(speaker, basename),
        )
        pitch = np.load(pitch_path)

        #load energy
        energy_path = os.path.join(
            self.preprocessed_path,
            "energy",
            "{}-energy-{}.npy".format(speaker, basename),
        )
        energy = np.load(energy_path)

        #load duration
        duration_path = os.path.join(
            self.preprocessed_path,
            "duration",
            "{}-duration-{}.npy".format(speaker, basename),
        )
        duration = np.load(duration_path)

        #load image
        image_path= os.path.join(
            self.preprocessed_path,
            "image",
            "{}-image-{}-{}-{}-{}.jpg".format(speaker, str(self.image_preprocess_width),str(self.image_preprocess_height),str(self.image_preprocess_fontsize),basename)
        )
        image=cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)

        sample = {
            "id": basename,
            "speaker": speaker_id,
            "text": text,
            "raw_text": raw_text,
            "mel": mel,
            "pitch": pitch,
            "energy": energy,
            "duration": duration,
            "image":image
        }

        return sample

    def process_meta(self, filename):
        with open(
            os.path.join(self.preprocessed_path, filename), "r", encoding="utf-8"
        ) as f:
            name = []
            speaker = []
            text = []
            raw_text = []
            for line in f.readlines():
                n, s, t, r = line.strip("\n").split("|")
                name.append(n)
                speaker.append(s)
                text.append(t)
                raw_text.append(r)
            
            return name, speaker, text, raw_text
    

    def reprocess(self, data, idxs):
        ids = [data[idx]["id"] for idx in idxs]                 
        speakers = [data[idx]["speaker"] for idx in idxs]       
        texts = [data[idx]["text"] for idx in idxs]             
        raw_texts = [data[idx]["raw_text"] for idx in idxs]

        mels = [data[idx]["mel"] for idx in idxs]
        pitches = [data[idx]["pitch"] for idx in idxs]
        energies = [data[idx]["energy"] for idx in idxs]
        durations = [data[idx]["duration"] for idx in idxs]

        images=[data[idx]["image"] for idx in idxs]
        images=pad_2D_gray_image(images)

        text_lens = np.array([text.shape[0] for text in texts])
        mel_lens = np.array([mel.shape[0] for mel in mels])
        speakers = np.array(speakers)
        texts = pad_1D(texts)
        mels = pad_2D(mels)
        pitches = pad_1D(pitches)
        energies = pad_1D(energies)
        durations = pad_1D(durations)
        
        response=[ids,raw_texts,speakers,texts,text_lens,max(text_lens),mels,mel_lens,max(mel_lens),pitches,energies,durations,images]
        
        return tuple(response)

    def collate_fn(self, data):
        data_size = len(data)

        if self.sort:
            len_arr = np.array([d["text"].shape[0] for d in data])
            idx_arr = np.argsort(-len_arr)
        else:
            idx_arr = np.arange(data_size)

        tail = idx_arr[len(idx_arr) - (len(idx_arr) % self.batch_size) :]
        idx_arr = idx_arr[: len(idx_arr) - (len(idx_arr) % self.batch_size)]
        idx_arr = idx_arr.reshape((-1, self.batch_size)).tolist()
        if not self.drop_last and len(tail) > 0:
            idx_arr += [tail.tolist()]

        output = list()
        for idx in idx_arr:
            output.append(self.reprocess(data, idx))

        return output


class TestDataset(Dataset):
    def __init__(self, filepath, preprocess_config):
        #path
        self.preprocessed_path = preprocess_config["path"]["preprocessed_path"]

        #get basic info of test data
        self.basename, self.speaker, self.text, self.raw_text = self.process_meta(
            filepath
        )

        #image info
        self.image_preprocess_width=preprocess_config["preprocessing"]["image"]["width"]
        self.image_preprocess_height=preprocess_config["preprocessing"]["image"]["height"]
        self.image_preprocess_fontsize=preprocess_config["preprocessing"]["image"]["font_size"]

        #speakers info
        with open(
            os.path.join(
                preprocess_config["path"]["preprocessed_path"], "speakers.json"
            )
        ) as f:
            self.speaker_map = json.load(f)

        #test batch
        self.data_num=len(self.basename)
        self.symbol_to_id = get_symbols()
        self.batchs=self.get_batch()

    def get_batch(self):
        batchs=[]
        for idx in range(self.data_num):
            #id
            ids = [self.basename[idx]]

            #speaker
            speaker = self.speaker[idx]
            speaker_id = np.array([self.speaker_map[speaker]])
            
            #raw_text
            raw_texts = [self.raw_text[idx]]
            
            #texts
            texts = np.array([[self.symbol_to_id[t] for t in self.text[idx].replace("{", "").replace("}", "").split()]])


            #text lens
            text_lens = np.array([len(texts[0])])

            #image
            image_path= os.path.join(
                self.preprocessed_path,
                "image",
                "{}-image-{}-{}-{}-{}.jpg".format(speaker, str(self.image_preprocess_width),str(self.image_preprocess_height),str(self.image_preprocess_fontsize),self.basename[idx])
            )
            image=[cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)]

            batchs.append((ids, raw_texts, speaker_id, texts, text_lens, max(text_lens),None,None,None,None,None,None,image))

        return batchs

    def process_meta(self, filename):
        with open(filename, "r", encoding="utf-8") as f:
            name = []
            speaker = []
            text = []
            raw_text = []
            for line in f.readlines():
                n, s, t, r = line.strip("\n").split("|")
                name.append(n)
                speaker.append(s)
                text.append(t)
                raw_text.append(r)
            return name, speaker, text, raw_text
