import json
import math
import os
import cv2

import numpy as np
from torch.utils.data import Dataset

from text import symbols, text_to_sequence
from utils.tools import pad_1D, pad_2D,pad_2D_gray_image
from utils.preimage import pre_seq

class Dataset(Dataset):
    def __init__(
        self, filename, preprocess_config, train_config, sort=False, drop_last=False
    ):
        #basic info
        self.dataset_name = preprocess_config["dataset"]
        self.preprocessed_path = preprocess_config["path"]["preprocessed_path"]
        self.cleaners = preprocess_config["preprocessing"]["text"]["text_cleaners"]
        self.batch_size = train_config["optimizer"]["batch_size"]
        self.use_image = preprocess_config["preprocessing"]["image"]["use_image"]
        self.symbol_to_id = {s: i for i, s in enumerate(symbols)}
        self.use_accent = preprocess_config["preprocessing"]["accent"]["use_accent"]
        self.accent_to_id = {'0':0, '[':1, ']':2, '#':3}
        self.sort = sort
        self.drop_last = drop_last

        #image information
        self.image_preprocess_width=preprocess_config["preprocessing"]["image"]["width"]
        self.image_preprocess_height=preprocess_config["preprocessing"]["image"]["height"]
        self.image_preprocess_fontsize=preprocess_config["preprocessing"]["image"]["font_size"]

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
        basename = self.basename[idx]           #BASIC5000_0147
        speaker = self.speaker[idx]             #JSUT 
        speaker_id = self.speaker_map[speaker]  #0
        raw_text = self.raw_text[idx]           #狼が犬に似ているように、おべっか使いは友達のように見える。
        phone = np.array([self.symbol_to_id[t] for t in self.text[idx].replace("{", "").replace("}", "").split()])  #self.text {o o k a m i g a i n u n i n i t e i r u y o o n i sp o b e q k a z u k a i w a t o m o d a ch i n o y o o n i m i e r u}
        
        #accent
        if self.use_accent:
            with open(os.path.join(self.preprocessed_path, "accent",basename+ '.accent')) as f:
                accent = f.read()
            accent = [self.accent_to_id[t] for t in accent]
            accent = np.array(accent[:len(phone)])

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

        #load image info
        if self.use_image:
            #load kana transcript
            text_kana_filename="{}_{}.lab".format(speaker, basename)
            with open(os.path.join(self.preprocessed_path, "text_kana",text_kana_filename), "r", encoding="utf-8") as f:
                f=f.read()
                text_kana=np.array([t for t in f.replace("{", "").replace("}", "").split()])
            
            #load pitch kana
            pitch_path = os.path.join(
                self.preprocessed_path,
                "pitch_kana",
                "{}-pitch-kana-{}.npy".format(speaker, basename),
            )
            pitch = np.load(pitch_path)

            #load energy kana
            energy_path = os.path.join(
                self.preprocessed_path,
                "energy_kana",
                "{}-energy-kana-{}.npy".format(speaker, basename),
            )
            energy = np.load(energy_path)

            #load duration kana
            duration_path = os.path.join(
                self.preprocessed_path,
                "duration_kana",
                "{}-duration-kana-{}.npy".format(speaker, basename),
            )
            duration = np.load(duration_path)

            #load image
            image_path= os.path.join(
                self.preprocessed_path,
                "image_kana",
                "{}-image-{}-{}-{}-{}.jpg".format(speaker, str(self.image_preprocess_width),str(self.image_preprocess_height),str(self.image_preprocess_fontsize),basename)
            )
            image=cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)

            

            
        sample = {
            "id": basename,
            "speaker": speaker_id,
            "text": phone,
            "raw_text": raw_text,
            "mel": mel,
            "pitch": pitch,
            "energy": energy,
            "duration": duration,
        }
        if self.use_accent:
            sample["accent"] = accent

        if self.use_image:
            sample["text"]=text_kana
            sample["image"]=image


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
        ids = [data[idx]["id"] for idx in idxs]                 #['BASIC5000_3888', 'BASIC5000_4148', 'ONOMAçTOPEE300_195', 'BASIC5000_4226', 'ONOMATOPEE300_093', 'UT-PARAPHRASE-sent102-phrase2', 'REPEAT500_set4_081', 'ONOMATOPEE300_021', 'BASIC5000_3045', 'ONOMATOPEE300_098', 'BASIC5000_4268', 'BASIC5000_1393', 'BASIC5000_4852', 'UT-PARAPHRASE-sent105-phrase1', 'BASIC5000_3057', 'BASIC5000_0776']
        speakers = [data[idx]["speaker"] for idx in idxs]       #[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        texts = [data[idx]["text"] for idx in idxs]             #[音素のsymbolのindexの配列]×BatchSize
        raw_texts = [data[idx]["raw_text"] for idx in idxs]     #['最近は、辞令もこわごわ出さざるを得ない。']×BatchSize

        mels = [data[idx]["mel"] for idx in idxs]
        pitches = [data[idx]["pitch"] for idx in idxs]
        energies = [data[idx]["energy"] for idx in idxs]
        durations = [data[idx]["duration"] for idx in idxs]

        if self.use_accent:
            accents = [data[idx]["accent"] for idx in idxs]
            accents = pad_1D(accents)
        if self.use_image:
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
        
        response=[ids,raw_texts,speakers,texts,text_lens,max(text_lens),mels,mel_lens,max(mel_lens),pitches,energies,durations]
        
        if self.use_accent:
            response.append(accents)
        else:
            response.append(None)
        if self.use_image:
            response.append(images)
        else:
            response.append(None)

        return tuple(response)

    def collate_fn(self, data):
        #data_size=batch_size×GroupSize
        data_size = len(data)

        #sort
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

        #output
        output = list()

        #length idx == batch_size
        for idx in idx_arr:
            output.append(self.reprocess(data, idx))

        return output


class TextDataset(Dataset):
    def __init__(self, filepath, preprocess_config):
        self.cleaners = preprocess_config["preprocessing"]["text"]["text_cleaners"]

        self.basename, self.speaker, self.text, self.raw_text = self.process_meta(
            filepath
        )
        with open(
            os.path.join(
                preprocess_config["path"]["preprocessed_path"], "speakers.json"
            )
        ) as f:
            self.speaker_map = json.load(f)

        self.use_accent = preprocess_config["preprocessing"]["accent"]["use_accent"]
        self.accent_to_id = {'0':0, '[':1, ']':2, '#':3}

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        basename = self.basename[idx]
        speaker = self.speaker[idx]
        speaker_id = self.speaker_map[speaker]
        raw_text = self.raw_text[idx]
        phone = np.array(text_to_sequence(self.text[idx], self.cleaners))
        accent = None
        if self.use_accent:
            with open(os.path.join(self.preprocessed_path, "accent",basename+ '.accent')) as f:
                accent = f.read()
            accent = [self.accent_to_id[t] for t in accent]
            accent = np.array(accent[:len(phone)])

        return (basename, speaker_id, phone, raw_text,accent)

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

    def collate_fn(self, data):
        ids = [d[0] for d in data]
        speakers = np.array([d[1] for d in data])
        texts = [d[2] for d in data]
        raw_texts = [d[3] for d in data]
        text_lens = np.array([text.shape[0] for text in texts])
        if self.use_accent:
            accents = [d[4] for d in data]

        texts = pad_1D(texts)
        accents = pad_1D(accents)

        return ids, raw_texts, speakers, texts, text_lens, max(text_lens), accents




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

        #image
        self.use_image = preprocess_config["preprocessing"]["image"]["use_image"]

        #test batch
        self.data_num=len(self.basename)
        self.batchs=self.get_batch()

        self.symbol_to_id = {s: i for i, s in enumerate(symbols)}


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
            text_kana_filename="{}_{}.lab".format(speaker, self.basename[idx])
            with open(os.path.join(self.preprocessed_path, "text_kana",text_kana_filename), "r", encoding="utf-8") as f:
                f=f.read()
                text_kana=np.array([t for t in f.replace("{", "").replace("}", "").split()])
            texts = np.array([text_kana])


            #text lens
            text_lens = np.array([len(texts[0])])

            #image
            image_path= os.path.join(
                self.preprocessed_path,
                "image_kana",
                "{}-image-{}-{}-{}-{}.jpg".format(speaker, str(self.image_preprocess_width),str(self.image_preprocess_height),str(self.image_preprocess_fontsize),self.basename[idx])
            )
            image=[cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)]

            batchs.append((ids, raw_texts, speaker_id, texts, text_lens, max(text_lens),None,None,None,None,None,None,None,image))

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



if __name__ == "__main__":
    # Test
    import torch
    import yaml
    from torch.utils.data import DataLoader
    from utils.utils import to_device

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    preprocess_config = yaml.load(
        open("./config/LJSpeech/preprocess.yaml", "r"), Loader=yaml.FullLoader
    )
    train_config = yaml.load(
        open("./config/LJSpeech/train.yaml", "r"), Loader=yaml.FullLoader
    )

    train_dataset = Dataset(
        "train.txt", preprocess_config, train_config, sort=True, drop_last=True
    )
    val_dataset = Dataset(
        "val.txt", preprocess_config, train_config, sort=False, drop_last=False
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config["optimizer"]["batch_size"] * 4,
        shuffle=True,
        collate_fn=train_dataset.collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_config["optimizer"]["batch_size"],
        shuffle=False,
        collate_fn=val_dataset.collate_fn,
    )

    n_batch = 0
    for batchs in train_loader:
        for batch in batchs:
            to_device(batch, device)
            n_batch += 1
    print(
        "Training set  with size {} is composed of {} batches.".format(
            len(train_dataset), n_batch
        )
    )

    n_batch = 0
    for batchs in val_loader:
        for batch in batchs:
            to_device(batch, device)
            n_batch += 1
    print(
        "Validation set  with size {} is composed of {} batches.".format(
            len(val_dataset), n_batch
        )
    )
