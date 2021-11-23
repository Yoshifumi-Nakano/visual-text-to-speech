import os
import random
import json

import tgt
import librosa
import numpy as np
import pyworld as pw
import cv2
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from utils.transform import Phoneme2Kana_ver2
from utils.getimage import get_text_images
<<<<<<< HEAD
import audio as Audio


=======


import audio as Audio




>>>>>>> parent of cffca65c... git rm -r cached .
class Preprocessor:
    def __init__(self, config):
        self.config = config
        self.in_dir = config["path"]["raw_path"]
        self.out_dir = config["path"]["preprocessed_path"]
        self.val_size = config["preprocessing"]["val_size"]
        self.test_size = config["preprocessing"]["test_size"]
        self.sampling_rate = config["preprocessing"]["audio"]["sampling_rate"]
        self.hop_length = config["preprocessing"]["stft"]["hop_length"]
        assert config["preprocessing"]["pitch"]["feature"] in [
            "phoneme_level",
            "frame_level",
        ]
        assert config["preprocessing"]["energy"]["feature"] in [
            "phoneme_level",
            "frame_level",
        ]

        #pitch info
        self.pitch_phoneme_averaging = (
            config["preprocessing"]["pitch"]["feature"] == "phoneme_level"
        )
<<<<<<< HEAD
=======
        self.pitch_kana_averaging=config["preprocessing"]["pitch"]["image"]
>>>>>>> parent of cffca65c... git rm -r cached .
        self.pitch_normalization = config["preprocessing"]["pitch"]["normalization"]

        #energy info
        self.energy_phoneme_averaging = (
            config["preprocessing"]["energy"]["feature"] == "phoneme_level"
        )
<<<<<<< HEAD
=======
        self.energy_kana_averaging=config["preprocessing"]["energy"]["image"]
>>>>>>> parent of cffca65c... git rm -r cached .
        self.energy_normalization = config["preprocessing"]["energy"]["normalization"]

        #STFT info
        self.STFT = Audio.stft.TacotronSTFT(
            config["preprocessing"]["stft"]["filter_length"],
            config["preprocessing"]["stft"]["hop_length"],
            config["preprocessing"]["stft"]["win_length"],
            config["preprocessing"]["mel"]["n_mel_channels"],
            config["preprocessing"]["audio"]["sampling_rate"],
            config["preprocessing"]["mel"]["mel_fmin"],
            config["preprocessing"]["mel"]["mel_fmax"],
        )

        #image info
        self.image_preprocess_width=config["preprocessing"]["image"]["width"]
        self.image_preprocess_height=config["preprocessing"]["image"]["height"]
        self.image_preprocess_fontsize=config["preprocessing"]["image"]["font_size"]


    def build_from_path(self):
        #make folder
        os.makedirs((os.path.join(self.out_dir, "mel")), exist_ok=True)
        os.makedirs((os.path.join(self.out_dir, "pitch")), exist_ok=True)
        os.makedirs((os.path.join(self.out_dir, "energy")), exist_ok=True)
        os.makedirs((os.path.join(self.out_dir, "duration")), exist_ok=True)
<<<<<<< HEAD
        os.makedirs((os.path.join(self.out_dir,"image")),exist_ok=True)
=======
        os.makedirs((os.path.join(self.out_dir, "pitch_kana")), exist_ok=True)
        os.makedirs((os.path.join(self.out_dir, "energy_kana")), exist_ok=True)
        os.makedirs((os.path.join(self.out_dir, "duration_kana")), exist_ok=True)
        os.makedirs((os.path.join(self.out_dir,"text_kana")),exist_ok=True)
        os.makedirs((os.path.join(self.out_dir,"image_kana")),exist_ok=True)
>>>>>>> parent of cffca65c... git rm -r cached .

        #normalization module
        pitch_scaler = StandardScaler()
        energy_scaler = StandardScaler()
<<<<<<< HEAD

        # Compute pitch, energy, duration, and mel-spectrogram
        speakers = {}
        out=[]
=======
        pitch_kana_scaler=StandardScaler()
        energy_kana_scaler=StandardScaler()

        # Compute pitch, energy, duration, and mel-spectrogram
        speakers = {}
        out={"BASIC5000":[],"other":[]}
>>>>>>> parent of cffca65c... git rm -r cached .
        n_frames = 0
        for i, speaker in enumerate(tqdm(os.listdir(self.in_dir))):            #in_dir : ./raw_data/JSUT
            speakers[speaker] = i
            for wav_name in os.listdir(os.path.join(self.in_dir, speaker)):
                if ".wav" not in wav_name:
                    continue
                basename = wav_name.split(".")[0]
                tg_path = os.path.join(
                    self.out_dir, "TextGrid", speaker, "{}.TextGrid".format(basename)    #tg_path: ./preprocessed_data/JSUT/TextGrid/JSUT/BASIC5000_0001.TextGrid
                )
                if os.path.exists(tg_path):
                    ret = self.process_utterance(speaker, basename)

                    #return build from path
                    if ret is None:
                        continue
                    else:
<<<<<<< HEAD
                        info, pitch, energy, n = ret

                    #for test dataset
                    out.append(info)
                else:
                    print(tg_path)
                    continue
=======
                        info, pitch, energy, n,pitch_kana,energy_kana = ret

                    #for test dataset
                    if basename[:9]=="BASIC5000":
                        out["BASIC5000"].append(info)
                    else:
                        out["other"].append(info)

                else:
                    raise ValueError(tg_path)
>>>>>>> parent of cffca65c... git rm -r cached .

                if len(pitch) > 0:
                    pitch_scaler.partial_fit(pitch.reshape((-1, 1)))
                if len(energy) > 0:
                    energy_scaler.partial_fit(energy.reshape((-1, 1)))
<<<<<<< HEAD
=======
                if len(pitch_kana)>0:
                    pitch_kana_scaler.partial_fit(pitch_kana.reshape((-1, 1)))
                if len(energy_kana)>0:
                    energy_kana_scaler.partial_fit(energy_kana.reshape((-1, 1)))
>>>>>>> parent of cffca65c... git rm -r cached .

                n_frames += n

        # Perform normalization if necessary
        if self.pitch_normalization:
            pitch_mean = pitch_scaler.mean_[0]
<<<<<<< HEAD
            pitch_std = pitch_scaler.scale_[0]
        else:
            pitch_mean = 0
            pitch_std = 1
            
        if self.energy_normalization:
            energy_mean = energy_scaler.mean_[0]
            energy_std = energy_scaler.scale_[0]
        else:
            energy_mean = 0
            energy_std = 1
=======
            pitch_kana_mean = pitch_kana_scaler.mean_[0]

            pitch_std = pitch_scaler.scale_[0]
            pitch_kana_std = pitch_kana_scaler.scale_[0]
        else:
            pitch_mean = 0
            pitch_std = 1
            pitch_kana_mean=0
            pitch_kana_std=1
            
        if self.energy_normalization:
            energy_mean = energy_scaler.mean_[0]
            energy_kana_mean = energy_kana_scaler.mean_[0]

            energy_std = energy_scaler.scale_[0]
            energy_kana_std = energy_kana_scaler.scale_[0]
        else:
            energy_mean = 0
            energy_kana_mean = 0
            energy_std = 1
            energy_kana_std=1
>>>>>>> parent of cffca65c... git rm -r cached .

        pitch_min, pitch_max = self.normalize(
            os.path.join(self.out_dir, "pitch"), pitch_mean, pitch_std
        )
        energy_min, energy_max = self.normalize(
            os.path.join(self.out_dir, "energy"), energy_mean, energy_std
        )

<<<<<<< HEAD
=======
        pitch_kana_min,pitch_kana_max=self.normalize(
            os.path.join(self.out_dir, "pitch_kana"), pitch_kana_mean, pitch_kana_std
        )
        energy_kana_min, energy_kana_max = self.normalize(
            os.path.join(self.out_dir, "energy_kana"), energy_kana_mean, energy_kana_std
        )

>>>>>>> parent of cffca65c... git rm -r cached .
        # Save files
        with open(os.path.join(self.out_dir, "speakers.json"), "w") as f:
            f.write(json.dumps(speakers))

        with open(os.path.join(self.out_dir, "stats.json"), "w") as f:
            stats = {
                "pitch": [
                    float(pitch_min),
                    float(pitch_max),
                    float(pitch_mean),
                    float(pitch_std),
<<<<<<< HEAD
=======
                    float(pitch_kana_min),
                    float(pitch_kana_max),
                    float(pitch_kana_mean),
                    float(pitch_kana_std),
>>>>>>> parent of cffca65c... git rm -r cached .
                ],
                "energy": [
                    float(energy_min),
                    float(energy_max),
                    float(energy_mean),
                    float(energy_std),
<<<<<<< HEAD
=======
                    float(energy_kana_min),
                    float(energy_kana_max),
                    float(energy_kana_mean),
                    float(energy_kana_std),
>>>>>>> parent of cffca65c... git rm -r cached .
                ],
            }
            f.write(json.dumps(stats))

        print(
            "Total time: {} hours".format(
                n_frames * self.hop_length / self.sampling_rate / 3600
            )
        )

        #shufflr test data and other data
<<<<<<< HEAD
        random.shuffle(out)
        out = [r for r in out if r is not None]

        # write metadata
        with open(os.path.join(self.out_dir, "val.txt"), "w", encoding="utf-8") as f:
            for m in out[: self.val_size]:
                f.write(m + "\n")

        with open(os.path.join(self.out_dir, "test.txt"), "w", encoding="utf-8") as f:
            for m in out[self.val_size:self.val_size+self.test_size]:
                f.write(m + "\n")

        with open(os.path.join(self.out_dir, "train.txt"), "w", encoding="utf-8") as f:
            for m in out[self.val_size+self.test_size:]:
                f.write(m + "\n")
=======
        random.shuffle(out["BASIC5000"])
        out_test=out["BASIC5000"][:self.test_size]
        out=out["BASIC5000"][self.test_size:]+out["other"]
        random.shuffle(out)

        out = [r for r in out if r is not None]
        out_test = [r for r in out_test if r is not None]

        # write metadata
        with open(os.path.join(self.out_dir, "train.txt"), "w", encoding="utf-8") as f:
            for m in out[self.val_size :]:
                f.write(m + "\n")
        with open(os.path.join(self.out_dir, "val.txt"), "w", encoding="utf-8") as f:
            for m in out[: self.val_size]:
                f.write(m + "\n")
        with open(os.path.join(self.out_dir, "test.txt"), "w", encoding="utf-8") as f:
            for m in out_test:
                f.write(m + "\n")

        out=out_test+out
>>>>>>> parent of cffca65c... git rm -r cached .

        return out

    def process_utterance(self, speaker, basename):
        #path info
        wav_path = os.path.join(self.in_dir, speaker, "{}.wav".format(basename))
        text_path = os.path.join(self.in_dir, speaker, "{}.lab".format(basename))
        tg_path = os.path.join(
            self.out_dir, "TextGrid", speaker, "{}.TextGrid".format(basename)
        ) 

        # get alignments
        textgrid = tgt.io.read_textgrid(tg_path)
<<<<<<< HEAD
        phone, duration, start, end = self.get_alignment(
            textgrid.get_tier_by_name("phones")
        )
        text = "{" + " ".join(phone) + "}"        #text {m i z u o m a r e e sh i a k a r a k a w a n a k u t e w a n a r a n a i n o d e s u}

        # validation
=======
        phone, duration, start, end,kanas,duration_kana = self.get_alignment(
            textgrid.get_tier_by_name("phones")
        )
        text = "{" + " ".join(phone) + "}"        #text {m i z u o m a r e e sh i a k a r a k a w a n a k u t e w a n a r a n a i n o d e s u}
        kanas = "{" + " ".join(kanas) + "}"


>>>>>>> parent of cffca65c... git rm -r cached .
        if start >= end:
            return None

        # Read and trim wav files
        wav, _ = librosa.load(wav_path,sr=22050)
        wav = wav[
            int(self.sampling_rate * start) : int(self.sampling_rate * end)
        ].astype(np.float32) 

        # Read raw text
        with open(text_path, "r") as f:
            raw_text = f.readline().strip("\n")

        # compute pitch
        pitch, t = pw.dio(
            wav.astype(np.float64),
            self.sampling_rate,
            frame_period=self.hop_length / self.sampling_rate * 1000,     #ホップサイズが「160」とすると（ズラす量のことで、サンプリング周波数が「16000」の場合、時間換算で「10ms」）frame_periodはSTFTが何秒間か表す
        )                                                                       
        pitch = pw.stonemask(wav.astype(np.float64), pitch, t, self.sampling_rate)
        pitch = pitch[: sum(duration)]
        if np.sum(pitch != 0) <= 1:
            return None

        # compute mel-scale spectrogram and energy
        mel_spectrogram, energy = Audio.tools.get_mel_from_wav(wav, self.STFT)
        mel_spectrogram = mel_spectrogram[:, : sum(duration)]
        energy = energy[: sum(duration)]

        # perform linear interpolation
        nonzero_ids = np.where(pitch != 0)[0]
        interp_fn = interp1d(
            nonzero_ids,
            pitch[nonzero_ids],
            fill_value=(pitch[nonzero_ids[0]], pitch[nonzero_ids[-1]]),
            bounds_error=False,
        )
        pitch = interp_fn(np.arange(0, len(pitch)))

        # average pitch by input 
<<<<<<< HEAD
=======
        if self.pitch_kana_averaging:
            pos = 0
            pitch_kana=[0]*len(duration_kana)

            for i, d in enumerate(duration_kana):
                if d > 0:
                    pitch_kana[i] = np.mean(pitch[pos : pos + d])
                else:
                    pitch_kana[i] = 0
                pos += d
            pitch_kana = pitch_kana[: len(duration_kana)]
>>>>>>> parent of cffca65c... git rm -r cached .
        if self.pitch_phoneme_averaging:
            pos = 0
            for i, d in enumerate(duration):
                if d > 0:
                    pitch[i] = np.mean(pitch[pos : pos + d])
                else:
                    pitch[i] = 0
                pos += d
            pitch = pitch[: len(duration)]

         # average energy by input 
<<<<<<< HEAD
=======
        if self.energy_kana_averaging:
            pos = 0
            energy_kana=[0]*len(duration_kana)
            for i, d in enumerate(duration_kana):
                if d > 0:
                    energy_kana[i] = np.mean(energy[pos : pos + d])
                else:
                    energy_kana[i] = 0
                pos += d
            energy_kana = energy_kana[: len(duration_kana)]
>>>>>>> parent of cffca65c... git rm -r cached .
        if self.energy_phoneme_averaging:
            pos = 0
            for i, d in enumerate(duration):
                if d > 0:
                    energy[i] = np.mean(energy[pos : pos + d])
                else:
                    energy[i] = 0
                pos += d
            energy = energy[: len(duration)]

        # save duration
        dur_filename = "{}-duration-{}.npy".format(speaker, basename)
        np.save(os.path.join(self.out_dir, "duration", dur_filename), duration)

<<<<<<< HEAD
        # save pitch
        pitch_filename = "{}-pitch-{}.npy".format(speaker, basename)
        np.save(os.path.join(self.out_dir, "pitch", pitch_filename), pitch)
=======
        dur_kana_filename = "{}-duration-kana-{}.npy".format(speaker, basename)
        np.save(os.path.join(self.out_dir, "duration_kana", dur_kana_filename), duration_kana)

        # save pitch
        pitch_filename = "{}-pitch-{}.npy".format(speaker, basename)
        np.save(os.path.join(self.out_dir, "pitch", pitch_filename), pitch)
        pitch_kana_filename="{}-pitch-kana-{}.npy".format(speaker, basename)
        np.save(os.path.join(self.out_dir, "pitch_kana", pitch_kana_filename), pitch_kana)
>>>>>>> parent of cffca65c... git rm -r cached .

        # save duration
        energy_filename = "{}-energy-{}.npy".format(speaker, basename)
        np.save(os.path.join(self.out_dir, "energy", energy_filename), energy)
<<<<<<< HEAD
=======
        energy_kana_filename = "{}-energy-kana-{}.npy".format(speaker, basename)
        np.save(os.path.join(self.out_dir, "energy_kana", energy_kana_filename), energy_kana)
>>>>>>> parent of cffca65c... git rm -r cached .

        # save mel
        mel_filename = "{}-mel-{}.npy".format(speaker, basename)
        np.save(
            os.path.join(self.out_dir, "mel", mel_filename),
            mel_spectrogram.T,
        )

<<<<<<< HEAD
        #save kana image
        iamge_filename="{}-image-{}-{}-{}-{}.jpg".format(speaker, str(self.image_preprocess_width),str(self.image_preprocess_height),str(self.image_preprocess_fontsize),basename)
        text_image=get_text_images(texts=[t for t in text.replace("{", "").replace("}", "").split()],width=self.image_preprocess_width,height=self.image_preprocess_height,font_size=self.image_preprocess_fontsize)
        cv2.imwrite(os.path.join(self.out_dir,"image",iamge_filename),text_image)
=======
        #save kana transcript
        kana_filename="{}_{}.lab".format(speaker, basename)
        with open(os.path.join(self.out_dir,"text_kana",kana_filename), mode='w') as f:
            f.write(kanas)

        #save kana image
        iamge_filename="{}-image-{}-{}-{}-{}.jpg".format(speaker, str(self.image_preprocess_width),str(self.image_preprocess_height),str(self.image_preprocess_fontsize),basename)
        text_image=get_text_images(texts=[t for t in kanas.replace("{", "").replace("}", "").split()],width=self.image_preprocess_width,height=self.image_preprocess_height,font_size=self.image_preprocess_fontsize)
        cv2.imwrite(os.path.join(self.out_dir,"image_kana",iamge_filename),text_image)
>>>>>>> parent of cffca65c... git rm -r cached .
        
        return (
            "|".join([basename, speaker, text, raw_text]),
            self.remove_outlier(pitch),
            self.remove_outlier(energy),
<<<<<<< HEAD
            mel_spectrogram.shape[1]
        )

    def get_alignment(self, tier):
        sil_phones = ["sil", "sp", "spn", 'silB', 'silE','.',"?"," ","","!"]
=======
            mel_spectrogram.shape[1],
            self.remove_outlier(pitch_kana),
            self.remove_outlier(energy_kana),
        )

    def get_alignment(self, tier):
        sil_phones = ["sil", "sp", "spn", 'silB', 'silE', '']
>>>>>>> parent of cffca65c... git rm -r cached .

        phones = []
        durations = []
        start_time = 0
        end_time = 0
        end_idx = 0
        for t in tier._objects:
            s, e, p = t.start_time, t.end_time, t.text
<<<<<<< HEAD
=======

>>>>>>> parent of cffca65c... git rm -r cached .
            # find phone and start_time that are not silent
            if phones == []:
                if p in sil_phones:
                    continue
                else:
                    start_time = s

            # ordinary phones → phones , silent phones → sp
            if p not in sil_phones:
                phones.append(p)
                end_time = e
                end_idx = len(phones)
            else:
                phones.append('sp')
            
            # add duration
            durations.append(
                int(
                    np.round(e * self.sampling_rate / self.hop_length)
                    - np.round(s * self.sampling_rate / self.hop_length)
                )
            )

        # delete last sp
        phones = phones[:end_idx]
        durations = durations[:end_idx]
        
<<<<<<< HEAD
        # test code
        assert len(phones) == len(durations)

        return phones, durations, start_time, end_time
=======
        # transform from phone duration to image duration 
        kanas,durations_kana=Phoneme2Kana_ver2(phones,durations) 

        # test code
        assert len(phones) == len(durations)
        assert len(kanas) == len(durations_kana)
        assert sum(durations) == sum(durations_kana)

        return phones, durations, start_time, end_time,kanas,durations_kana
>>>>>>> parent of cffca65c... git rm -r cached .

    def remove_outlier(self, values):
        values = np.array(values)
        p25 = np.percentile(values, 25)
        p75 = np.percentile(values, 75)
        lower = p25 - 1.5 * (p75 - p25)
        upper = p75 + 1.5 * (p75 - p25)
        normal_indices = np.logical_and(values > lower, values < upper)

        return values[normal_indices]

    def normalize(self, in_dir, mean, std):
        max_value = np.finfo(np.float64).min
        min_value = np.finfo(np.float64).max
        for filename in os.listdir(in_dir):
            filename = os.path.join(in_dir, filename)
            values = (np.load(filename) - mean) / std
            np.save(filename, values)

            max_value = max(max_value, max(values))
            min_value = min(min_value, min(values))

<<<<<<< HEAD
        return min_value, max_value
=======
        return min_value, max_value
>>>>>>> parent of cffca65c... git rm -r cached .
