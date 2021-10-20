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
from utils.transform import Phoneme2Kana 
from utils.getimage import get_text_images


import audio as Audio




class Preprocessor:
    def __init__(self, config):
        self.config = config
        self.in_dir = config["path"]["raw_path"]
        self.out_dir = config["path"]["preprocessed_path"]
        self.val_size = config["preprocessing"]["val_size"]
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
        self.pitch_phoneme_averaging = (
            config["preprocessing"]["pitch"]["feature"] == "phoneme_level"
        )

        self.pitch_kana_averaging=config["preprocessing"]["pitch"]["image"]

        self.energy_phoneme_averaging = (
            config["preprocessing"]["energy"]["feature"] == "phoneme_level"
        )
        
        self.energy_kana_averaging=config["preprocessing"]["energy"]["image"]


        self.pitch_normalization = config["preprocessing"]["pitch"]["normalization"]
        self.energy_normalization = config["preprocessing"]["energy"]["normalization"]

        self.STFT = Audio.stft.TacotronSTFT(
            config["preprocessing"]["stft"]["filter_length"],
            config["preprocessing"]["stft"]["hop_length"],
            config["preprocessing"]["stft"]["win_length"],
            config["preprocessing"]["mel"]["n_mel_channels"],
            config["preprocessing"]["audio"]["sampling_rate"],
            config["preprocessing"]["mel"]["mel_fmin"],
            config["preprocessing"]["mel"]["mel_fmax"],
        )

        self.image_preprocess_width=config["preprocessing"]["image"]["width"]
        self.image_preprocess_height=config["preprocessing"]["image"]["height"]
        self.image_preprocess_fontsize=config["preprocessing"]["image"]["font_size"]


    def build_from_path(self):
        os.makedirs((os.path.join(self.out_dir, "mel")), exist_ok=True)
        os.makedirs((os.path.join(self.out_dir, "pitch")), exist_ok=True)
        os.makedirs((os.path.join(self.out_dir, "energy")), exist_ok=True)
        os.makedirs((os.path.join(self.out_dir, "duration")), exist_ok=True)
        os.makedirs((os.path.join(self.out_dir, "pitch_kana")), exist_ok=True)
        os.makedirs((os.path.join(self.out_dir, "energy_kana")), exist_ok=True)
        os.makedirs((os.path.join(self.out_dir, "duration_kana")), exist_ok=True)
        os.makedirs((os.path.join(self.out_dir,"text_kana")),exist_ok=True)
        os.makedirs((os.path.join(self.out_dir,"image_kana")),exist_ok=True)

        print("Processing Data ...")
        out = list()
        n_frames = 0
        pitch_scaler = StandardScaler()
        energy_scaler = StandardScaler()

        pitch_kana_scaler=StandardScaler()
        energy_kana_scaler=StandardScaler()

        # Compute pitch, energy, duration, and mel-spectrogram
        speakers = {}
        #in_dir : ./raw_data/JSUT
        for i, speaker in enumerate(tqdm(os.listdir(self.in_dir))):
            speakers[speaker] = i
            #speaker JUST
            for wav_name in os.listdir(os.path.join(self.in_dir, speaker)):
                if ".wav" not in wav_name:
                    continue

                basename = wav_name.split(".")[0]
                tg_path = os.path.join(
                    self.out_dir, "TextGrid", speaker, "{}.TextGrid".format(basename)
                )
                #tg_path: ./preprocessed_data/JSUT/TextGrid/JSUT/BASIC5000_0001.TextGrid
                if os.path.exists(tg_path):
                    ret = self.process_utterance(speaker, basename)
                    if ret is None:
                        continue
                    else:
                        info, pitch, energy, n,pitch_kana,energy_kana = ret
                    out.append(info)
                else:
                    raise ValueError(tg_path)

                if len(pitch) > 0:
                    pitch_scaler.partial_fit(pitch.reshape((-1, 1)))
                if len(energy) > 0:
                    energy_scaler.partial_fit(energy.reshape((-1, 1)))
                if len(pitch_kana)>0:
                    pitch_kana_scaler.partial_fit(pitch_kana.reshape((-1, 1)))
                if len(energy_kana)>0:
                    energy_kana_scaler.partial_fit(energy_kana.reshape((-1, 1)))

                n_frames += n

        print("Computing statistic quantities ...")
        # Perform normalization if necessary
        if self.pitch_normalization:
            pitch_mean = pitch_scaler.mean_[0]
            pitch_kana_mean = pitch_kana_scaler.mean_[0]

            pitch_std = pitch_scaler.scale_[0]
            pitch_kana_std = pitch_kana_scaler.scale_[0]
        else:
            # A numerical trick to avoid normalization...
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


        pitch_min, pitch_max = self.normalize(
            os.path.join(self.out_dir, "pitch"), pitch_mean, pitch_std
        )
        energy_min, energy_max = self.normalize(
            os.path.join(self.out_dir, "energy"), energy_mean, energy_std
        )

        pitch_kana_min,pitch_kana_max=self.normalize(
            os.path.join(self.out_dir, "pitch_kana"), pitch_kana_mean, pitch_kana_std
        )

        energy_kana_min, energy_kana_max = self.normalize(
            os.path.join(self.out_dir, "energy_kana"), energy_kana_mean, energy_kana_std
        )

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
                    float(pitch_kana_min),
                    float(pitch_kana_max),
                    float(pitch_kana_mean),
                    float(pitch_kana_std),
                ],
                "energy": [
                    float(energy_min),
                    float(energy_max),
                    float(energy_mean),
                    float(energy_std),
                    float(energy_kana_min),
                    float(energy_kana_max),
                    float(energy_kana_mean),
                    float(energy_kana_std),
                ],
            }
            f.write(json.dumps(stats))

        print(
            "Total time: {} hours".format(
                n_frames * self.hop_length / self.sampling_rate / 3600
            )
        )

        random.shuffle(out)
        out = [r for r in out if r is not None]

        # Write metadata
        with open(os.path.join(self.out_dir, "train.txt"), "w", encoding="utf-8") as f:
            for m in out[self.val_size :]:
                f.write(m + "\n")
        with open(os.path.join(self.out_dir, "val.txt"), "w", encoding="utf-8") as f:
            for m in out[: self.val_size]:
                f.write(m + "\n")

        return out

    def process_utterance(self, speaker, basename):
        #speaker, basename,JUST BASIC0001
        wav_path = os.path.join(self.in_dir, speaker, "{}.wav".format(basename))
        text_path = os.path.join(self.in_dir, speaker, "{}.lab".format(basename))
        tg_path = os.path.join(
            self.out_dir, "TextGrid", speaker, "{}.TextGrid".format(basename)
        )    #tg_path ./preprocessed_data/JSUT/TextGrid/JSUT/BASIC5000_0001.TextGrid
        

        # Get alignments
        textgrid = tgt.io.read_textgrid(tg_path)
        phone, duration, start, end,kanas,duration_kana = self.get_alignment(
            textgrid.get_tier_by_name("phones")
        )
        text = "{" + " ".join(phone) + "}" #text {m i z u o m a r e e sh i a k a r a k a w a n a k u t e w a n a r a n a i n o d e s u}
        kanas = "{" + " ".join(kanas) + "}"


        if start >= end:
            return None

        # Read and trim wav files
        wav, _ = librosa.load(wav_path,sr=22050)
        wav = wav[
            int(self.sampling_rate * start) : int(self.sampling_rate * end)
        ].astype(np.float32) #音がある区間のwavファイルを取り出している

        # Read raw text
        with open(text_path, "r") as f:
            raw_text = f.readline().strip("\n")

        # Compute fundamental frequency
        pitch, t = pw.dio(
            wav.astype(np.float64),
            self.sampling_rate,
            frame_period=self.hop_length / self.sampling_rate * 1000,
        )#ホップサイズが「160」とすると（ズラす量、サンプリング周波数が「16000」の場合、時間換算で「10ms」）frame_periodはSTFTが何秒間か表す
        pitch = pw.stonemask(wav.astype(np.float64), pitch, t, self.sampling_rate)

        pitch = pitch[: sum(duration)]
        if np.sum(pitch != 0) <= 1:
            return None

        # Compute mel-scale spectrogram and energy
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

        if self.pitch_kana_averaging:
            #カナ文字ごとの平均
            pos = 0
            pitch_kana=[0]*len(duration_kana)

            for i, d in enumerate(duration_kana):
                if d > 0:
                    pitch_kana[i] = np.mean(pitch[pos : pos + d])
                else:
                    pitch_kana[i] = 0
                pos += d
            pitch_kana = pitch_kana[: len(duration_kana)]


        if self.pitch_phoneme_averaging:
            # Phoneme-level average
            pos = 0
            for i, d in enumerate(duration):
                if d > 0:
                    pitch[i] = np.mean(pitch[pos : pos + d])
                else:
                    pitch[i] = 0
                pos += d
            pitch = pitch[: len(duration)]
        
        if self.energy_kana_averaging:
            # Phoneme-level average
            pos = 0
            energy_kana=[0]*len(duration_kana)
            for i, d in enumerate(duration_kana):
                if d > 0:
                    energy_kana[i] = np.mean(energy[pos : pos + d])
                else:
                    energy_kana[i] = 0
                pos += d
            energy_kana = energy_kana[: len(duration_kana)]

        if self.energy_phoneme_averaging:
            # Phoneme-level average
            pos = 0
            for i, d in enumerate(duration):
                if d > 0:
                    energy[i] = np.mean(energy[pos : pos + d])
                else:
                    energy[i] = 0
                pos += d
            energy = energy[: len(duration)]

        

        # Save files(duration,energy,pitch)
        dur_filename = "{}-duration-{}.npy".format(speaker, basename)
        np.save(os.path.join(self.out_dir, "duration", dur_filename), duration)

        dur_kana_filename = "{}-duration-kana-{}.npy".format(speaker, basename)
        np.save(os.path.join(self.out_dir, "duration_kana", dur_kana_filename), duration_kana)

        pitch_filename = "{}-pitch-{}.npy".format(speaker, basename)
        np.save(os.path.join(self.out_dir, "pitch", pitch_filename), pitch)
        if self.pitch_kana_averaging:
            pitch_kana_filename="{}-pitch-kana-{}.npy".format(speaker, basename)
            np.save(os.path.join(self.out_dir, "pitch_kana", pitch_kana_filename), pitch_kana)


        energy_filename = "{}-energy-{}.npy".format(speaker, basename)
        np.save(os.path.join(self.out_dir, "energy", energy_filename), energy)
        if self.energy_kana_averaging:
            energy_kana_filename = "{}-energy-kana-{}.npy".format(speaker, basename)
            np.save(os.path.join(self.out_dir, "energy_kana", energy_kana_filename), energy_kana)
        

        mel_filename = "{}-mel-{}.npy".format(speaker, basename)
        np.save(
            os.path.join(self.out_dir, "mel", mel_filename),
            mel_spectrogram.T,
        )
        

        #save kana transcript
        kana_filename="{}_{}.lab".format(speaker, basename)
        with open(os.path.join(self.out_dir,"text_kana",kana_filename), mode='w') as f:
            f.write(kanas)

        #save kana image
        iamge_filename="{}-image-{}-{}-{}-{}.jpg".format(speaker, str(self.image_preprocess_width),str(self.image_preprocess_height),str(self.image_preprocess_fontsize),basename)
        text_image=get_text_images(texts=[t for t in kanas.replace("{", "").replace("}", "").split()],width=self.image_preprocess_width,height=self.image_preprocess_height,font_size=self.image_preprocess_fontsize)
        cv2.imwrite(os.path.join(self.out_dir,"image_kana",iamge_filename),text_image)
        

        

        return (
            "|".join([basename, speaker, text, raw_text]),
            self.remove_outlier(pitch),
            self.remove_outlier(energy),
            mel_spectrogram.shape[1],
            self.remove_outlier(pitch_kana),
            self.remove_outlier(energy_kana),
        )

    def get_alignment(self, tier):
        sil_phones = ["sil", "sp", "spn", 'silB', 'silE', '']

        phones = []
        durations = []
        start_time = 0
        end_time = 0
        end_idx = 0
        for t in tier._objects:
            s, e, p = t.start_time, t.end_time, t.text
            #0.3125 0.3525 m

            # Trim leading silences
            if phones == []:
                if p in sil_phones:
                    continue
                else:
                    start_time = s

            if p not in sil_phones:
                # For ordinary phones
                phones.append(p)
                end_time = e
                end_idx = len(phones)
            else:
                # For silent phones
                phones.append('sp')
            #リストで何ブロック目から何ブロック目かを表す
            durations.append(
                int(
                    np.round(e * self.sampling_rate / self.hop_length)
                    - np.round(s * self.sampling_rate / self.hop_length)
                )
            )

        # Trim tailing silences
        phones = phones[:end_idx]
        durations = durations[:end_idx]
        
        kanas,durations_kana=Phoneme2Kana(phones,durations) 

        assert len(phones) == len(durations)
        assert len(kanas) == len(durations_kana)
        assert sum(durations) == sum(durations_kana)
        return phones, durations, start_time, end_time,kanas,durations_kana

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

        return min_value, max_value
