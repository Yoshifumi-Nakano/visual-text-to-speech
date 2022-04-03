## visual-text-to-speech
This is a PyTorch implementation of [**visual-text to speech (vTTS)**](http://arxiv.org/abs/2203.14725).

## Set up envirnment
Set up environment
```
git clone https://github.com/Yoshifumi-Nakano/visual-text-to-speech.git
git submodule update --init
unzip hifigan/generator_universal.pth.tar.zip -d hifigan/
pip install -r requirements.txt
```

## Prepare Dataset
make folder for dataset and copy LJspeech (sound and trascript) to that folder.
```
mkdir -p raw_data/LJSpeech/LJSpeech
cp your_ljspeech_path/*.wav  raw_data/LJSpeech/LJSpeech/
python retrieve_transcript_LJSpeech.py
```

## Preprocessing
prepare acoustic features and visual text for training
```
python preprocess.py config/LJSpeech/preprocess.yaml
```

## Training
```
python train.py -p config/LJSpeech/preprocess.yaml -m config/LJSpeech/model.yaml -t config/LJSpeech/train.yaml
```

## Test
```
python test.py --source ./preprocessed_data/LJSpeech/test.txt --speaker_id * --restore_step *** -p config/LJSpeech/preprocess.yaml -m config/LJSpeech/model.yaml -t config/LJSpeech/train.yaml
```
