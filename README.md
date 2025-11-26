
### Create Environment
```
conda create -n video-speech-edit python=3.10 -y
conda activate video-speech-edit
```


### Setup FFmpeg
1. [Download](https://github.com/BtbN/FFmpeg-Builds/releases) the ffmpeg-static package
2. Configure FFmpeg based on your operating system


### Clone MuseTalk
```
git clone https://github.com/TMElyralab/MuseTalk.git
cd MuseTalk
```

### Download MuseTalk weights
```
sh ./download_weights.sh
wget https://huggingface.co/TMElyralab/MuseTalk/resolve/main/musetalkV15/musetalk.json?download=true -O models/musetalkV15/musetalk.json
wget https://huggingface.co/stabilityai/sd-vae-ft-mse/resolve/main/config.json?download=true -O models/sd-vae/config.json
wget https://huggingface.co/openai/whisper-tiny/resolve/main/config.json?download=true -O ./models/whisper/config.json
```

### Install MuseTalk
```
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
pip install --no-cache-dir -U openmim
mim install mmengine
mim install "mmcv==2.0.1"
mim install "mmdet==3.1.0"
pip install --no-build-isolation mmpose==1.1.0
```

### Clone and Install chatterbox
```
cd ..
git clone https://github.com/resemble-ai/chatterbox.git
cd chatterbox
pip install -e .
```

If you encounter version conflict issue of gradio, please try to remove gradio version in `pyproject.toml`