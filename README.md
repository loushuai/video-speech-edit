
# Video Speech Edit

AI-powered video speech editing using MuseTalk and Chatterbox.

## Prerequisites

- Python 3.10
- CUDA-compatible GPU (recommended)
- Git

## Installation


### 1. Setup FFmpeg

Download and install FFmpeg:
- **Windows/Linux**: Download from [FFmpeg Builds](https://github.com/BtbN/FFmpeg-Builds/releases)
- **macOS**: `brew install ffmpeg`

### 2. Install MuseTalk

Clone MuseTalk:

```bash
git clone https://github.com/TMElyralab/MuseTalk.git
cd MuseTalk
```

Download model weights:

```bash
sh ./download_weights.sh
wget https://huggingface.co/TMElyralab/MuseTalk/resolve/main/musetalkV15/musetalk.json?download=true -O models/musetalkV15/musetalk.json
wget https://huggingface.co/stabilityai/sd-vae-ft-mse/resolve/main/config.json?download=true -O models/sd-vae/config.json
wget https://huggingface.co/openai/whisper-tiny/resolve/main/config.json?download=true -O ./models/whisper/config.json
```

Install dependencies:

```bash
conda create -n MuseTalk python==3.10
conda activate MuseTalk
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
pip install --no-cache-dir -U openmim
mim install mmengine
mim install "mmcv==2.0.1"
mim install "mmdet==3.1.0"
pip install --no-build-isolation mmpose==1.1.0
```

### 3. Install Chatterbox

Clone chatterbox

```bash
cd ..
git clone https://github.com/resemble-ai/chatterbox.git
cd chatterbox
```

Install

```
conda create -n chatterbox python==3.10
conda activate chatterbox
pip install -e .
pip install langid
```

### 4. Install other dependencies

```bash
cd ..
conda create -n video-speech-edit python=3.10
conda activate video-speech-edit
pip install -r requirements.txt
```

## Start Streamlit

```bash
streamlit run ui.py
```

You can access the UI by navigating to port 8501 on your localhost or your server’s address.


## Troubleshooting

- **Gradio version conflicts**: Remove gradio version specification in `pyproject.toml`
- **CUDA issues**: Ensure CUDA 11.8 is installed and compatible with your GPU
- **Memory errors**: Use a GPU with at least 8GB VRAM for optimal performance