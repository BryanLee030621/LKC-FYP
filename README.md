# 🎤 Bahasa Rojak Whisper Fine-tuning Preprocessing Pipeline
A comprehensive pipeline for preprocessing YouTube audio data (Bahasa Rojak) for fine-tuning OpenAI's Whisper model. This tool handles audio conversion, segmentation, noise reduction, and transcription preparation.

## 📋 Overview
This pipeline processes raw YouTube audio files (m4a, opus) into Whisper-ready training data:

Converts various audio formats to WAV (16kHz mono)

Splits audio into ≤30-second segments at natural pauses

Applies optional noise reduction for background music/sounds

Generates transcription templates for manual correction

Organizes output in a structured format for easy management

## 📁 Folder Structure
Input Structure
```
Youtube/
├── channel_1/
│   ├── video1.m4a
│   ├── video2.opus
│   └── archive.txt
├── channel_2/
│   └── ...
└── ...
```
Output Structure
```
Youtube/preprocess/
├── channel_1/
│   ├── video1_title/
│   │   ├── segment_001.wav
│   │   ├── segment_002.wav
│   │   ├── ...
│   │   └── transcript.json
│   └── video2_title/
├── channel_2/
├── metadata.jsonl
└── processing_log.txt
```
## 🚀 Quick Start
### 1. Installation
bash
#### Clone repository (if applicable)
git clone <repository-url>
cd whisper-bahasa-rojak

#### Install Python dependencies
pip install -r requirements.txt

#### Install system dependencies (Ubuntu/Debian)
sudo apt-get update
sudo apt-get install ffmpeg

### 2. Basic Usage
```
# Run the complete preprocessing pipeline
python preprocessor.py

# Or run step by step:
python preprocessor.py --step convert    # Only convert audio
python preprocessor.py --step split      # Only split into segments
python preprocessor.py --step metadata   # Only generate metadata
```
## 🔧 Detailed Setup
### Requirements
Create requirements.txt:

```
librosa==0.10.1
soundfile==0.12.1
pydub==0.25.1
numpy==1.24.3
openai-whisper==20231117
noisereduce==2.0.1
tqdm==4.66.1
Install all dependencies:
```
```
pip install -r requirements.txt
```

## 📊 Pipeline Steps
### Step 1: Preprocess all audio
```
# Run the main preprocessor
python preprocessor.py
```
What happens:

Converts all m4a/opus files to WAV format (16kHz mono)

Detects natural pauses in speech

Splits audio into segments ≤30 seconds

Applies noise reduction to segments with background noise

Saves each segment as separate WAV file

### Step 2: Generate initial transcriptions (run on Colab with GPU)
```
# Generate empty transcription templates
python whisper_transcribe.py
```

### Step 3: Manual correction
```
# Launch the correction interface
python correction_tool.py
```
