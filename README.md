# 🎤 Bahasa Rojak Whisper Fine-tuning Preprocessing Pipeline
A comprehensive, multimodal pipeline for preprocessing YouTube audio and video data (Bahasa Rojak) for fine-tuning speech models. This tool handles advanced audio denoising, Voice Activity Detection (VAD), dual-ASR transcription, and visual subtitle extraction via OCR.

## 📋 Overview
This pipeline processes raw YouTube streams (.webm, .mkv, .m4a) into highly accurate, Whisper-ready training data by combining audio transcription with hardcoded visual subtitles:

- Audio Pass (pipeline): Splits into 10-minute chunks, denoises audio via DeepFilterNet, splits human speech segments using Silero VAD, and generates baseline transcripts using both Whisper Large and Qwen3-ASR (1.7B).

- Visual Pass (ocr_sweep): Bypasses broken YouTube .webm indexes by instantly rebuilding them into .mkv via FFmpeg, then sweeps the bottom 50% of the video using PaddleOCR to extract hardcoded subtitles, filtering out visual hallucinations.

- Integration: Merges the visual subtitle data with the exact audio segments to provide high-quality ground truth for manual correction.

## 📁 Folder Structure
Input Structure
```
Youtube/
├── channel_1/
│   ├── audio1.wav
│   ├── audio2.wav
│   ├── audio2.mp3
│   └── video/
│       └── video1.opus
│       └── video2.mp4
│       └── video2.mkv
├── channel_2/
│   └── ...
│   └── video/
│       └── ...
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

#### Clone repository (if applicable)
```
git clone <repository-url>
cd whisper-bahasa-rojak
```
#### Install Python dependencies
```
pip install -r requirements.txt
```
#### Install system dependencies (Ubuntu/Debian)
```
sudo apt-get update
sudo apt-get install ffmpeg
```

## 📊 Pipeline Steps 
### Step 1: Preprocess all audio
```
# Run the main audio pipeline
python test_tool.py --tool pipeline --input "Youtube/channel/video/video_name.webm"
```
What happens:

- Extracts audio and applies DeepFilterNet noise reduction in 10-minute chunks

- Detects natural pauses in speech using Silero VAD

- Splits audio into exact speaking segments and saves them as separate WAV files (16kHz mono)

- Generates initial ASR guesses using both Whisper Large and Qwen3-ASR

- Saves the baseline data into a structured transcript.json file

### Step 2: Visual subtitle extraction
```
# Run the OCR sweep to extract hardcoded subtitles
python test_tool.py --tool ocr_sweep --input "Youtube/channel/video/video_name.webm" --sample_rate 2.0
```
What happens:

- Safely rebuilds the video index via FFmpeg to prevent premature crashes on YouTube .webm files

- Sweeps the bottom 50% of the video frames every 2 seconds to locate hardcoded text

- Uses PaddleOCR to extract text while filtering out visual hallucinations (confidence > 0.75)

- Automatically maps the extracted visual text back to the exact audio segments in your transcript.json



### Step 3: Manual correction and error classification
After generating initial transcriptions, you can use the integrated correction tool to:

1. Manually correct transcriptions by listening to audio segments and editing the gold‑standard text.

2. Classify errors in the Whisper and Qwen model outputs by highlighting mismatched spans and assigning one of eight predefined error categories.
```
# Launch the correction interface
python correction_tool.py
```
## 🔧 Tools Details
### 📝 1. The Extraction Tool (test_tool.py)

test_tool.py is the core engine for generating your segmented audio and transcript data. It runs in two distinct phases:

#### Step 1: Audio Pass (Denoise, Segment, & Transcribe)
This step extracts the audio, cleans it, slices it into exact speaking segments using VAD, and runs it through both Whisper and Qwen.

```
python test_tool.py --tool pipeline --input "Youtube/channel/video/video_name.webm"
```
*Output: Creates a folder containing .wav chunks and a transcript.json with initial ASR guesses.*

#### Step 2: Visual OCR Pass (Extract Hardcoded Subtitles)
This step sweeps the video frames to read hardcoded subtitles (commonly found in live streams) and maps them back to the audio chunks.

```
python test_tool.py --tool ocr_sweep --input "Youtube/channel/video/video_name.webm" --sample_rate 2.0
```
*Output: Updates transcript.json with a new text field containing the exact visual subtitles that appeared during that audio segment.*

### 📝 2. The Legacy Preprocessor (preprocessor.py)
(Note: Many of these features have been superseded by test_tool.py, but are retained for legacy batch processing).

```
# Run batch conversions or legacy splitting
python preprocessor.py --step convert    # Converts audio to 16kHz WAV
python preprocessor.py --step split      # Splits audio (Legacy method)
python preprocessor.py --step metadata   # Generates HuggingFace-ready metadata.jsonl
```

### 📝 3. Annotation GUI (Waveform Corrector)
This GUI provides an interactive waveform display and audio playback for precise transcription editing.

- Waveform – click‑drag to select a region for playback or editing.

- Playback controls – play the full segment or just the selected region.

- Edit operations – split the segment at the selection, delete the selection, or delete the entire segment.

- Gold‑standard text – an editable text box where you type/correct the transcription.

- Reference transcriptions – read‑only boxes showing the Whisper and Qwen outputs for reference.

- Verification checkbox – mark the segment as verified once it is correctly transcribed.

All changes are saved to the transcript.json file in the video’s output folder.

### ❌ 4. Error Classification GUI
This interface helps you systematically label errors in the model transcriptions. It stores annotations directly in the transcript.json (under error_annotations and error_counts).

- Three text boxes – Gold (editable? no, but selectable), Whisper, and Qwen.

- Paired annotation –

a. Select a span in the Gold box.
b. Select the corresponding span in either Whisper or Qwen (order can be swapped).
c. Click one of the eight category buttons.
Both selected spans are highlighted with the same background colour (unique per category).
- Hover effect – move your mouse over any highlighted span to make its paired span glow yellow, visually linking the two parts of the error.

- Counts per category – a statistics panel shows how many errors of each type have been annotated for the current segment.

- Remove error – click inside any highlighted span and press Remove Error Under Cursor to delete the entire error pair and adjust the counts.

The eight error categories are those defined in your report (Spelling, Language/Translate, Named Entity, Substitution, Particle, Merge, Deletion, Hallucination). The counts and annotations persist across sessions, enabling you to review and refine the error labels at any time.

## Output Structure (Updated)
The transcript.json file now includes two additional fields per segment:

- "error_counts": a dictionary mapping each category to the number of errors annotated in that segment.

- "error_annotations": a list of annotation objects, each containing the model type, character indices of the span, the corresponding gold span indices, and the category.

These augmentations enable later analysis without affecting the original annotation GUI.



