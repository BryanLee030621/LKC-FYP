# whisper_transcribe.py
import whisper
import json
from pathlib import Path

def transcribe_video(video_dir):
    """Transcribe all segments in a video directory"""
    model = whisper.load_model("medium")  # or "large" for better accuracy
    
    transcript_path = video_dir / "transcript.json"
    with open(transcript_path, 'r') as f:
        data = json.load(f)
    
    for segment in data["segments"]:
        audio_path = video_dir / segment["segment"]
        
        # Transcribe with language hint (Malay/English mix)
        result = model.transcribe(
            str(audio_path),
            language="ms",  # Malay as base, but Whisper will detect code-switching
            fp16=False  # Set to True if using GPU
        )
        
        segment["text"] = result["text"]
        print(f"Transcribed {segment['segment']}: {result['text'][:50]}...")
    
    # Save updated transcript
    with open(transcript_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

# Process all videos
for video_dir in Path("Youtube/preprocess").glob("*/*/"):
    transcribe_video(video_dir)