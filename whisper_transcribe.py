# whisper_transcribe.py
import whisper
import json
from pathlib import Path

def transcribe_video(video_dir):
    """Transcribe all segments in a video directory"""
    print(f"\n===== Processing folder: {video_dir} =====")
    
    model = whisper.load_model("tiny")  # or "large" for better accuracy
    
    transcript_path = video_dir / "transcript.json"
    if not transcript_path.exists():
        print(f"❌ No transcript.json found in {video_dir}, skipping folder.")
        return
    with open(transcript_path, 'r') as f:
        data = json.load(f)
    
    for segment in data["segments"]:
        if "text" in segment and segment["text"].strip():
            print(f"Skipping {segment['segment']} (already transcribed)")
            continue

        audio_path = video_dir / segment["segment"]
        
        # Transcribe with language hint (Malay/English mix)
        result = model.transcribe(
            str(audio_path),
            fp16=False  # Set to True if using GPU
        )
        
        segment["text"] = result["text"]
        print(f"Transcribed {segment['segment']}: {result['text'][:50]}...")
    
    # Save updated transcript
    with open(transcript_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"===== Finished folder: {video_dir} =====\n")

# Process all videos
for video_dir in Path("Youtube/preprocess").glob("*/*/"):
    transcribe_video(video_dir)