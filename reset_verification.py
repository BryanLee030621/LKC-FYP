# reset_verification_status.py
import json
from pathlib import Path

def reset_verification():
    base_dir = Path("Youtube/preprocess")
    
    for transcript_path in base_dir.rglob("transcript.json"):
        try:
            with open(transcript_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # Remove any existing verified flags
            for seg in data.get("segments", []):
                seg.pop("verified", None)
            
            # Save back
            with open(transcript_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            print(f"Reset: {transcript_path}")
            
        except Exception as e:
            print(f"Error with {transcript_path}: {e}")

if __name__ == "__main__":
    reset_verification()