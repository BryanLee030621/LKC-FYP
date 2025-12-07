import os
import json
import subprocess
from pathlib import Path
import librosa
import soundfile as sf
from pydub import AudioSegment, silence
import numpy as np

class YouTubePreprocessor:
    def __init__(self, root_dir="Youtube"):
        self.root_dir = Path(root_dir)
        self.output_dir = self.root_dir / "preprocess"
        self.output_dir.mkdir(exist_ok=True)
        
    def convert_to_wav(self, input_path, output_path):
        """Convert any audio format to WAV 16kHz mono"""
        cmd = [
            "ffmpeg", "-i", str(input_path),
            "-ar", "16000", "-ac", "1",
            "-acodec", "pcm_s16le",
            "-y", str(output_path)  # -y to overwrite
        ]
        subprocess.run(cmd, capture_output=True)
        
    def split_at_silence(self, audio_path, max_duration=30, min_silence_len=500, silence_thresh=-40):
        """
        Split audio at natural pauses (silence)
        Returns list of (start_ms, end_ms) segments
        """
        audio = AudioSegment.from_wav(audio_path)
        
        # Detect non-silent chunks
        chunks = silence.detect_nonsilent(
            audio,
            min_silence_len=min_silence_len,
            silence_thresh=silence_thresh
        )
        
        # Merge chunks into segments ≤ max_duration
        segments = []
        current_start, current_end = chunks[0]
        
        for start, end in chunks[1:]:
            if (end - current_start) / 1000 <= max_duration:
                current_end = end
            else:
                # Segment too long, save current and start new
                segments.append((current_start, current_end))
                current_start, current_end = start, end
        
        # Add last segment
        segments.append((current_start, current_end))
        
        # Final pass: split any segment > max_duration
        final_segments = []
        for start, end in segments:
            duration = (end - start) / 1000
            
            if duration <= max_duration:
                final_segments.append((start, end))
            else:
                # Split evenly but try to find silence points
                num_splits = int(np.ceil(duration / max_duration))
                split_duration = duration / num_splits
                
                for i in range(num_splits):
                    seg_start = start + int(i * split_duration * 1000)
                    seg_end = start + int((i + 1) * split_duration * 1000)
                    final_segments.append((seg_start, min(seg_end, end)))
        
        return final_segments
    
    def reduce_noise(self, audio_array, sr):
        """Simple noise reduction using spectral gating"""
        import noisereduce as nr
        return nr.reduce_noise(
            y=audio_array,
            sr=sr,
            stationary=False,
            prop_decrease=0.75  # Reduce noise by 75%
        )
    
    def preprocess_video(self, channel, video_file):
        """Process a single video file"""
        # Clean video name (remove extension and special chars)
        video_name = Path(video_file).stem
        video_name_clean = "".join(c for c in video_name if c.isalnum() or c in " _-")
        
        # Create output directory
        video_output_dir = self.output_dir / channel / video_name_clean
        video_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Convert to WAV
        input_path = self.root_dir / channel / video_file
        temp_wav = video_output_dir / "temp_full.wav"
        self.convert_to_wav(input_path, temp_wav)
        
        # Split into segments
        segments = self.split_at_silence(temp_wav)
        
        # Process each segment
        transcript_data = []
        
        for i, (start_ms, end_ms) in enumerate(segments, 1):
            # Load the segment
            audio = AudioSegment.from_wav(temp_wav)
            segment = audio[start_ms:end_ms]
            
            # Export segment
            segment_filename = f"segment_{i:03d}.wav"
            segment_path = video_output_dir / segment_filename
            segment.export(segment_path, format="wav")
            
            # Optional: Apply noise reduction
            y, sr = librosa.load(segment_path, sr=16000)
            if self.has_background_noise(y, sr):
                y_clean = self.reduce_noise(y, sr)
                sf.write(segment_path, y_clean, sr)
            
            # Add to transcript data
            duration = (end_ms - start_ms) / 1000.0
            transcript_data.append({
                "segment": segment_filename,
                "start_time": start_ms / 1000.0,
                "end_time": end_ms / 1000.0,
                "duration": duration,
                "text": "",  # To be filled by Whisper
                "audio_path": str(segment_path.relative_to(self.root_dir))
            })
            
            # Print progress
            print(f"  Created {segment_filename} ({duration:.1f}s)")
        
        # Remove temp file
        temp_wav.unlink()
        
        # Save transcript JSON
        transcript_path = video_output_dir / "transcript.json"
        with open(transcript_path, 'w', encoding='utf-8') as f:
            json.dump({
                "video_name": video_name,
                "channel": channel,
                "segments": transcript_data,
                "total_duration": sum(s["duration"] for s in transcript_data)
            }, f, indent=2, ensure_ascii=False)
        
        return len(segments)
    
    def has_background_noise(self, audio, sr, threshold=0.02):
        """Simple check for background noise"""
        energy = np.mean(audio**2)
        return energy < threshold
    
    def run(self):
        """Main processing loop"""
        total_segments = 0
        total_videos = 0
        
        # Process each channel folder
        for channel_dir in self.root_dir.iterdir():
            if not channel_dir.is_dir() or channel_dir.name == "preprocess":
                continue
            
            channel = channel_dir.name
            print(f"\nProcessing channel: {channel}")
            
            # Process each audio file in channel
            for audio_file in channel_dir.glob("*"):
                if audio_file.suffix.lower() in ['.m4a', '.opus', '.mp3', '.mp4', '.mkv']:
                    print(f"\nProcessing: {audio_file.name}")
                    segments = self.preprocess_video(channel, audio_file.name)
                    total_segments += segments
                    total_videos += 1
        
        # Create master metadata file
        self.create_master_metadata()
        
        print(f"\n✅ Preprocessing complete!")
        print(f"   Processed {total_videos} videos")
        print(f"   Created {total_segments} segments")
        print(f"   Output saved to: {self.output_dir}")
    
    def create_master_metadata(self):
        """Create a master JSONL file for all videos"""
        master_data = []
        
        for channel_dir in self.output_dir.iterdir():
            if not channel_dir.is_dir():
                continue
            
            for video_dir in channel_dir.iterdir():
                if not video_dir.is_dir():
                    continue
                
                transcript_file = video_dir / "transcript.json"
                if transcript_file.exists():
                    with open(transcript_file, 'r') as f:
                        video_data = json.load(f)
                    
                    # Add each segment to master data
                    for segment in video_data["segments"]:
                        master_data.append({
                            "audio_path": segment["audio_path"],
                            "text": segment["text"],  # Empty for now
                            "duration": segment["duration"],
                            "video": video_data["video_name"],
                            "channel": video_data["channel"]
                        })
        
        # Save master metadata
        master_path = self.output_dir / "metadata.jsonl"
        with open(master_path, 'w', encoding='utf-8') as f:
            for item in master_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        print(f"\nCreated master metadata: {master_path}")

# Run the preprocessor
if __name__ == "__main__":
    preprocessor = YouTubePreprocessor()
    preprocessor.run()