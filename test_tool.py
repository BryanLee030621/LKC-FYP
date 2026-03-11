import os
import json
import argparse
import subprocess
import shutil
from pathlib import Path
import librosa
import torchaudio
import torch

# --- Deep Learning Dependencies ---
try:
    from df.enhance import init_df, enhance, load_audio, save_audio
    import whisper
    from qwen_asr import Qwen3ASRModel
except ImportError as e:
    print(f"Warning: Missing dependency - {e}")

class ToolTester:
    def __init__(self, output_dir="test_outputs"):
        self.output_base = Path(output_dir)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.target_sr = 16000
        print(f"Using device: {self.device.upper()}")

    def _convert_to_df_wav(self, input_path, df_sr, temp_dir):
        """Converts raw video/audio to DeepFilterNet's native sample rate."""
        out_path = temp_dir / "temp_df_in.wav"
        subprocess.run([
            "ffmpeg", "-i", str(input_path),
            "-ar", str(df_sr), "-ac", "1",
            "-acodec", "pcm_s16le", "-y", str(out_path), "-loglevel", "error"
        ])
        return out_path

    def group_vad_timestamps(self, timestamps, max_sec=29.0):
        """Groups raw VAD timestamps into Whisper-friendly <30s chunks."""
        max_samples = int(max_sec * self.target_sr)
        if not timestamps: return []

        grouped = []
        current_start = timestamps[0]['start']
        current_end = timestamps[0]['end']

        for i in range(1, len(timestamps)):
            ts = timestamps[i]
            if (ts['end'] - current_start) <= max_samples:
                current_end = ts['end']
            else:
                grouped.append({'start': current_start, 'end': current_end})
                current_start = ts['start']
                current_end = ts['end']

        grouped.append({'start': current_start, 'end': current_end})
        return grouped

    def test_pipeline(self, audio_path):
        # 1. Setup Output Directory mimicking preprocessor.py
        video_name = audio_path.stem
        video_name_clean = "".join(c for c in video_name if c.isalnum() or c in " _-")
        video_output_dir = self.output_base / video_name_clean
        video_output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n📁 Output directory created at: {video_output_dir}")

        print("\n--- Initializing Models ---")
        df_model, df_state, _ = init_df()
        
        print("Loading Whisper Large...")
        whisper_model = whisper.load_model("large").to(self.device)
        
        print("Loading Qwen3-ASR 1.7B...")
        # Using float16 for safety on most GPUs. Switch to bfloat16 if using Ampere (A40/A100/RTX3000+)
        qwen_model = Qwen3ASRModel.from_pretrained(
            "Qwen/Qwen3-ASR-1.7B", 
            dtype=torch.float16, 
            device_map="auto"
        )
        
        print("Loading Silero VAD...")
        vad_model, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=False, trust_repo=True
        )
        get_speech_timestamps = utils[0]

        print(f"\n--- 1. Loading & Converting {video_name_clean} ---")
        wav_path = self._convert_to_df_wav(audio_path, df_state.sr(), video_output_dir)
        wav, sr = torchaudio.load(wav_path)
        
        total_samples = wav.shape[1]
        chunk_sec = 10 * 60
        chunk_samples = df_state.sr() * chunk_sec
        
        transcript_data = []
        segment_counter = 1

        print("\n--- 2. Processing in 10-Minute Blocks ---")
        for start_idx in range(0, total_samples, chunk_samples):
            end_idx = min(start_idx + chunk_samples, total_samples)
            ten_min_chunk = wav[:, start_idx:end_idx]
            
            start_min = (start_idx / sr) / 60
            end_min = (end_idx / sr) / 60
            print(f"\n  -> Block {start_min:.1f}m to {end_min:.1f}m:")
            
            # DeepFilterNet
            print("     - Denoising...", end="\r")
            enhanced_48k = enhance(df_model, df_state, ten_min_chunk)
            
            # Resample & VAD
            print("     - Running VAD...  ", end="\r")
            enhanced_16k = torchaudio.functional.resample(enhanced_48k, orig_freq=df_state.sr(), new_freq=self.target_sr)
            raw_timestamps = get_speech_timestamps(enhanced_16k.squeeze(), vad_model, sampling_rate=self.target_sr)
            
            # Group into <30s
            grouped_timestamps = self.group_vad_timestamps(raw_timestamps)
            print(f"     - Found {len(grouped_timestamps)} Whisper-ready chunks.")

            # Process all chunks in this 10-min block
            for ts in grouped_timestamps:
                # Calculate absolute time
                rel_start_ms = (ts['start'] / self.target_sr) * 1000
                rel_end_ms = (ts['end'] / self.target_sr) * 1000
                
                abs_start_ms = ((start_idx / sr) * 1000) + rel_start_ms
                abs_end_ms = ((start_idx / sr) * 1000) + rel_end_ms
                duration = (abs_end_ms - abs_start_ms) / 1000.0
                
                # Setup filename and save audio
                segment_filename = f"segment_{segment_counter:04d}.wav"
                final_segment_path = video_output_dir / segment_filename
                
                chunk_tensor = enhanced_16k[:, ts['start']:ts['end']]
                torchaudio.save(final_segment_path, chunk_tensor, self.target_sr)
                
                # --- Transcribe with Whisper ---
                whisper_result = whisper_model.transcribe(
                    str(final_segment_path), 
                    language="ms", 
                    initial_prompt="Berikut adalah perbualan dalam bahasa Melayu:"
                )
                whisper_text = whisper_result['text'].strip()
                
                # --- Transcribe with Qwen3-ASR ---
                qwen_result = qwen_model.transcribe(audio=str(final_segment_path))
                qwen_text = qwen_result[0].text.strip() if qwen_result else ""
                
                # Append to JSON data list
                transcript_data.append({
                    "segment": segment_filename,
                    "start_time": abs_start_ms / 1000.0,
                    "end_time": abs_end_ms / 1000.0,
                    "duration": duration,
                    "text": "",             # Blank for fast test_tool run (requires PaddleOCR/Video)
                    "whisper_text": whisper_text,      
                    "qwen_text": qwen_text, 
                    "audio_path": str(final_segment_path.relative_to(self.output_base.parent)),
                    "verified": False       # Default to False for correction_tool.py
                })
                
                print(f"\n       File: {segment_filename}")
                print(f"       🗣️ Whisper: {whisper_text[:100]}...")
                print(f"       🤖 Qwen3:   {qwen_text[:100]}...")
                segment_counter += 1

        # Cleanup temp full audio
        wav_path.unlink()
        
        # 3. Save JSON Metadata
        transcript_path = video_output_dir / "transcript.json"
        with open(transcript_path, 'w', encoding='utf-8') as f:
            json.dump({
                "video_name": video_name,
                "channel": "TestChannel",
                "segments": transcript_data,
                "total_duration": sum(s["duration"] for s in transcript_data)
            }, f, indent=2, ensure_ascii=False)
            
        print(f"\n✅ Ultimate Pipeline Test Complete! Processed {segment_counter - 1} total segments.")
        print(f"📂 Check '{video_output_dir}' for audio files and your newly formatted transcript.json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tool", type=str, required=True, 
                        choices=['vad', 'deepfilter', 'resemble', 'whisper', 'qwen', 'ocr', 'pipeline'])
    parser.add_argument("--input", type=str, required=True)
    args = parser.parse_args()
    input_path = Path(args.input)

    tester = ToolTester()
    if args.tool == 'pipeline': tester.test_pipeline(input_path)
#t_whisper(input_path)
    elif args.tool == 'qwen':
        tester.test_qwen(input_path)
    elif args.tool == 'ocr':
        tester.test_ocr(input_path, start_s=args.start, end_s=args.end)