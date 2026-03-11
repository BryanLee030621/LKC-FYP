import os
import json
import argparse
import subprocess
import shutil
from pathlib import Path
import torchaudio
import torch
import cv2  
import logging # Added to silence PaddleOCR

# Silence PaddleOCR debug logs safely
logging.getLogger('ppocr').setLevel(logging.ERROR)

# --- Deep Learning Dependencies ---
try:
    from df.enhance import init_df, enhance
    import whisper
    from qwen_asr import Qwen3ASRModel
    from paddleocr import PaddleOCR  
except ImportError as e:
    print(f"Warning: Missing dependency - {e}")

class ToolTester:
    def __init__(self, output_dir="test_outputs"):
        self.output_base = Path(output_dir)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.target_sr = 16000
        print(f"Using device: {self.device.upper()}")

    def _convert_to_df_wav(self, input_path, df_sr, temp_dir):
        out_path = temp_dir / "temp_df_in.wav"
        subprocess.run([
            "ffmpeg", "-i", str(input_path),
            "-ar", str(df_sr), "-ac", "1",
            "-acodec", "pcm_s16le", "-y", str(out_path), "-loglevel", "error"
        ])
        return out_path

    def group_vad_timestamps(self, timestamps, max_sec=29.0):
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

    def find_video_file(self, audio_path):
        video_dir = audio_path.parent / "video"
        video_extensions = ['.mp4', '.webm', '.mkv', '.avi']
        for ext in video_extensions:
            potential_video = video_dir / (audio_path.stem + ext)
            if potential_video.exists():
                return potential_video
        return None

    def extract_subtitle_from_frame(self, video_path, start_sec, end_sec, ocr_model):
        if not video_path or not video_path.exists():
            return ""

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return ""

        fps = cap.get(cv2.CAP_PROP_FPS)
        mid_time = start_sec + ((end_sec - start_sec) / 2)
        frame_no = int(mid_time * fps)

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            return ""

        h, w = frame.shape[:2]
        crop_img = frame[int(h * 0.7):h, 0:w]

        result = ocr_model.ocr(crop_img)
        
        if not result or not result[0]:
            return ""
            
        texts = [line[1][0] for line in result[0]]
        return " ".join(texts).strip()

    # ==========================================
    # STEP 1: AUDIO PASS (VAD + Whisper + Qwen)
    # ==========================================
    def test_pipeline(self, audio_path):
        video_name = audio_path.stem
        video_name_clean = "".join(c for c in video_name if c.isalnum() or c in " _-")
        video_output_dir = self.output_base / video_name_clean
        video_output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n📁 Output directory created at: {video_output_dir}")
        print("\n--- Initializing Audio Models (Step 1) ---")
        
        df_model, df_state, _ = init_df()
        print("Loading Whisper Large...")
        whisper_model = whisper.load_model("large").to(self.device)
        print("Loading Qwen3-ASR 1.7B...")
        qwen_model = Qwen3ASRModel.from_pretrained("Qwen/Qwen3-ASR-1.7B", dtype=torch.float16, device_map="auto")
        print("Loading Silero VAD...")
        vad_model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=False, trust_repo=True)
        get_speech_timestamps = utils[0]

        print(f"\n--- Processing Audio: {video_name_clean} ---")
        wav_path = self._convert_to_df_wav(audio_path, df_state.sr(), video_output_dir)
        wav, sr = torchaudio.load(wav_path)
        
        total_samples = wav.shape[1]
        chunk_sec = 10 * 60
        chunk_samples = df_state.sr() * chunk_sec
        
        transcript_data = []
        segment_counter = 1

        for start_idx in range(0, total_samples, chunk_samples):
            end_idx = min(start_idx + chunk_samples, total_samples)
            ten_min_chunk = wav[:, start_idx:end_idx]
            
            start_min = (start_idx / sr) / 60
            end_min = (end_idx / sr) / 60
            print(f"\n  -> Block {start_min:.1f}m to {end_min:.1f}m:")
            
            print("     - Denoising...", end="\r")
            enhanced_48k = enhance(df_model, df_state, ten_min_chunk)
            print("     - Running VAD...  ", end="\r")
            enhanced_16k = torchaudio.functional.resample(enhanced_48k, orig_freq=df_state.sr(), new_freq=self.target_sr)
            raw_timestamps = get_speech_timestamps(enhanced_16k.squeeze(), vad_model, sampling_rate=self.target_sr)
            
            grouped_timestamps = self.group_vad_timestamps(raw_timestamps)
            print(f"     - Found {len(grouped_timestamps)} Whisper-ready chunks.")

            for ts in grouped_timestamps:
                rel_start_ms = (ts['start'] / self.target_sr) * 1000
                rel_end_ms = (ts['end'] / self.target_sr) * 1000
                
                abs_start_ms = ((start_idx / sr) * 1000) + rel_start_ms
                abs_end_ms = ((start_idx / sr) * 1000) + rel_end_ms
                
                start_sec = abs_start_ms / 1000.0
                end_sec = abs_end_ms / 1000.0
                duration = end_sec - start_sec
                
                segment_filename = f"segment_{segment_counter:04d}.wav"
                final_segment_path = video_output_dir / segment_filename
                
                chunk_tensor = enhanced_16k[:, ts['start']:ts['end']]
                torchaudio.save(final_segment_path, chunk_tensor, self.target_sr)
                
                whisper_result = whisper_model.transcribe(
                    str(final_segment_path), task="transcribe", language="ms", best_of=5,
                    condition_on_previous_text=False, carry_initial_prompt=True, temperature=1, logprob_threshold=-2,
                    initial_prompt="Berikut adalah perbualan dalam bahasa Melayu. Keep the word umm, eh, lah."
                )
                whisper_text = whisper_result['text'].strip()
                
                qwen_result = qwen_model.transcribe(audio=str(final_segment_path))
                qwen_text = qwen_result[0].text.strip() if qwen_result else ""
                
                transcript_data.append({
                    "segment": segment_filename,
                    "start_time": start_sec,
                    "end_time": end_sec,
                    "duration": duration,
                    "text": "",  # Left blank for the OCR pass!
                    "whisper_text": whisper_text,      
                    "qwen_text": qwen_text, 
                    "audio_path": str(final_segment_path.relative_to(self.output_base.parent)),
                    "verified": False
                })
                
                print(f"\n       File: {segment_filename}")
                print(f"       🗣️ Whisper: {whisper_text[:100]}...")
                print(f"       🤖 Qwen3:   {qwen_text[:100]}...")
                segment_counter += 1

        wav_path.unlink()
        
        transcript_path = video_output_dir / "transcript.json"
        with open(transcript_path, 'w', encoding='utf-8') as f:
            json.dump({
                "video_name": video_name,
                "channel": "TestChannel",
                "segments": transcript_data,
                "total_duration": sum(s["duration"] for s in transcript_data)
            }, f, indent=2, ensure_ascii=False)
            
        print(f"\n✅ Step 1 Complete! Saved {segment_counter - 1} segments to transcript.json")

    # ==========================================
    # STEP 2: VIDEO PASS (PaddleOCR JSON Updater)
    # ==========================================
    def test_ocr_pass(self, audio_path):
        video_name = audio_path.stem
        video_name_clean = "".join(c for c in video_name if c.isalnum() or c in " _-")
        video_output_dir = self.output_base / video_name_clean
        transcript_path = video_output_dir / "transcript.json"

        print(f"\n--- Starting Video OCR Pass (Step 2) ---")
        
        if not transcript_path.exists():
            print(f"❌ Error: Could not find {transcript_path}. Run '--tool pipeline' first!")
            return

        video_path = self.find_video_file(audio_path)
        if not video_path:
            print(f"❌ Error: Could not find matching video in '{audio_path.parent / 'video'}'")
            return

        print(f"🎥 Found video: {video_path.name}")
        print("Loading PaddleOCR (Malay)...")
        # Updated to fix the crashing arguments
        ocr_model = PaddleOCR(use_textline_orientation=False, lang='ms')

        # Load existing JSON
        with open(transcript_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        print(f"📝 Processing {len(data['segments'])} segments from JSON...")
        
        # Loop through JSON and update
        for idx, segment in enumerate(data['segments']):
            start_s = segment['start_time']
            end_s = segment['end_time']
            
            print(f"   [{idx+1}/{len(data['segments'])}] Extracting OCR for {segment['segment']} ({start_s:.1f}s - {end_s:.1f}s)...", end="\r")
            
            ocr_text = self.extract_subtitle_from_frame(video_path, start_s, end_s, ocr_model)
            segment['text'] = ocr_text

        # Save updated JSON
        with open(transcript_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"\n\n✅ Step 2 Complete! Successfully updated transcript.json with OCR text.")

    # ==========================================
    # QUICK TESTER: Single Frame OCR
    # ==========================================
    def test_ocr_single(self, audio_path, start_s=60.0, end_s=65.0):
        print(f"\n--- Testing Single Frame OCR ---")
        video_path = self.find_video_file(audio_path)
        
        if not video_path:
            print(f"❌ Error: Could not find matching video in '{audio_path.parent / 'video'}'")
            return

        print(f"🎥 Found video: {video_path.name}")
        print("Loading PaddleOCR (Malay)...")
        # Updated to fix the crashing arguments
        ocr_model = PaddleOCR(use_textline_orientation=False, lang='ms')
        
        print(f"⏱️ Grabbing frame between {start_s}s and {end_s}s...")
        text = self.extract_subtitle_from_frame(video_path, start_s, end_s, ocr_model)
        
        print("\n📝 OCR Result:")
        print(f"[{text}]" if text else "[No text detected]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tool", type=str, required=True, 
                        choices=['vad', 'deepfilter', 'resemble', 'whisper', 'qwen', 'pipeline', 'ocr_pass', 'ocr_test'])
    parser.add_argument("--input", type=str, required=True)
    
    parser.add_argument("--start", type=float, default=60.0)
    parser.add_argument("--end", type=float, default=65.0)
    
    args = parser.parse_args()
    input_path = Path(args.input)

    tester = ToolTester()
    if args.tool == 'pipeline': 
        tester.test_pipeline(input_path)
    elif args.tool == 'ocr_pass':
        tester.test_ocr_pass(input_path)
    elif args.tool == 'ocr_test':
        tester.test_ocr_single(input_path, start_s=args.start, end_s=args.end)
#t_whisper(input_path)
    elif args.tool == 'qwen':
        tester.test_qwen(input_path)
    elif args.tool == 'ocr':
        tester.test_ocr(input_path, start_s=args.start, end_s=args.end)