import os
import json
import argparse
import subprocess
import shutil
from pathlib import Path
import torchaudio
import torch
import cv2  
import logging
from tqdm import tqdm

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

    # --- Utilities ---
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
        # Also check same directory as audio
        search_dirs = [video_dir, audio_path.parent]
        for d in search_dirs:
            for ext in video_extensions:
                potential_video = d / (audio_path.stem + ext)
                if potential_video.exists():
                    return potential_video
        return None

    def merge_visual_results(self, raw_detections):
        """Groups consecutive detections of same text into start/end chunks."""
        if not raw_detections: return []
        merged = []
        curr = None
        for det in raw_detections:
            if curr is None:
                curr = {"start": det['time'], "end": det['time'], "text": det['text']}
            elif det['text'] == curr['text']:
                curr['end'] = det['time']
            else:
                merged.append(curr)
                curr = {"start": det['time'], "end": det['time'], "text": det['text']}
        if curr: merged.append(curr)
        return merged

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
        chunk_samples = df_state.sr() * (10 * 60) # 10 min blocks
        
        transcript_data = []
        segment_counter = 1

        for start_idx in range(0, total_samples, chunk_samples):
            end_idx = min(start_idx + chunk_samples, total_samples)
            ten_min_chunk = wav[:, start_idx:end_idx]
            
            print(f"\n  -> Block {(start_idx/sr)/60:.1f}m to {(end_idx/sr)/60:.1f}m:")
            print("     - Denoising...", end="\r")
            enhanced_48k = enhance(df_model, df_state, ten_min_chunk)
            print("     - Running VAD...  ", end="\r")
            enhanced_16k = torchaudio.functional.resample(enhanced_48k, orig_freq=df_state.sr(), new_freq=self.target_sr)
            raw_timestamps = get_speech_timestamps(enhanced_16k.squeeze(), vad_model, sampling_rate=self.target_sr)
            
            grouped_timestamps = self.group_vad_timestamps(raw_timestamps)
            print(f"     - Found {len(grouped_timestamps)} segments.")

            for ts in grouped_timestamps:
                abs_start_s = ((start_idx / sr)) + (ts['start'] / self.target_sr)
                abs_end_s = ((start_idx / sr)) + (ts['end'] / self.target_sr)
                
                segment_filename = f"segment_{segment_counter:04d}.wav"
                final_segment_path = video_output_dir / segment_filename
                torchaudio.save(final_segment_path, enhanced_16k[:, ts['start']:ts['end']], self.target_sr)
                
                whisper_result = whisper_model.transcribe(str(final_segment_path), language="ms", initial_prompt="Keep umm, eh, lah.")
                qwen_result = qwen_model.transcribe(audio=str(final_segment_path))
                
                transcript_data.append({
                    "segment": segment_filename,
                    "start_time": abs_start_s, "end_time": abs_end_s,
                    "duration": abs_end_s - abs_start_s,
                    "text": "", "whisper_text": whisper_result['text'].strip(),
                    "qwen_text": qwen_result[0].text.strip() if qwen_result else "",
                    "audio_path": str(final_segment_path.relative_to(self.output_base.parent)),
                    "verified": False
                })
                print(f"       File: {segment_filename} | Whisper: {transcript_data[-1]['whisper_text'][:50]}...")
                segment_counter += 1

        with open(video_output_dir / "transcript.json", 'w', encoding='utf-8') as f:
            json.dump({"video_name": video_name, "segments": transcript_data}, f, indent=2, ensure_ascii=False)
        print(f"\n✅ Step 1 Complete! Saved to transcript.json")

    # ==========================================
    # STEP 2 (OPTION A): SEGMENTED OCR (Original)
    # ==========================================
    def extract_subtitle_from_frame(self, video_path, start_sec, end_sec, ocr_model):
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        mid_time = start_sec + ((end_sec - start_sec) / 2)
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(mid_time * fps))
        ret, frame = cap.read()
        cap.release()
        if not ret: return ""
        h, w = frame.shape[:2]
        crop = frame[int(h * 0.7):h, :] # Bottom 30%
        result = ocr_model.ocr(crop)
        return " ".join([line[1][0] for line in result[0]]).strip() if (result and result[0]) else ""

    def test_ocr_pass(self, audio_path):
        video_name_clean = "".join(c for c in audio_path.stem if c.isalnum() or c in " _-")
        json_p = self.output_base / video_name_clean / "transcript.json"
        video_p = self.find_video_file(audio_path)
        if not json_p.exists() or not video_p: return print("Missing JSON or Video!")

        print(f"--- Starting Segmented OCR ---")
        ocr_model = PaddleOCR(lang='ms')
        with open(json_p, 'r') as f: data = json.load(f)
        
        for seg in tqdm(data['segments']):
            seg['text'] = self.extract_subtitle_from_frame(video_p, seg['start_time'], seg['end_time'], ocr_model)
        
        with open(json_p, 'w') as f: json.dump(data, f, indent=2, ensure_ascii=False)
        print("✅ Segmented OCR Pass Done.")

    # ==========================================
    # STEP 2 (OPTION B): SAMPLED SWEEP (Enhanced)
    # ==========================================
    def test_ocr_sweep(self, audio_path, sample_rate=2.0):
        video_name_clean = "".join(c for c in audio_path.stem if c.isalnum() or c in " _-")
        json_p = self.output_base / video_name_clean / "transcript.json"
        video_p = self.find_video_file(audio_path)
        
        if not video_p: 
            print("❌ Video not found!")
            return
        
        print(f"\n--- Starting Visual Sweep (every {sample_rate}s) ---")
        ocr_model = PaddleOCR(lang='ms') # Keep it clean to avoid PaddleOCR parameter errors
        
        abs_video_path = str(video_p.resolve())
        print(f"📂 Original video path: {abs_video_path}")
        
        # --- THE BRUTE-FORCE COPY TRICK ---
        # OpenCV sees through symlinks. We physically copy the file to guarantee a clean path.
        temp_file = Path(f"/tmp/temp_cv2_video{video_p.suffix}")
        if temp_file.exists():
            temp_file.unlink() # Remove old temp file if it exists
            
        print(f"⏳ Copying video to a temporary safe location: {temp_file}...")
        shutil.copy(abs_video_path, temp_file)
        
        cap = cv2.VideoCapture(str(temp_file))
        
        # --- NEW SAFETY CHECKS ---
        if not cap.isOpened():
            print(f"❌ Error: OpenCV could not open the copied video file. The file itself might be corrupted or missing video streams.")
            temp_file.unlink()
            return
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0.0:
            print(f"❌ Error: OpenCV reported 0 FPS. The video format might be unsupported.")
            cap.release()
            temp_file.unlink()
            return
        # -------------------------
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        
        raw_hits = []
        for sec in tqdm(range(0, int(duration), int(sample_rate)), desc="Sweeping Video"):
            cap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
            ret, frame = cap.read()
            if not ret: break
            
            h, w = frame.shape[:2]
            crop = frame[int(h*0.7):h, :] # Focus on bottom area
            res = ocr_model.ocr(crop)
            if res and res[0]:
                text = " ".join([l[1][0] for l in res[0]]).strip()
                if text: raw_hits.append({"time": sec, "text": text})
        
        cap.release()
        temp_file.unlink() # Clean up the copy!

        visual_segments = self.merge_visual_results(raw_hits)
        print(f"✨ Found {len(visual_segments)} unique visual text events.")

        # --- Mapping back to transcript.json ---
        if json_p.exists():
            print("🔗 Mapping visual text to audio segments...")
            with open(json_p, 'r') as f: data = json.load(f)
            
            for seg in data['segments']:
                s_start, s_end = seg['start_time'], seg['end_time']
                # Find visual text that appeared during this audio segment
                matches = [v['text'] for v in visual_segments if (v['start'] <= s_end and v['end'] >= s_start)]
                if matches:
                    # Take the most frequent or longest match, here we just join unique ones
                    seg['text'] = " | ".join(list(dict.fromkeys(matches)))

            with open(json_p, 'w') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print(f"✅ Updated {json_p} with visual data.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tool", type=str, required=True, choices=['pipeline', 'ocr_pass', 'ocr_sweep', 'ocr_test'])
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--sample_rate", type=float, default=2.0)
    args = parser.parse_args()

    tester = ToolTester()
    inp = Path(args.input)
    if args.tool == 'pipeline': tester.test_pipeline(inp)
    elif args.tool == 'ocr_pass': tester.test_ocr_pass(inp)
    elif args.tool == 'ocr_sweep': tester.test_ocr_sweep(inp, args.sample_rate)
#t_whisper(input_path)
    elif args.tool == 'qwen':
        tester.test_qwen(input_path)
    elif args.tool == 'ocr':
        tester.test_ocr(input_path, start_s=args.start, end_s=args.end)