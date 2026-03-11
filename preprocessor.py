import os
import json
import subprocess
import shutil
from pathlib import Path
import librosa
import soundfile as sf
import torchaudio
from pydub import AudioSegment
import numpy as np
import torch
import cv2

# --- Deep Learning Dependencies ---
try:
    import whisper
    from df.enhance import init_df, enhance, load_audio, save_audio
    from paddleocr import PaddleOCR
    from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration
except ImportError as e:
    print(f"Warning: Missing dependency - {e}")
    print("Please install required neural libraries before running full pipeline.")

class YouTubePreprocessor:
    def __init__(self, root_dir="Youtube", delete_video_after=False):
        self.root_dir = Path(root_dir)
        self.output_dir = self.root_dir / "preprocess"
        self.output_dir.mkdir(exist_ok=True)
        
        self.delete_video_after = delete_video_after
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.target_sr = 16000
        print(f"Initializing pipeline on {self.device.upper()}...")

        # 1. Init Silero VAD
        self.vad_model, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad', model='silero_vad',
            force_reload=False, trust_repo=True
        )
        self.get_speech_timestamps = utils[0]

        # 2. Init DeepFilterNet (Primary Denoiser)
        self.df_model, self.df_state, _ = init_df()

        # 3. Init ASR Models
        print("Loading Whisper Large...")
        self.whisper_model = whisper.load_model("large").to(self.device)
        
        print("Loading Qwen1.7B Audio...")
        self.qwen_processor = AutoProcessor.from_pretrained("Qwen/Qwen-Audio")
        self.qwen_model = Qwen2AudioForConditionalGeneration.from_pretrained(
            "Qwen/Qwen-Audio", device_map="auto"
        )

        # 4. Init PaddleOCR
        self.ocr = PaddleOCR(use_angle_cls=True, lang='ms', use_gpu=(self.device == "cuda"))

    def convert_to_wav(self, input_path, output_path, target_sr):
        cmd = [
            "ffmpeg", "-i", str(input_path),
            "-ar", str(target_sr), "-ac", "1",
            "-acodec", "pcm_s16le",
            "-y", str(output_path), "-loglevel", "error"
        ]
        subprocess.run(cmd)

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

    def extract_gold_transcript(self, video_path, start_ms, end_ms):
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        start_frame = int((start_ms / 1000.0) * fps)
        end_frame = int((end_ms / 1000.0) * fps)
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        extracted_texts = set()
        frame_step = int(fps) 
        
        for frame_idx in range(start_frame, end_frame, frame_step):
            ret, frame = cap.read()
            if not ret: break
                
            height, width = frame.shape[:2]
            subtitle_crop = frame[int(height*0.8):height, :] 
            
            result = self.ocr.ocr(subtitle_crop, cls=True)
            if result and result[0]:
                for line in result[0]:
                    extracted_texts.add(line[1][0])
                    
        cap.release()
        return " ".join(list(extracted_texts))

    def transcribe_with_asr(self, audio_path):
        whisper_result = self.whisper_model.transcribe(str(audio_path), language="ms", initial_prompt="Berikut adalah perbualan dalam bahasa Melayu:")
        whisper_text = whisper_result["text"].strip()
        
        audio_np, sr = librosa.load(str(audio_path), sr=self.target_sr)
        inputs = self.qwen_processor(audios=audio_np, return_tensors="pt").to(self.device)
        generated_ids = self.qwen_model.generate(**inputs, max_length=256)
        qwen_text = self.qwen_processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        
        return whisper_text, qwen_text

    def preprocess_video(self, channel, audio_path, video_path):
        video_name = audio_path.stem
        video_name_clean = "".join(c for c in video_name if c.isalnum() or c in " _-")
        
        video_output_dir = self.output_dir / channel / video_name_clean
        video_output_dir.mkdir(parents=True, exist_ok=True)

        if (video_output_dir / "transcript.json").exists():
            print(f"Skipping {audio_path.name}, already processed")
            return 0
            
        temp_wav = video_output_dir / "temp_full.wav"
        
        print(f"  -> Converting to Native DF Sample Rate ({self.df_state.sr()}Hz)...")
        self.convert_to_wav(audio_path, temp_wav, self.df_state.sr())
        
        wav, sr = torchaudio.load(temp_wav)
        total_samples = wav.shape[1]
        
        # 10-Minute Chunk Math
        chunk_sec = 10 * 60
        chunk_samples = self.df_state.sr() * chunk_sec
        
        transcript_data = []
        segment_counter = 1
        
        for start_idx in range(0, total_samples, chunk_samples):
            end_idx = min(start_idx + chunk_samples, total_samples)
            ten_min_chunk = wav[:, start_idx:end_idx]
            
            start_min = (start_idx / sr) / 60
            end_min = (end_idx / sr) / 60
            print(f"\n  -> Processing 10-min block: {start_min:.1f}m to {end_min:.1f}m...")
            
            # 1. DeepFilterNet Denoising
            try:
                enhanced_48k = enhance(self.df_model, self.df_state, ten_min_chunk)
            except Exception as e:
                print(f"    ⚠️ DeepFilterNet failed on block ({e}). Skipping this 10m block.")
                continue

            # 2. Resample to 16kHz for VAD and ASR
            enhanced_16k = torchaudio.functional.resample(enhanced_48k, orig_freq=self.df_state.sr(), new_freq=self.target_sr)
            
            # 3. Silero VAD
            raw_timestamps = self.get_speech_timestamps(enhanced_16k.squeeze(), self.vad_model, sampling_rate=self.target_sr)
            
            # 4. Grouping (<30s)
            grouped_timestamps = self.group_vad_timestamps(raw_timestamps)
            print(f"    VAD found {len(grouped_timestamps)} Whisper-ready chunks in this block.")

            # 5. Extract, Transcribe, and Save Chunks
            for ts in grouped_timestamps:
                # Calculate absolute milliseconds (Relative chunk time + Absolute block time)
                rel_start_ms = (ts['start'] / self.target_sr) * 1000
                rel_end_ms = (ts['end'] / self.target_sr) * 1000
                
                abs_start_ms = ((start_idx / sr) * 1000) + rel_start_ms
                abs_end_ms = ((start_idx / sr) * 1000) + rel_end_ms
                duration = (abs_end_ms - abs_start_ms) / 1000.0
                
                segment_filename = f"segment_{segment_counter:04d}.wav"
                final_segment_path = video_output_dir / segment_filename
                
                # Slice the pristine 16k audio and save
                chunk_tensor = enhanced_16k[:, ts['start']:ts['end']]
                torchaudio.save(final_segment_path, chunk_tensor, self.target_sr)
                
                # Transcription & OCR
                gold_text = ""
                if video_path and video_path.exists():
                    gold_text = self.extract_gold_transcript(video_path, abs_start_ms, abs_end_ms)
                
                whisper_text, qwen_text = self.transcribe_with_asr(final_segment_path)
                
                transcript_data.append({
                    "segment": segment_filename,
                    "start_time": abs_start_ms / 1000.0,
                    "end_time": abs_end_ms / 1000.0,
                    "duration": duration,
                    "text": gold_text,                 
                    "whisper_text": whisper_text,      
                    "qwen_text": qwen_text,            
                    "audio_path": str(final_segment_path.relative_to(self.root_dir))
                })
                
                print(f"    ✅ Processed {segment_filename} ({duration:.1f}s)")
                segment_counter += 1
            
        temp_wav.unlink()
        
        # Save Metadata
        transcript_path = video_output_dir / "transcript.json"
        with open(transcript_path, 'w', encoding='utf-8') as f:
            json.dump({
                "video_name": video_name,
                "channel": channel,
                "segments": transcript_data,
                "total_duration": sum(s["duration"] for s in transcript_data)
            }, f, indent=2, ensure_ascii=False)
            
        if self.delete_video_after and video_path and video_path.exists():
            print(f"  -> Cleaning up video file: {video_path.name}")
            video_path.unlink()
            
        return segment_counter - 1

    def run(self):
        total_segments = 0
        total_videos = 0
        
        for channel_dir in self.root_dir.iterdir():
            if not channel_dir.is_dir() or channel_dir.name == "preprocess": continue
                
            channel = channel_dir.name
            for audio_path in channel_dir.glob("*"):
                if audio_path.is_file() and audio_path.suffix.lower() in ['.m4a', '.opus', '.mp3', '.wav', '.flac']:
                    video_dir = channel_dir / "video"
                    video_path = None
                    if video_dir.exists():
                        for ext in ['.mp4', '.mkv', '.webm', '.avi']:
                            potential_video = video_dir / f"{audio_path.stem}{ext}"
                            if potential_video.exists():
                                video_path = potential_video
                                break
                    
                    segments = self.preprocess_video(channel, audio_path, video_path)
                    total_segments += segments
                    total_videos += 1
                    
        self.create_master_metadata()
        print(f"\n✅ Preprocessing complete! Created {total_segments} segments across {total_videos} videos.")

    def create_master_metadata(self):
        master_data = []
        for channel_dir in self.output_dir.iterdir():
            if not channel_dir.is_dir(): continue
            for video_dir in channel_dir.iterdir():
                if not video_dir.is_dir(): continue
                transcript_file = video_dir / "transcript.json"
                if transcript_file.exists():
                    with open(transcript_file, 'r', encoding='utf-8') as f:
                        video_data = json.load(f)
                    for segment in video_data["segments"]:
                        master_data.append({
                            "audio_path": segment["audio_path"],
                            "text": segment["text"],          
                            "whisper_text": segment.get("whisper_text", ""),
                            "qwen_text": segment.get("qwen_text", ""),
                            "duration": segment["duration"],
                            "video": video_data["video_name"],
                            "channel": video_data["channel"]
                        })
                        
        master_path = self.output_dir / "metadata.jsonl"
        with open(master_path, 'w', encoding='utf-8') as f:
            for item in master_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

if __name__ == "__main__":
    preprocessor = YouTubePreprocessor(delete_video_after=False)
    preprocessor.run()      with open(master_path, 'w', encoding='utf-8') as f:
            for item in master_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

if __name__ == "__main__":
    preprocessor = YouTubePreprocessor(delete_video_after=False)
    preprocessor.run()