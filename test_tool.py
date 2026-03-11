import os
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
except ImportError as e:
    print(f"Warning: Missing dependency - {e}")

class ToolTester:
    def __init__(self, output_dir="test_outputs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.target_sr = 16000
        print(f"Using device: {self.device.upper()}")

    def _convert_to_df_wav(self, input_path, df_sr):
        """Converts raw video/audio to DeepFilterNet's native sample rate."""
        out_path = self.output_dir / "temp_df_in.wav"
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
        print("\n--- Initializing Models ---")
        df_model, df_state, _ = init_df()
        whisper_model = whisper.load_model("large").to(self.device)
        vad_model, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=False, trust_repo=True
        )
        get_speech_timestamps = utils[0]

        print("\n--- 1. Loading & Cutting to Direct 10-Minute Block ---")
        wav_path = self._convert_to_df_wav(audio_path, df_state.sr())
        wav, sr = torchaudio.load(wav_path)
        
        # We now slice EXACTLY 10 minutes (the proven safe limit for the A40)
        ten_mins_samples = df_state.sr() * 600
        wav_10min = wav[:, :ten_mins_samples]
        print(f"Loaded 10-minute block cleanly. Shape: {wav_10min.shape}")

        print("\n--- 2. DeepFilterNet Denoising ---")
        # No more confusing loops! Denoise the 10-minute block in one direct shot.
        enhanced_48k = enhance(df_model, df_state, wav_10min)
        print("✅ Denoising complete! Audio is pristine.")

        print("\n--- 3. Resampling & Silero VAD ---")
        enhanced_16k = torchaudio.functional.resample(enhanced_48k, orig_freq=df_state.sr(), new_freq=self.target_sr)
        
        raw_timestamps = get_speech_timestamps(enhanced_16k.squeeze(), vad_model, sampling_rate=self.target_sr)
        print(f"VAD found {len(raw_timestamps)} clean segments of speech.")

        print("\n--- 4. Grouping (<30s) & Transcribing ---")
        grouped_timestamps = self.group_vad_timestamps(raw_timestamps)
        print(f"Grouped into {len(grouped_timestamps)} Whisper-ready chunks.")

        # Transcribe the first 3 chunks to verify
        for i in range(min(3, len(grouped_timestamps))):
            ts = grouped_timestamps[i]
            chunk_tensor = enhanced_16k[:, ts['start']:ts['end']]
            
            chunk_path = self.output_dir / f"ultimate_chunk_{i+1}.wav"
            torchaudio.save(chunk_path, chunk_tensor, self.target_sr)
            
            # Using prompt to help prevent the "Ika. Ika. Ika." hallucination we saw earlier
            result = whisper_model.transcribe(str(chunk_path), language="ms", initial_prompt="Berikut adalah perbualan dalam bahasa Melayu:")
            print(f"🗣️ Chunk {i+1}: {result['text'].strip()}")

        wav_path.unlink()
        print("\n✅ Ultimate Pipeline Test Complete!")

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