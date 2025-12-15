# extract_hardcoded_subs.py
from videocr import save_subtitles_to_file
import json
from pathlib import Path

def run_videocr(video_path, output_srt_path, lang='eng'):  # Changed from 'ms' to 'mal'
    try:
        # Call the library function directly - NOTE: Different parameters!
        save_subtitles_to_file(
            video_path=str(video_path),
            file_path=str(output_srt_path),
            lang=lang,                # Language code for Malay - use 'mal' or 'mal+eng'
            conf_threshold=65,        # Default is 65 - adjust as needed
            sim_threshold=90,         # Default is 90 - adjust as needed
            # Note: use_gpu parameter doesn't exist in this version
            # use_fullframe=False (default) - only bottom half is used for OCR
        )
        print(f"✓ Subtitles extracted: {output_srt_path}")
        return True, None
    except Exception as e:
        print(f"✗ Extraction failed: {e}")
        return False, str(e)

# Rest of your functions remain the same...
def parse_srt_to_segments(srt_path, audio_duration):
    """
    Parse SRT file and convert to segment format matching transcript.json
    SRT format:
    1
    00:00:00,000 --> 00:00:04,000
    Hello world
    
    Returns: List of dicts with start, end, text
    """
    segments = []
    
    if not srt_path.exists():
        return segments
    
    with open(srt_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # SRT parsing
    blocks = content.strip().split('\n\n')
    
    for block in blocks:
        lines = block.strip().split('\n')
        if len(lines) >= 3:
            try:
                # Parse timestamp line (e.g., "00:00:00,000 --> 00:00:04,000")
                time_line = lines[1]
                start_str, end_str = time_line.split(' --> ')
                
                # Convert to seconds
                def srt_time_to_seconds(srt_time):
                    h, m, rest = srt_time.split(':')

                    # rest can be "52,540" OR "52.540440"
                    if ',' in rest:
                        s, frac = rest.split(',')
                    elif '.' in rest:
                        s, frac = rest.split('.')
                    else:
                        s, frac = rest, '0'

                    return (
                        int(h) * 3600
                        + int(m) * 60
                        + float(f"{s}.{frac}")
                    )
                
                start_sec = srt_time_to_seconds(start_str)
                end_sec = srt_time_to_seconds(end_str)
                
                # Get text (might be multiple lines)
                text = ' '.join(lines[2:]).strip()
                
                # Only add if text is not empty and duration is reasonable
                if text and (end_sec - start_sec) > 0.5:
                    segments.append({
                        'start': start_sec,
                        'end': end_sec,
                        'text': text,
                        'source': 'hardcoded'
                    })
                    
            except Exception as e:
                print(f"Warning: Could not parse SRT block: {e}")
                continue
    
    print(f"Parsed {len(segments)} subtitle segments from SRT")
    return segments

def map_subs_to_audio_segments(hardcoded_segments, audio_segments):
    """
    Map hardcoded subtitle timestamps to your 30s audio segments
    Returns: Updated audio segments with hardcoded text where available
    """
    for audio_seg in audio_segments:
        audio_start = audio_seg.get('start_time', 0)
        audio_end = audio_seg.get('end_time', 0)
        audio_seg['hardcoded_text'] = ''
        
        # Collect all hardcoded text that falls within this audio segment
        texts_in_segment = []
        
        for sub in hardcoded_segments:
            sub_start = sub['start']
            sub_end = sub['end']
            
            # Check if subtitle overlaps with audio segment
            if not (sub_end < audio_start or sub_start > audio_end):
                # Calculate overlap percentage
                overlap_start = max(sub_start, audio_start)
                overlap_end = min(sub_end, audio_end)
                overlap_duration = overlap_end - overlap_start
                sub_duration = sub_end - sub_start
                
                if overlap_duration / sub_duration > 0.5:  # More than 50% overlap
                    texts_in_segment.append(sub['text'])
        
        if texts_in_segment:
            # Join multiple subtitle lines with space
            audio_seg['hardcoded_text'] = ' '.join(texts_in_segment)
            audio_seg['has_hardcoded'] = True
    
    return audio_segments

def process_channel_videos(base_dir="Youtube"):
    """Process all videos in a channel to extract hardcoded subtitles"""
    base_path = Path(base_dir)
    
    for channel_dir in base_path.iterdir():
        if not channel_dir.is_dir():
            continue
            
        print(f"\nProcessing channel: {channel_dir.name}")
        
        # Find video files (mp4, mkv, etc.)
        video_files = list(channel_dir.glob("video_*.mp4")) + \
                     list(channel_dir.glob("video_*.mkv")) + \
                     list(channel_dir.glob("video_*.webm"))
        
        for video_path in video_files:
            # Corresponding audio file name (remove 'video_' prefix)
            audio_stem = video_path.stem.replace('video_', '')
            
            # Check if transcript.json exists for this video
            preprocess_root = base_path / "preprocess" / channel_dir.name

            if not preprocess_root.exists():
                continue

            for possible_dir in preprocess_root.iterdir():
                if possible_dir.is_dir() and possible_dir.name.startswith(audio_stem):
                    transcript_path = possible_dir / "transcript.json"
                    
                    if transcript_path.exists():
                        print(f"\n  Processing: {video_path.name}")
                        print(f"  For audio: {audio_stem}")
                        
                        # Run VideOCR
                        srt_path = video_path.with_suffix('.srt')
                        # Try Malay first, fallback to English if needed
                        success, error = run_videocr(video_path, srt_path, lang="chi_sim+eng+mal")
                        
                        if not success:
                            print(f"  Malay+Chinese+English OCR failed, trying English...")
                            success, error = run_videocr(video_path, srt_path, lang='eng')
                        
                        if success:
                            # Parse SRT and update transcript.json
                            with open(transcript_path, 'r', encoding='utf-8') as f:
                                transcript_data = json.load(f)
                            
                            # Get total duration from first segment or calculate
                            if transcript_data['segments']:
                                last_seg = transcript_data['segments'][-1]
                                total_duration = last_seg['end_time']
                                
                                # Parse hardcoded subtitles
                                hardcoded_segments = parse_srt_to_segments(srt_path, total_duration)
                                
                                # Map to audio segments
                                updated_segments = map_subs_to_audio_segments(
                                    hardcoded_segments, 
                                    transcript_data['segments']
                                )
                                
                                transcript_data['segments'] = updated_segments
                                transcript_data['has_hardcoded_subs'] = len(hardcoded_segments) > 0
                                
                                # Save updated transcript
                                with open(transcript_path, 'w', encoding='utf-8') as f:
                                    json.dump(transcript_data, f, indent=2, ensure_ascii=False)
                                
                                print(f"  ✓ Updated transcript with hardcoded subtitles")
                        
                        # Optional: Delete video file to save space
                        if video_path.exists():
                            video_path.unlink()
                            print(f"  ✓ Deleted video file to save space")
                        # Optional: Delete SRT file (data is in JSON now)
                        if srt_path.exists():
                            srt_path.unlink()

if __name__ == "__main__":
    # Add multiprocessing guard for Windows compatibility
    process_channel_videos()