# download_audio_video.py
import yt_dlp
import json
from pathlib import Path
import subprocess
import sys

def download_channel(channel_url, output_dir="Youtube"):
    """Download best audio and best video (with subtitles if available) from a channel"""
    
    output_dir = Path(output_dir)
    
    # AUDIO DOWNLOAD (Existing logic - best audio quality)
    audio_ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': str(output_dir / '%(channel)s' / '%(title)s.%(ext)s'),
        'quiet': False,
        'no_warnings': False,
        'extract_audio': True,
        'audio_format': 'best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'm4a',
        }],
        'writesubtitles': False,  # We don't want auto-generated subs
        'writeautomaticsub': False,
        'ignoreerrors': True,
        'nooverwrites': True,
    }
    
    # VIDEO DOWNLOAD (For VideOCR - best video with audio)
    video_ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
        'outtmpl': str(output_dir / '%(channel)s' / 'video_%(title)s.%(ext)s'),
        'quiet': False,
        'no_warnings': False,
        'ignoreerrors': True,
        'nooverwrites': True,
        'writesubtitles': False,
        'writeautomaticsub': False,
        'postprocessors': [],
    }
    
    print(f"Downloading from channel: {channel_url}")
    
    try:
        # Download audio
        print("\n=== DOWNLOADING AUDIO ===")
        with yt_dlp.YoutubeDL(audio_ydl_opts) as ydl:
            ydl.download([channel_url])
        
        # Download video
        print("\n=== DOWNLOADING VIDEO (for subtitle extraction) ===")
        with yt_dlp.YoutubeDL(video_ydl_opts) as ydl:
            ydl.download([channel_url])
            
    except Exception as e:
        print(f"Download error: {e}")

if __name__ == "__main__":
    # Example: python download_audio_video.py "https://www.youtube.com/@ChannelName"
    if len(sys.argv) > 1:
        download_channel(sys.argv[1])
    else:
        print("Usage: python download_audio_video.py <youtube_channel_url>")