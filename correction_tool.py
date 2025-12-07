import json
import pygame
import tkinter as tk
from pathlib import Path

class TranscriptionCorrector:
    def __init__(self, transcript_path):
        self.transcript_path = transcript_path
        with open(transcript_path, 'r') as f:
            self.data = json.load(f)
        
        self.current_index = 0
        pygame.mixer.init()
        
        # Create GUI
        self.root = tk.Tk()
        self.setup_gui()
    
    def setup_gui(self):
        # Audio playback controls
        tk.Button(self.root, text="Play", command=self.play_audio).pack()
        
        # Transcription text editor
        self.text_var = tk.StringVar()
        self.text_entry = tk.Entry(self.root, textvariable=self.text_var, width=100)
        self.text_entry.pack()
        
        # Navigation
        tk.Button(self.root, text="Next", command=self.next_segment).pack()
        tk.Button(self.root, text="Save", command=self.save).pack()
        
        self.load_segment()
    
    def play_audio(self):
        segment = self.data["segments"][self.current_index]
        audio_path = self.transcript_path.parent / segment["segment"]
        pygame.mixer.music.load(str(audio_path))
        pygame.mixer.music.play()
    
    def load_segment(self):
        segment = self.data["segments"][self.current_index]
        self.text_var.set(segment["text"])
        self.root.title(f"Segment {self.current_index + 1}/{len(self.data['segments'])}")
    
    def next_segment(self):
        # Save current text
        self.data["segments"][self.current_index]["text"] = self.text_var.get()
        
        # Move to next
        if self.current_index < len(self.data["segments"]) - 1:
            self.current_index += 1
            self.load_segment()
    
    def save(self):
        with open(self.transcript_path, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)
        print("Saved!")

# Run corrector on a specific video
corrector = TranscriptionCorrector("Youtube/preprocess/channel_name/video_name/transcript.json")
corrector.root.mainloop()