# correction_tool_waveform_enhanced_fixed.py
import json
import os
import shutil
from pathlib import Path
import tempfile
from datetime import datetime
import threading

import numpy as np
from pydub import AudioSegment
import pygame

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# reduce SDL video usage (safer in VMs)
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

# -------------------------
# Helper audio utilities
# -------------------------
def load_audio_pydub(path: Path) -> AudioSegment:
    return AudioSegment.from_file(path)

def export_wav(audio: AudioSegment, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    audio.export(str(out_path), format="wav")

def ms_to_hms(ms):
    s = ms / 1000.0
    m = int(s // 60)
    s2 = s - m * 60
    return f"{m:02d}:{s2:05.2f}"

# -------------------------
# Project Manager
# -------------------------
class ProjectManager:
    def __init__(self, base_dir="Youtube/preprocess"):
        self.base_dir = Path(base_dir)
        self.channels = []
        self.videos = {}  # channel -> list of videos
        self.transcripts = {}  # video_path -> transcript data
        self.status_tracker = {}  # video_path -> {"verified": bool, "whisper_generated": bool, "last_modified": float}
        self.load_project()
    
    def load_project(self):
        """Scan directory structure and load all transcripts"""
        self.channels = [d for d in self.base_dir.iterdir() if d.is_dir()]

        for channel in self.channels:
            video_dirs = [v for v in channel.iterdir() if v.is_dir()]
            self.videos[channel.name] = video_dirs

            for video_dir in video_dirs:
                transcript_path = video_dir / "transcript.json"
                if transcript_path.exists():
                    try:
                        with open(transcript_path, "r", encoding="utf-8") as f:
                            data = json.load(f)
                            self.transcripts[str(video_dir)] = data

                            segments = data.get("segments", [])

                            # Check for verification status
                            has_verified = False
                            has_any_text = False

                            for seg in segments:
                                text = seg.get("text", "").strip()
                                if text:
                                    has_any_text = True
                                if seg.get("verified", False):
                                    has_verified = True

                            mod_time = transcript_path.stat().st_mtime
                            self.status_tracker[str(video_dir)] = {
                                "verified": has_verified,
                                "has_text": has_any_text,  # Simple: has text or not
                                "last_modified": mod_time,
                                "segments_verified": 0,
                                "total_segments": len(segments)
                            }
                    except Exception as e:
                        print(f"Error loading {transcript_path}: {e}")
                else:
                    # Video folder exists but no transcript.json
                    self.status_tracker[str(video_dir)] = {
                        "verified": False,
                        "has_text": False,
                        "last_modified": 0,
                        "segments_verified": 0,
                        "total_segments": 0
                    }
    
    def get_video_info(self, video_dir):
        """Get video information for display"""
        if str(video_dir) not in self.status_tracker:
            return "○ No transcript"

        status = self.status_tracker[str(video_dir)]

        # Calculate verified segments count
        verified_count = 0
        total_segments = 0

        if str(video_dir) in self.transcripts:
            segments = self.transcripts[str(video_dir)].get("segments", [])
            total_segments = len(segments)
            verified_count = sum(1 for seg in segments if seg.get("verified", False))

            # Update the status tracker
            status["segments_verified"] = verified_count
            status["total_segments"] = total_segments

        # SIMPLIFIED LOGIC:
        if status["verified"]:
            display_status = "✓ Verified"
        elif status["has_text"]:
            display_status = "○ Whisper Generated"
        elif total_segments == 0:
            # No transcript.json file or empty segments array
            display_status = "○ No transcript"
        else:
            # Has segments but all text is empty
            display_status = "○ Empty segments"

        # Add segment counts if we have segments
        if total_segments > 0:
            display_status = f"{display_status} ({verified_count}/{total_segments})"

        # Add modification time if verified
        if status["verified"] and status["last_modified"]:
            mod_str = datetime.fromtimestamp(status["last_modified"]).strftime("%Y-%m-%d %H:%M")
            display_status = f"{display_status} @ {mod_str}"

        return display_status

    def mark_verified(self, video_dir, segment_index=None):
        """Mark a video or specific segment as manually verified"""
        if str(video_dir) in self.status_tracker:
            self.status_tracker[str(video_dir)]["verified"] = True
            self.status_tracker[str(video_dir)]["last_modified"] = datetime.now().timestamp()
            
            # Update segment verification status if specified
            if segment_index is not None and str(video_dir) in self.transcripts:
                segments = self.transcripts[str(video_dir)].get("segments", [])
                if 0 <= segment_index < len(segments):
                    segments[segment_index]["verified"] = True
    
    def mark_all_verified(self, video_dir):
        """Mark all segments in a video as verified"""
        if str(video_dir) in self.status_tracker:
            self.status_tracker[str(video_dir)]["verified"] = True
            self.status_tracker[str(video_dir)]["last_modified"] = datetime.now().timestamp()
            
            # Mark all segments as verified
            if str(video_dir) in self.transcripts:
                segments = self.transcripts[str(video_dir)].get("segments", [])
                for seg in segments:
                    seg["verified"] = True
                # Update verification count
                self.status_tracker[str(video_dir)]["segments_verified"] = len(segments)
    
    def save_all(self):
        """Save all transcripts"""
        saved = 0
        for video_path_str, data in self.transcripts.items():
            video_path = Path(video_path_str)
            transcript_path = video_path / "transcript.json"
            try:
                with open(transcript_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                saved += 1
            except Exception as e:
                print(f"Error saving {transcript_path}: {e}")
        return saved
    
    def export_progress_report(self):
        """Export a progress report CSV"""
        report_path = self.base_dir / "transcription_progress.csv"
        lines = ["Channel,Video,Status,Segments Verified/Total,Whisper Generated,Last Modified"]
        
        for channel_name, video_dirs in self.videos.items():
            for video_dir in video_dirs:
                if str(video_dir) in self.status_tracker:
                    status = self.status_tracker[str(video_dir)]
                    
                    # Determine status text
                    if status["verified"]:
                        status_text = "Verified"
                    elif status.get("whisper_generated", False):
                        status_text = "Whisper Generated"
                    else:
                        status_text = "Empty"
                    
                    # Format last modified
                    if status["last_modified"]:
                        mod_str = datetime.fromtimestamp(status["last_modified"]).strftime("%Y-%m-%d %H:%M")
                    else:
                        mod_str = "N/A"
                    
                    video_name = video_dir.name
                    line = f'"{channel_name}","{video_name}","{status_text}",'
                    line += f'"{status["segments_verified"]}/{status["total_segments"]}",'
                    line += f'"{status.get("whisper_generated", False)}","{mod_str}"'
                    lines.append(line)
        
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        
        return report_path

# -------------------------
# Main Annotation GUI (Waveform Corrector) – full implementation
# -------------------------
class WaveformCorrector:
    def __init__(self, transcript_path: Path, project_manager: ProjectManager):
        self.transcript_path = Path(transcript_path)
        self.project_manager = project_manager
        self.video_dir = self.transcript_path.parent
        
        if not self.transcript_path.exists():
            raise FileNotFoundError(self.transcript_path)

        with open(self.transcript_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

        if "segments" not in self.data or not isinstance(self.data["segments"], list):
            raise ValueError("transcript.json must contain a 'segments' list")

        # Initialize verified flag if not present
        for seg in self.data["segments"]:
            if "verified" not in seg:
                seg["verified"] = False

        # state
        self.current_index = 0
        self.audio = None  # AudioSegment
        self.audio_path = None
        self.sample_rate = 44100
        self.selection = None  # (start_ms, end_ms)
        self.dragging = False
        self.fig = None
        self.ax = None
        self.canvas = None
        self.line_sel = None
        self.wav_temp = None
        self.selection_rect = None
        self.text_modified = False
        self.verification_status = tk.BooleanVar(value=False)

        # Widgets for reference transcriptions
        self.whisper_text_widget = None
        self.qwen_text_widget = None

        # init pygame mixer for playback
        try:
            pygame.mixer.init()
        except Exception as e:
            print("pygame.mixer init warning:", e)

        # Build UI
        self.root = tk.Toplevel()
        self.root.geometry("1100x750")
        self.root.title(f"Waveform Corrector - {self.video_dir.name}")
        self.setup_ui()
        self.load_segment()
        
        # Auto-save on close
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def setup_ui(self):
        # Top control frame
        top_frame = tk.Frame(self.root)
        top_frame.pack(fill="x", padx=6, pady=6)

        # Navigation info
        nav_frame = tk.Frame(top_frame)
        nav_frame.pack(side="left", fill="x", expand=True)
        
        self.lbl_file = tk.Label(nav_frame, text="No file", anchor="w", font=("Arial", 10))
        self.lbl_file.pack(side="top", fill="x", padx=4)
        
        self.lbl_idx = tk.Label(nav_frame, text="", anchor="w", font=("Arial", 9))
        self.lbl_idx.pack(side="top", fill="x", padx=4)

        # Playback controls
        control_frame = tk.Frame(top_frame)
        control_frame.pack(side="right", padx=10)
        
        btn_play = tk.Button(control_frame, text="▶ Play", command=self.play_full, width=8)
        btn_play.grid(row=0, column=0, padx=2)
        
        btn_play_sel = tk.Button(control_frame, text="▶ Sel", command=self.play_selection, width=8)
        btn_play_sel.grid(row=0, column=1, padx=2)
        
        btn_stop = tk.Button(control_frame, text="⏹ Stop", command=self.stop_playback, width=8)
        btn_stop.grid(row=0, column=2, padx=2)

        # Edit controls
        edit_frame = tk.Frame(top_frame)
        edit_frame.pack(side="right", padx=10)
        
        btn_split = tk.Button(edit_frame, text="✂ Split", command=self.split_selection, width=8)
        btn_split.grid(row=0, column=0, padx=2)
        
        btn_delete = tk.Button(edit_frame, text="🗑 Delete", command=self.delete_selection, width=8)
        btn_delete.grid(row=0, column=1, padx=2)

        btn_delete_segment = tk.Button(edit_frame, text="🗑 Del Seg", command=self.delete_current_segment, width=10)
        btn_delete_segment.grid(row=0, column=2, padx=2)

        # Navigation controls
        nav_btn_frame = tk.Frame(self.root)
        nav_btn_frame.pack(fill="x", padx=6, pady=(0, 6))
        
        btn_prev = tk.Button(nav_btn_frame, text="← Prev", command=self.prev_segment, width=10)
        btn_prev.pack(side="left", padx=2)
        
        btn_save = tk.Button(nav_btn_frame, text="💾 Save", command=self.save_transcript, width=10, 
                           bg="#4CAF50", fg="white")
        btn_save.pack(side="left", padx=2)
        
        btn_next = tk.Button(nav_btn_frame, text="Next →", command=self.next_segment, width=10)
        btn_next.pack(side="left", padx=2)
        
        # Verification checkbox
        self.verify_check = tk.Checkbutton(nav_btn_frame, text="✓ Verified", 
                                          variable=self.verification_status,
                                          command=self.on_verification_changed)
        self.verify_check.pack(side="left", padx=20)
        
        # Status indicator
        self.status_label = tk.Label(nav_btn_frame, text="", fg="green")
        self.status_label.pack(side="right", padx=10)

        # --- Reference transcriptions (read‑only) ---
        ref_frame = tk.LabelFrame(self.root, text="Reference Transcriptions", padx=10, pady=5)
        ref_frame.pack(fill="x", padx=6, pady=(0, 6))

        # Left: Whisper
        whisper_frame = tk.Frame(ref_frame)
        whisper_frame.pack(side="left", fill="both", expand=True, padx=(0, 5))

        tk.Label(whisper_frame, text="Whisper Transcription", font=("Arial", 9, "bold")).pack(anchor="w")
        self.whisper_text_widget = tk.Text(whisper_frame, height=3, wrap="word", font=("Arial", 10))
        self.whisper_text_widget.pack(fill="x")
        self.whisper_text_widget.config(state=tk.DISABLED)

        # Right: Qwen
        qwen_frame = tk.Frame(ref_frame)
        qwen_frame.pack(side="right", fill="both", expand=True, padx=(5, 0))

        tk.Label(qwen_frame, text="Qwen Transcription", font=("Arial", 9, "bold")).pack(anchor="w")
        self.qwen_text_widget = tk.Text(qwen_frame, height=3, wrap="word", font=("Arial", 10))
        self.qwen_text_widget.pack(fill="x")
        self.qwen_text_widget.config(state=tk.DISABLED)

        # Gold standard transcription (editable)
        txt_frame = tk.LabelFrame(self.root, text="Gold Standard Transcription", padx=10, pady=5)
        txt_frame.pack(fill="x", padx=6, pady=(0, 6))
        
        self.text_widget = tk.Text(txt_frame, height=4, wrap="word", font=("Arial", 11))
        self.text_widget.pack(fill="x")
        
        # Bind text modification
        self.text_widget.bind("<KeyRelease>", self.on_text_modified)

        # Matplotlib waveform area
        fig_frame = tk.LabelFrame(self.root, text="Waveform (click-drag to select region)", padx=10, pady=5)
        fig_frame.pack(fill="both", expand=True, padx=6, pady=6)
        
        self.fig, self.ax = plt.subplots(figsize=(10, 4))
        self.ax.set_ylabel("Amplitude")
        self.ax.set_xlabel("Time (s)")
        plt.tight_layout()

        self.canvas = FigureCanvasTkAgg(self.fig, master=fig_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill="both", expand=True)

        # connect mouse events
        self.canvas.mpl_connect("button_press_event", self.on_mouse_press)
        self.canvas.mpl_connect("motion_notify_event", self.on_mouse_move)
        self.canvas.mpl_connect("button_release_event", self.on_mouse_release)

    def on_text_modified(self, event=None):
        self.text_modified = True
        self.status_label.config(text="Unsaved changes...", fg="orange")

    def on_verification_changed(self):
        # Update verification status for current segment
        self.data["segments"][self.current_index]["verified"] = self.verification_status.get()
        self.project_manager.mark_verified(self.video_dir, self.current_index)
        self.text_modified = True
        self.status_label.config(text="Verification changed", fg="orange")

    def load_segment(self):
        seg = self.data["segments"][self.current_index]
        seg_name = seg.get("segment")
        self.audio_path = self.transcript_path.parent / seg_name
        if not self.audio_path.exists():
            messagebox.showerror("Audio not found", f"Audio missing: {self.audio_path}")
            return

        # load audio
        self.audio = load_audio_pydub(self.audio_path)
        self.sample_rate = self.audio.frame_rate

        # draw waveform
        self.plot_waveform()

        # load gold text
        self.text_widget.delete("1.0", tk.END)
        text = seg.get("text") or ""
        self.text_widget.insert("1.0", text)
        self.text_modified = False

        # load whisper text (read‑only)
        whisper_text = seg.get("whisper_text", "")
        self.whisper_text_widget.config(state=tk.NORMAL)
        self.whisper_text_widget.delete("1.0", tk.END)
        self.whisper_text_widget.insert("1.0", whisper_text)
        self.whisper_text_widget.config(state=tk.DISABLED)

        # load qwen text (read‑only)
        qwen_text = seg.get("qwen_text", "")
        self.qwen_text_widget.config(state=tk.NORMAL)
        self.qwen_text_widget.delete("1.0", tk.END)
        self.qwen_text_widget.insert("1.0", qwen_text)
        self.qwen_text_widget.config(state=tk.DISABLED)

        # load verification status
        verified = seg.get("verified", False)
        self.verification_status.set(verified)
        
        self.status_label.config(text="", fg="green")

        # update labels
        total = len(self.data["segments"])
        channel = self.video_dir.parent.name
        video = self.video_dir.name
        
        self.lbl_file.config(text=f"{channel} / {video} / {seg_name}")
        
        dur_text = ms_to_hms(len(self.audio))
        verified_text = "✓" if verified else "○"
        self.lbl_idx.config(text=f"Segment {self.current_index+1}/{total} • Duration: {dur_text} • {verified_text}")

        # reset selection
        self.selection = None
        self.update_selection_lines()

    def plot_waveform(self):
        # convert to mono numpy array (downsample for plotting)
        samples = np.array(self.audio.get_array_of_samples())
        if self.audio.channels == 2:
            samples = samples.reshape((-1, 2))
            samples = samples.mean(axis=1)
    
        total_samples = samples.shape[0]
        duration_s = len(self.audio) / 1000.0
    
        # downsample to at most 3000 points
        max_points = 3000
        step = max(1, total_samples // max_points)
    
        y = samples[::step]
        y = y.astype(float)
    
        # normalize (avoid div0)
        maxv = np.max(np.abs(y))
        if maxv != 0:
            y = y / maxv
    
        # create matching time axis
        n = len(y)
        t = np.linspace(0, duration_s, num=n)
    
        # ensure equal length
        if len(t) != len(y):
            m = min(len(t), len(y))
            t = t[:m]
            y = y[:m]
    
        self.ax.clear()
        
        # Color waveform based on verification status
        seg = self.data["segments"][self.current_index]
        color = 'green' if seg.get("verified", False) else 'blue'
        
        self.ax.plot(t, y, linewidth=0.5, color=color, alpha=0.7)
        self.ax.axhline(y=0, color='black', linewidth=0.5, alpha=0.3)
        self.ax.set_xlim(0, duration_s)
        self.ax.set_ylim(-1.05, 1.05)
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Amplitude")
        self.ax.grid(True, alpha=0.3)
        
        # Add segment number in corner with verification status
        verified = seg.get("verified", False)
        status_text = f"Segment {self.current_index+1} {'✓' if verified else '○'}"
        bg_color = "lightgreen" if verified else "yellow"
        
        self.ax.text(0.02, 0.95, status_text, 
                    transform=self.ax.transAxes, fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor=bg_color, alpha=0.5))
        
        self.canvas.draw_idle()

    # ---------------------------
    # Mouse selection handlers
    # ---------------------------
    def on_mouse_press(self, event):
        if event.inaxes != self.ax:
            return
        self.dragging = True
        self.sel_start = max(0.0, event.xdata)
        self.sel_end = self.sel_start
        self.selection = (int(self.sel_start * 1000), int(self.sel_end * 1000))
        self.update_selection_lines()

    def on_mouse_move(self, event):
        if not self.dragging or event.inaxes != self.ax:
            return
        self.sel_end = max(0.0, min(event.xdata, self.ax.get_xlim()[1]))
        self.selection = (int(min(self.sel_start, self.sel_end) * 1000), 
                         int(max(self.sel_start, self.sel_end) * 1000))
        self.update_selection_lines()

    def on_mouse_release(self, event):
        if not self.dragging:
            return
        self.dragging = False
        if event.inaxes == self.ax:
            self.sel_end = max(0.0, min(event.xdata, self.ax.get_xlim()[1]))
            self.selection = (int(min(self.sel_start, self.sel_end) * 1000), 
                             int(max(self.sel_start, self.sel_end) * 1000))
            self.update_selection_lines()

    def update_selection_lines(self):
        # Clear previous selection visualization
        for patch in self.ax.patches:
            patch.remove()
        for text in self.ax.texts[1:]:  # Keep segment number text
            text.remove()
        
        # Draw new selection
        if self.selection and (self.selection[1] - self.selection[0]) > 0:
            start_s = self.selection[0] / 1000.0
            end_s = self.selection[1] / 1000.0
            width = end_s - start_s
            
            # Draw semi-transparent rectangle
            rect = Rectangle((start_s, -1), width, 2, 
                           alpha=0.3, color='red', label='selection')
            self.ax.add_patch(rect)
            
            # Add time labels
            self.ax.text((start_s + end_s) / 2, 0.9, 
                        f"{ms_to_hms(self.selection[0])} - {ms_to_hms(self.selection[1])}", 
                        ha="center", va="center", fontsize=9, 
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        self.canvas.draw_idle()

    # ---------------------------
    # Playback helpers
    # ---------------------------
    def _play_audiosegment(self, audio_segment: AudioSegment):
        def play_thread():
            fd, tmp = tempfile.mkstemp(suffix=".wav")
            os.close(fd)
            tmp_path = Path(tmp)
            try:
                audio_segment.export(str(tmp_path), format="wav")
                if pygame.mixer.get_init():
                    pygame.mixer.music.stop()
                pygame.mixer.music.load(str(tmp_path))
                pygame.mixer.music.play()
                # Wait for playback to finish
                while pygame.mixer.music.get_busy():
                    pygame.time.Clock().tick(10)
                # Cleanup temp file
                tmp_path.unlink()
            except Exception as e:
                print(f"Playback error: {e}")
        
        # Run playback in separate thread to avoid UI freeze
        thread = threading.Thread(target=play_thread)
        thread.daemon = True
        thread.start()

    def play_full(self):
        if self.audio is None:
            return
        self._play_audiosegment(self.audio)

    def play_selection(self):
        if not self.selection or self.selection[1] <= self.selection[0]:
            messagebox.showinfo("No selection", "Please select a region to play (click-drag on waveform).")
            return
        start_ms, end_ms = self.selection
        seg = self.audio[start_ms:end_ms]
        self._play_audiosegment(seg)

    def stop_playback(self):
        try:
            pygame.mixer.music.stop()
        except:
            pass

    # ---------------------------
    # Edit operations
    # ---------------------------
    def split_selection(self):
        if not self.selection or self.selection[1] <= self.selection[0]:
            messagebox.showinfo("No selection", "Select a region to split around (click-drag).")
            return

        start_ms, end_ms = self.selection
        left = self.audio[:start_ms]
        middle = self.audio[start_ms:end_ms]
        right = self.audio[end_ms:]

        # create filenames
        orig_name = self.audio_path.stem
        ext = self.audio_path.suffix or ".wav"
        left_name = f"{orig_name}_part1{ext}"
        right_name = f"{orig_name}_part2{ext}"
        left_path = self.audio_path.parent / left_name
        right_path = self.audio_path.parent / right_name

        # export left and right
        export_wav(left, left_path)
        export_wav(right, right_path)

        # update transcript.json
        cur_seg = self.data["segments"][self.current_index]
        left_seg = {**cur_seg, "segment": left_path.name, "verified": False}
        right_seg = {**cur_seg, "segment": right_path.name, "text": "", "verified": False}
        
        # Replace current segment with two segments
        self.data["segments"].pop(self.current_index)
        self.data["segments"].insert(self.current_index, right_seg)
        self.data["segments"].insert(self.current_index, left_seg)

        # backup original audio
        backup = self.audio_path.with_suffix(self.audio_path.suffix + ".bak")
        if not backup.exists():
            shutil.copy2(self.audio_path, backup)

        messagebox.showinfo("Split complete", 
                          f"Saved:\n{left_path.name}\n{right_path.name}\n\nOriginal backed up as {backup.name}")
        
        # Update project manager
        self.project_manager.mark_verified(self.video_dir)
        
        # Reload to show first (left) new segment
        self.load_segment()

    def delete_selection(self):
        if not self.selection or self.selection[1] <= self.selection[0]:
            messagebox.showinfo("No selection", "Select a region to delete (click-drag).")
            return
        start_ms, end_ms = self.selection
        new_audio = self.audio[:start_ms] + self.audio[end_ms:]

        # backup original
        backup = self.audio_path.with_suffix(self.audio_path.suffix + ".bak")
        if not backup.exists():
            shutil.copy2(self.audio_path, backup)

        # overwrite file with new_audio
        export_wav(new_audio, self.audio_path)

        # Update project manager
        self.project_manager.mark_verified(self.video_dir)
        
        # update in-memory audio and replot
        self.audio = new_audio
        messagebox.showinfo("Delete complete", 
                          f"Deleted region {ms_to_hms(start_ms)} - {ms_to_hms(end_ms)}. Original backed up as {backup.name}")
        self.load_segment()

    def delete_current_segment(self):
        """Delete the current entire segment: audio file and JSON entry"""
        if not messagebox.askyesno("Confirm Delete", f"Delete this entire segment?\n\nAudio: {self.audio_path.name}\nThis action cannot be undone."):
            return

        try:
            # 1. Backup the original audio file before deletion
            backup = self.audio_path.with_suffix(self.audio_path.suffix + ".bak")
            if not backup.exists():
                shutil.copy2(self.audio_path, backup)
            
            # 2. Delete the physical audio file from disk
            self.audio_path.unlink()
            print(f"Deleted audio file: {self.audio_path}")
            
            # 3. Remove the segment entry from the in-memory data list
            deleted_segment = self.data["segments"].pop(self.current_index)
            print(f"Removed segment entry: {deleted_segment.get('segment')}")
            
            # 4. Update project manager status
            self.project_manager.mark_verified(self.video_dir)
            
            # 5. Handle navigation after deletion
            total_remaining = len(self.data["segments"])
            
            if total_remaining == 0:
                # No segments left in this video
                messagebox.showinfo("Last Segment", "All segments deleted. Closing this video.")
                self.on_close()
                return
            elif self.current_index >= total_remaining:
                # If we deleted the last segment, move to the new last one
                self.current_index = total_remaining - 1
            
            # 6. Save the updated transcript (without the deleted segment) to file
            with open(self.transcript_path, "w", encoding="utf-8") as f:
                json.dump(self.data, f, indent=2, ensure_ascii=False)
            
            # 7. Reload the UI to show the now-current segment
            messagebox.showinfo("Delete Complete", 
                              f"Segment deleted successfully.\n{total_remaining} segments remaining.")
            self.load_segment()
            
        except Exception as e:
            messagebox.showerror("Delete Failed", f"Error deleting segment:\n{str(e)}")

    # ---------------------------
    # navigation & save
    # ---------------------------
    def save_transcript(self):
        # copy text back to segment
        text = self.text_widget.get("1.0", tk.END).strip()
        self.data["segments"][self.current_index]["text"] = text
        
        # Save to file
        with open(self.transcript_path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)
        
        # Update project manager
        self.project_manager.transcripts[str(self.video_dir)] = self.data
        self.project_manager.mark_verified(self.video_dir, self.current_index)
        
        self.text_modified = False
        self.status_label.config(text="✓ Saved", fg="green")
        
        # Auto-clear status after 2 seconds
        self.root.after(2000, lambda: self.status_label.config(text=""))

    def next_segment(self):
        # save current text to data
        if self.text_modified:
            text = self.text_widget.get("1.0", tk.END).strip()
            self.data["segments"][self.current_index]["text"] = text
            self.project_manager.mark_verified(self.video_dir, self.current_index)
            self.text_modified = False
        
        if self.current_index < len(self.data["segments"]) - 1:
            self.current_index += 1
            self.load_segment()

    def prev_segment(self):
        if self.text_modified:
            text = self.text_widget.get("1.0", tk.END).strip()
            self.data["segments"][self.current_index]["text"] = text
            self.project_manager.mark_verified(self.video_dir, self.current_index)
            self.text_modified = False
        
        if self.current_index > 0:
            self.current_index -= 1
            self.load_segment()

    def on_close(self):
        # Auto-save if modified
        if self.text_modified:
            if messagebox.askyesno("Unsaved Changes", "Save changes before closing?"):
                self.save_transcript()
        
        # cleanup
        try:
            pygame.mixer.quit()
        except:
            pass
        self.root.destroy()

# -------------------------
# Error Classification GUI with paired annotations (fixed)
# -------------------------
class ErrorClassificationGUI:
    CATEGORIES = [
        "Spelling Error",
        "Language / Translate Error",
        "Named Entity Error",
        "Substitution Error",
        "Particle Error",
        "Merge Error",
        "Deletion Error",
        "Hallucination Error"
    ]
    # Distinct background colors per category
    STYLES = [
        ("Spelling", "lightblue"),
        ("Language", "lightcoral"),
        ("Named Entity", "lightgreen"),
        ("Substitution", "lightsalmon"),
        ("Particle", "lightyellow"),
        ("Merge", "plum"),
        ("Deletion", "lightpink"),
        ("Hallucination", "lightgray")
    ]

    def __init__(self, transcript_path: Path, project_manager: ProjectManager):
        self.transcript_path = Path(transcript_path)
        self.project_manager = project_manager
        self.video_dir = self.transcript_path.parent

        if not self.transcript_path.exists():
            raise FileNotFoundError(self.transcript_path)

        with open(self.transcript_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

        if "segments" not in self.data or not isinstance(self.data["segments"], list):
            raise ValueError("transcript.json must contain a 'segments' list")

        # Ensure error tracking fields exist
        for seg in self.data["segments"]:
            if "error_counts" not in seg:
                seg["error_counts"] = {cat: 0 for cat in self.CATEGORIES}
            if "error_annotations" not in seg:
                seg["error_annotations"] = []  # list of dicts: model, model_start, model_end, gold_start, gold_end, category

        self.current_index = 0
        self.audio = None
        self.audio_path = None
        self.sample_rate = 44100

        # UI widgets
        self.whisper_text = None
        self.qwen_text = None
        self.gold_text = None
        self.count_labels = {}  # category -> label widget

        # Pending pair selection
        self.pending_first = None  # {'widget': widget, 'start': index, 'end': index}
        self.pending_second = None

        # Hover tracking
        self.current_hover_annotation = None

        # Build UI
        self.root = tk.Toplevel()
        self.root.geometry("1200x800")
        self.root.title(f"Error Classification - {self.video_dir.name}")
        self.setup_ui()
        self.load_segment()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def setup_ui(self):
        # Top control (navigation, save, prev/next)
        top_frame = tk.Frame(self.root)
        top_frame.pack(fill="x", padx=6, pady=6)

        self.lbl_idx = tk.Label(top_frame, text="", font=("Arial", 10))
        self.lbl_idx.pack(side="left", padx=5)

        btn_prev = tk.Button(top_frame, text="← Prev", command=self.prev_segment)
        btn_prev.pack(side="left", padx=2)

        btn_save = tk.Button(top_frame, text="💾 Save", command=self.save_transcript, bg="#4CAF50", fg="white")
        btn_save.pack(side="left", padx=2)

        btn_next = tk.Button(top_frame, text="Next →", command=self.next_segment)
        btn_next.pack(side="left", padx=2)

        self.status_label = tk.Label(top_frame, text="", fg="green")
        self.status_label.pack(side="right", padx=10)

        # Main content: three text boxes
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill="both", expand=True, padx=6, pady=6)

        # Gold (read-only but selectable)
        gold_frame = tk.LabelFrame(main_frame, text="Gold Standard Transcription (selectable)", padx=5, pady=5)
        gold_frame.pack(fill="x", pady=2)
        self.gold_text = tk.Text(gold_frame, height=4, wrap="word", font=("Arial", 11))
        self.gold_text.pack(fill="x")
        self.gold_text.bind("<Button-1>", self.on_click_gold)
        self.gold_text.bind("<ButtonRelease-1>", self.on_select_gold)
        # Block typing
        self.gold_text.bind("<Key>", lambda e: "break")
        self.gold_text.bind("<Motion>", self.on_motion)

        # Whisper & Qwen side by side
        model_frame = tk.Frame(main_frame)
        model_frame.pack(fill="both", expand=True, pady=5)

        # Whisper (selectable)
        whisper_frame = tk.LabelFrame(model_frame, text="Whisper Transcription", padx=5, pady=5)
        whisper_frame.pack(side="left", fill="both", expand=True, padx=(0, 2))
        self.whisper_text = tk.Text(whisper_frame, height=6, wrap="word", font=("Arial", 11))
        self.whisper_text.pack(fill="both", expand=True)
        self.whisper_text.bind("<Button-1>", self.on_click_model)
        self.whisper_text.bind("<ButtonRelease-1>", self.on_select_model)
        self.whisper_text.bind("<Key>", lambda e: "break")  # prevent editing
        self.whisper_text.bind("<Motion>", self.on_motion)

        # Qwen (selectable)
        qwen_frame = tk.LabelFrame(model_frame, text="Qwen Transcription", padx=5, pady=5)
        qwen_frame.pack(side="right", fill="both", expand=True, padx=(2, 0))
        self.qwen_text = tk.Text(qwen_frame, height=6, wrap="word", font=("Arial", 11))
        self.qwen_text.pack(fill="both", expand=True)
        self.qwen_text.bind("<Button-1>", self.on_click_model)
        self.qwen_text.bind("<ButtonRelease-1>", self.on_select_model)
        self.qwen_text.bind("<Key>", lambda e: "break")
        self.qwen_text.bind("<Motion>", self.on_motion)

        # Configure tags for categories
        for cat, color in self.STYLES:
            self.whisper_text.tag_configure(cat, background=color)
            self.qwen_text.tag_configure(cat, background=color)
            self.gold_text.tag_configure(cat, background=color)

        # Temporary highlight tag for hover
        self.whisper_text.tag_configure("hover", background="yellow", borderwidth=2, relief="solid")
        self.qwen_text.tag_configure("hover", background="yellow", borderwidth=2, relief="solid")
        self.gold_text.tag_configure("hover", background="yellow", borderwidth=2, relief="solid")

        # Bottom: category buttons and statistics
        bottom_frame = tk.Frame(self.root)
        bottom_frame.pack(fill="x", padx=6, pady=6)

        # Buttons for each category
        btn_frame = tk.LabelFrame(bottom_frame, text="Error Categories", padx=5, pady=5)
        btn_frame.pack(side="left", fill="both", expand=True)

        # Arrange in two rows of four
        for i, cat in enumerate(self.CATEGORIES):
            row = i // 4
            col = i % 4
            btn = tk.Button(btn_frame, text=cat, command=lambda c=cat: self.assign_category(c),
                            width=20, bg=self.STYLES[i][1])
            btn.grid(row=row, column=col, padx=2, pady=2, sticky="ew")
        btn_frame.grid_columnconfigure(0, weight=1)
        btn_frame.grid_columnconfigure(1, weight=1)
        btn_frame.grid_columnconfigure(2, weight=1)
        btn_frame.grid_columnconfigure(3, weight=1)

        # Statistics frame
        stats_frame = tk.LabelFrame(bottom_frame, text="Counts per Category", padx=5, pady=5)
        stats_frame.pack(side="right", fill="y")

        for i, cat in enumerate(self.CATEGORIES):
            lbl = tk.Label(stats_frame, text=f"{cat[:10]}: 0", anchor="w", width=20)
            lbl.pack(anchor="w")
            self.count_labels[cat] = lbl

        # Remove button
        remove_btn = tk.Button(bottom_frame, text="❌ Remove Error Under Cursor", command=self.remove_annotation,
                               bg="#f44336", fg="white")
        remove_btn.pack(side="bottom", pady=5)

        # Instructions label
        self.instruction = tk.Label(self.root, text="Select a span in Gold, then a span in Whisper/Qwen (or vice versa), then click a category.",
                                     fg="blue")
        self.instruction.pack(side="bottom", pady=2)

    # ------------------------------------------------------------
    # Selection handling
    # ------------------------------------------------------------
    def get_selection(self, widget):
        """Return (start, end) indices of current selection in widget, or None."""
        try:
            sel = widget.tag_ranges(tk.SEL)
            if sel:
                return (widget.index(sel[0]), widget.index(sel[1]))
        except Exception:
            pass
        return None

    def clear_pending(self):
        self.pending_first = None
        self.pending_second = None
        self.instruction.config(text="Select a span in Gold, then a span in Whisper/Qwen (or vice versa).")

    def on_click_gold(self, event):
        pass

    def on_click_model(self, event):
        pass

    def on_select_gold(self, event):
        sel = self.get_selection(self.gold_text)
        if not sel:
            return
        if self.pending_first is None:
            # First selection is gold
            self.pending_first = {'widget': 'gold', 'start': sel[0], 'end': sel[1]}
            self.instruction.config(text="Now select the corresponding span in Whisper or Qwen.")
        elif self.pending_second is None:
            # We already have a first selection; check that it's not gold (should be model)
            if self.pending_first['widget'] != 'gold':
                self.pending_second = {'widget': 'gold', 'start': sel[0], 'end': sel[1]}
                self.instruction.config(text="Ready to assign a category.")
            else:
                # Both selections are gold? Reset
                self.clear_pending()
                self.instruction.config(text="Invalid: both selections are gold. Start over.")
        else:
            # Both already set – ignore
            pass

    def on_select_model(self, event):
        widget = event.widget
        if widget == self.whisper_text:
            model = 'whisper'
        else:
            model = 'qwen'
        sel = self.get_selection(widget)
        if not sel:
            return
        if self.pending_first is None:
            self.pending_first = {'widget': model, 'start': sel[0], 'end': sel[1]}
            self.instruction.config(text="Now select the corresponding span in Gold.")
        elif self.pending_second is None:
            if self.pending_first['widget'] == model:
                # Same model twice? Reset
                self.clear_pending()
                self.instruction.config(text="Invalid: both selections are the same model. Start over.")
            else:
                self.pending_second = {'widget': model, 'start': sel[0], 'end': sel[1]}
                self.instruction.config(text="Ready to assign a category.")
        else:
            pass

    # ------------------------------------------------------------
    # Category assignment
    # ------------------------------------------------------------
    def assign_category(self, category):
        if self.pending_first is None or self.pending_second is None:
            messagebox.showwarning("Incomplete", "Please select two spans (one Gold and one Whisper/Qwen) first.")
            return

        # Determine which is gold and which is model
        gold_sel = None
        model_sel = None
        model_type = None
        for sel in [self.pending_first, self.pending_second]:
            if sel['widget'] == 'gold':
                gold_sel = sel
            else:
                model_sel = sel
                model_type = sel['widget']

        if gold_sel is None or model_sel is None:
            messagebox.showerror("Invalid", "One selection must be Gold and the other a model.")
            self.clear_pending()
            return

        # Apply tags (widgets are already in NORMAL state)
        model_widget = self.whisper_text if model_type == 'whisper' else self.qwen_text
        model_widget.tag_add(category, model_sel['start'], model_sel['end'])
        self.gold_text.tag_add(category, gold_sel['start'], gold_sel['end'])

        # Store annotation
        seg = self.data["segments"][self.current_index]
        annotations = seg.get("error_annotations", [])
        ann = {
            'model': model_type,
            'model_start': model_sel['start'],
            'model_end': model_sel['end'],
            'gold_start': gold_sel['start'],
            'gold_end': gold_sel['end'],
            'category': category
        }
        annotations.append(ann)
        seg["error_annotations"] = annotations
        seg["error_counts"][category] = seg["error_counts"].get(category, 0) + 1

        # Update counts label
        self.count_labels[category].config(text=f"{category[:15]}: {seg['error_counts'][category]}")

        self.clear_pending()
        self.status_label.config(text="Annotation added", fg="green")
        self.root.after(2000, lambda: self.status_label.config(text=""))

    # ------------------------------------------------------------
    # Hover effect using Motion event
    # ------------------------------------------------------------
    def on_motion(self, event):
        widget = event.widget
        index = widget.index("@%d,%d" % (event.x, event.y))
        tags = widget.tag_names(index)
        # Find the first category tag
        cat_tag = None
        for t in tags:
            if t in [c for c, _ in self.STYLES]:
                cat_tag = t
                break

        if not cat_tag:
            # No category under cursor – remove hover if any
            if self.current_hover_annotation is not None:
                self._clear_hover()
            return

        # Find the annotation that contains this position in this widget
        seg = self.data["segments"][self.current_index]
        for ann in seg.get("error_annotations", []):
            if ann['category'] != cat_tag:
                continue
            # Determine if this widget and position match
            if widget == self.gold_text:
                start = ann['gold_start']
                end = ann['gold_end']
            elif widget == self.whisper_text and ann['model'] == 'whisper':
                start = ann['model_start']
                end = ann['model_end']
            elif widget == self.qwen_text and ann['model'] == 'qwen':
                start = ann['model_start']
                end = ann['model_end']
            else:
                continue

            if widget.compare(index, ">=", start) and widget.compare(index, "<=", end):
                # Found the annotation – if it's different from current hover, update
                if self.current_hover_annotation != ann:
                    self._clear_hover()
                    self._apply_hover(ann)
                    self.current_hover_annotation = ann
                return

        # If we get here, no matching annotation found
        self._clear_hover()

    def _clear_hover(self):
        for w in (self.whisper_text, self.qwen_text, self.gold_text):
            w.tag_remove("hover", "1.0", tk.END)
        self.current_hover_annotation = None

    def _apply_hover(self, ann):
        # Apply hover tag to model span
        if ann['model'] == 'whisper':
            self.whisper_text.tag_add("hover", ann['model_start'], ann['model_end'])
        else:
            self.qwen_text.tag_add("hover", ann['model_start'], ann['model_end'])
        # Apply hover tag to gold span
        self.gold_text.tag_add("hover", ann['gold_start'], ann['gold_end'])

    # ------------------------------------------------------------
    # Remove annotation
    # ------------------------------------------------------------
    def remove_annotation(self):
        seg = self.data["segments"][self.current_index]
        annotations = seg.get("error_annotations", [])
        if not annotations:
            return

        # Check each widget for cursor position
        for widget in (self.whisper_text, self.qwen_text, self.gold_text):
            try:
                cursor = widget.index(tk.INSERT)
            except:
                continue
            for ann in annotations:
                # Determine if cursor is inside this annotation in this widget
                if widget == self.gold_text:
                    start = ann['gold_start']
                    end = ann['gold_end']
                elif widget == self.whisper_text and ann['model'] == 'whisper':
                    start = ann['model_start']
                    end = ann['model_end']
                elif widget == self.qwen_text and ann['model'] == 'qwen':
                    start = ann['model_start']
                    end = ann['model_end']
                else:
                    continue

                if widget.compare(cursor, ">=", start) and widget.compare(cursor, "<=", end):
                    # Found it – remove this annotation
                    cat = ann['category']
                    self.gold_text.tag_remove(cat, ann['gold_start'], ann['gold_end'])
                    if ann['model'] == 'whisper':
                        self.whisper_text.tag_remove(cat, ann['model_start'], ann['model_end'])
                    else:
                        self.qwen_text.tag_remove(cat, ann['model_start'], ann['model_end'])

                    annotations.remove(ann)
                    seg["error_annotations"] = annotations
                    seg["error_counts"][cat] = max(0, seg["error_counts"].get(cat, 0) - 1)

                    # Update count label
                    self.count_labels[cat].config(text=f"{cat[:15]}: {seg['error_counts'][cat]}")
                    self.status_label.config(text="Annotation removed", fg="orange")
                    self.root.after(2000, lambda: self.status_label.config(text=""))
                    self._clear_hover()  # remove any lingering hover
                    return

        messagebox.showinfo("No Annotation", "No annotation found at cursor position.")

    # ------------------------------------------------------------
    # Load / save / navigation
    # ------------------------------------------------------------
    def load_segment(self):
        seg = self.data["segments"][self.current_index]

        # Gold text (NORMAL, selectable, non-editable)
        self.gold_text.config(state=tk.NORMAL)
        self.gold_text.delete("1.0", tk.END)
        self.gold_text.insert("1.0", seg.get("text", ""))
        for cat, _ in self.STYLES:
            self.gold_text.tag_remove(cat, "1.0", tk.END)
        # Keep state NORMAL for selection

        # Whisper text (NORMAL, selectable, non-editable)
        whisper_txt = seg.get("whisper_text", "")
        self.whisper_text.config(state=tk.NORMAL)
        self.whisper_text.delete("1.0", tk.END)
        self.whisper_text.insert("1.0", whisper_txt)
        for cat, _ in self.STYLES:
            self.whisper_text.tag_remove(cat, "1.0", tk.END)
        # Keep state NORMAL

        # Qwen text (NORMAL, selectable, non-editable)
        qwen_txt = seg.get("qwen_text", "")
        self.qwen_text.config(state=tk.NORMAL)
        self.qwen_text.delete("1.0", tk.END)
        self.qwen_text.insert("1.0", qwen_txt)
        for cat, _ in self.STYLES:
            self.qwen_text.tag_remove(cat, "1.0", tk.END)
        # Keep state NORMAL

        # Apply stored annotations
        annotations = seg.get("error_annotations", [])
        for ann in annotations:
            cat = ann['category']
            # Gold
            self.gold_text.tag_add(cat, ann['gold_start'], ann['gold_end'])
            # Model
            if ann['model'] == 'whisper':
                self.whisper_text.tag_add(cat, ann['model_start'], ann['model_end'])
            else:
                self.qwen_text.tag_add(cat, ann['model_start'], ann['model_end'])

        # Update count labels
        counts = seg.get("error_counts", {})
        for cat in self.CATEGORIES:
            self.count_labels[cat].config(text=f"{cat[:15]}: {counts.get(cat, 0)}")

        # Update segment index label
        total = len(self.data["segments"])
        self.lbl_idx.config(text=f"Segment {self.current_index+1}/{total}")

        self.clear_pending()
        self._clear_hover()

    def save_transcript(self):
        with open(self.transcript_path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)
        self.project_manager.transcripts[str(self.video_dir)] = self.data
        self.status_label.config(text="✓ Saved", fg="green")
        self.root.after(2000, lambda: self.status_label.config(text=""))

    def next_segment(self):
        if self.current_index < len(self.data["segments"]) - 1:
            self.current_index += 1
            self.load_segment()

    def prev_segment(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.load_segment()

    def on_close(self):
        self.save_transcript()
        self.root.destroy()

# -------------------------
# Main Browser Window (updated)
# -------------------------
class TranscriptionBrowser:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Bahasa Rojak Transcription Browser")
        self.root.geometry("1000x700")
        
        # Initialize project manager
        self.project_manager = ProjectManager()
        self.current_corrector = None
        self.current_error_classifier = None
        
        self.setup_ui()
        self.refresh_lists()
        
    def setup_ui(self):
        # Header
        header_frame = tk.Frame(self.root, bg="#2c3e50", height=60)
        header_frame.pack(fill="x")
        header_frame.pack_propagate(False)
        
        title = tk.Label(header_frame, text="🎤 Bahasa Rojak Transcription Tools", 
                        font=("Arial", 16, "bold"), fg="white", bg="#2c3e50")
        title.pack(side="left", padx=20, pady=10)
        
        # Status bar
        self.status_bar = tk.Label(self.root, text="Ready", bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Main content area
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Left panel: Channels
        left_frame = tk.LabelFrame(main_frame, text="Channels", padx=10, pady=10)
        left_frame.pack(side="left", fill="y", padx=(0, 5))
        
        tk.Label(left_frame, text="Select Channel:", font=("Arial", 10, "bold")).pack(anchor="w", pady=(0, 5))
        
        self.channel_listbox = tk.Listbox(left_frame, width=25, height=20, font=("Arial", 10))
        self.channel_listbox.pack(fill="both", expand=True)
        self.channel_listbox.bind("<<ListboxSelect>>", self.on_channel_select)
        
        # Middle panel: Videos in selected channel
        middle_frame = tk.LabelFrame(main_frame, text="Videos", padx=10, pady=10)
        middle_frame.pack(side="left", fill="both", expand=True, padx=5)
        
        # Video filter/search
        search_frame = tk.Frame(middle_frame)
        search_frame.pack(fill="x", pady=(0, 10))
        
        tk.Label(search_frame, text="Search:").pack(side="left", padx=(0, 5))
        self.search_var = tk.StringVar()
        self.search_var.trace("w", self.filter_videos)
        search_entry = tk.Entry(search_frame, textvariable=self.search_var, width=30)
        search_entry.pack(side="left")
        
        # Container for tree + scrollbars
        tree_container = tk.Frame(middle_frame)
        tree_container.pack(fill="both", expand=True)
        
        # Video list with status
        self.video_tree = ttk.Treeview(tree_container, columns=("status",), show="tree", height=15)
        self.video_tree.heading("#0", text="Video Name")
        self.video_tree.heading("status", text="Status")
        
        vsb = ttk.Scrollbar(tree_container, orient="vertical", command=self.video_tree.yview)
        hsb = ttk.Scrollbar(tree_container, orient="horizontal", command=self.video_tree.xview)
        self.video_tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        
        self.video_tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")
        
        tree_container.grid_rowconfigure(0, weight=1)
        tree_container.grid_columnconfigure(0, weight=1)
    
        self.video_tree.bind("<Double-1>", self.on_video_double_click)
        
        # Right panel: Actions and Info
        right_frame = tk.LabelFrame(main_frame, text="Actions", padx=10, pady=10)
        right_frame.pack(side="right", fill="y", padx=(5, 0))
        
        # Annotation GUI button
        btn_open_annot = tk.Button(right_frame, text="📝 Open Annotation GUI", 
                                    command=self.open_annotation_gui, 
                                    width=25, height=2, bg="#3498db", fg="white", font=("Arial", 10, "bold"))
        btn_open_annot.pack(pady=5)
        
        # Error Classification GUI button
        btn_open_error = tk.Button(right_frame, text="❌ Open Error Classification GUI", 
                                    command=self.open_error_gui, 
                                    width=25, height=2, bg="#e67e22", fg="white", font=("Arial", 10, "bold"))
        btn_open_error.pack(pady=5)
        
        btn_refresh = tk.Button(right_frame, text="🔄 Refresh List", command=self.refresh_lists,
                              width=25, height=2)
        btn_refresh.pack(pady=5)
        
        btn_export = tk.Button(right_frame, text="📊 Export Progress", command=self.export_progress,
                             width=25, height=2, bg="#27ae60", fg="white")
        btn_export.pack(pady=5)
        
        btn_save_all = tk.Button(right_frame, text="💾 Save All", command=self.save_all_transcripts,
                               width=25, height=2, bg="#e74c3c", fg="white")
        btn_save_all.pack(pady=5)
        
        btn_mark_all = tk.Button(right_frame, text="✓ Mark All as Verified", command=self.mark_all_verified,
                               width=25, height=2, bg="#f39c12", fg="white")
        btn_mark_all.pack(pady=5)
        
        # Statistics frame
        stats_frame = tk.LabelFrame(right_frame, text="Statistics", padx=10, pady=10)
        stats_frame.pack(fill="x", pady=20)
        
        self.stats_text = tk.Text(stats_frame, height=8, width=30, font=("Arial", 9))
        self.stats_text.pack(fill="both")
        self.stats_text.config(state=tk.DISABLED)
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(right_frame, variable=self.progress_var, length=250)
        self.progress_bar.pack(pady=10)
        
        self.progress_label = tk.Label(right_frame, text="0% complete", font=("Arial", 9))
        self.progress_label.pack()
    
    def refresh_lists(self):
        self.channel_listbox.delete(0, tk.END)
        self.video_tree.delete(*self.video_tree.get_children())
        self.project_manager.load_project()
        for channel in sorted(self.project_manager.channels, key=lambda x: x.name):
            self.channel_listbox.insert(tk.END, channel.name)
        self.update_statistics()
        self.status_bar.config(text=f"Loaded {len(self.project_manager.channels)} channels")
    
    def update_statistics(self):
        total_videos = sum(len(videos) for videos in self.project_manager.videos.values())
        verified_videos = 0
        whisper_videos = 0
        empty_videos = 0
        total_segments = 0
        verified_segments = 0
        for status in self.project_manager.status_tracker.values():
            if status["verified"]:
                verified_videos += 1
            elif status["has_text"]:
                whisper_videos += 1
            else:
                empty_videos += 1
            verified_segments += status["segments_verified"]
            total_segments += status["total_segments"]
        if total_segments > 0:
            progress = (verified_segments / total_segments) * 100
            self.progress_var.set(progress)
            self.progress_label.config(text=f"{progress:.1f}% verified ({verified_segments}/{total_segments} segments)")
        else:
            self.progress_var.set(0)
            self.progress_label.config(text="0% verified")
        self.stats_text.config(state=tk.NORMAL)
        self.stats_text.delete(1.0, tk.END)
        stats = f"""📊 Project Statistics
    ────────────────
    Channels: {len(self.project_manager.channels)}
    Videos: {total_videos}
    ✓ Verified: {verified_videos}
    ○ Whisper Generated: {whisper_videos}
    ○ Empty/No Transcript: {empty_videos}
    ────────────────
    Total Segments: {total_segments}
    ✓ Verified Segments: {verified_segments}
    ○ Pending Segments: {total_segments - verified_segments}"""
        self.stats_text.insert(1.0, stats)
        self.stats_text.config(state=tk.DISABLED)
    
    def on_channel_select(self, event):
        selection = self.channel_listbox.curselection()
        if not selection:
            return
        channel_name = self.channel_listbox.get(selection[0])
        self.populate_videos(channel_name)
    
    def populate_videos(self, channel_name):
        self.video_tree.delete(*self.video_tree.get_children())
        video_dirs = self.project_manager.videos.get(channel_name, [])
        for video_dir in sorted(video_dirs, key=lambda x: x.name):
            status = self.project_manager.get_video_info(video_dir)
            if "✓ Verified" in status:
                tag = "verified"
            elif "Whisper Generated" in status:
                tag = "whisper"
            else:
                tag = "empty"
            self.video_tree.insert("", "end", text=video_dir.name, values=(status,), tags=(tag,))
        self.video_tree.tag_configure("verified", foreground="green")
        self.video_tree.tag_configure("whisper", foreground="orange")
        self.video_tree.tag_configure("empty", foreground="gray")
    
    def filter_videos(self, *args):
        search_text = self.search_var.get().lower()
        selection = self.channel_listbox.curselection()
        if not selection:
            return
        channel_name = self.channel_listbox.get(selection[0])
        video_dirs = self.project_manager.videos.get(channel_name, [])
        self.video_tree.delete(*self.video_tree.get_children())
        for video_dir in video_dirs:
            if search_text in video_dir.name.lower():
                status = self.project_manager.get_video_info(video_dir)
                if "✓ Verified" in status:
                    tag = "verified"
                elif "Whisper Generated" in status:
                    tag = "whisper"
                else:
                    tag = "empty"
                self.video_tree.insert("", "end", text=video_dir.name, values=(status,), tags=(tag,))
    
    def get_selected_video(self):
        selection = self.video_tree.selection()
        if not selection:
            return None
        channel_selection = self.channel_listbox.curselection()
        if not channel_selection:
            return None
        channel_name = self.channel_listbox.get(channel_selection[0])
        video_name = self.video_tree.item(selection[0])["text"]
        for video_dir in self.project_manager.videos[channel_name]:
            if video_dir.name == video_name:
                return video_dir
        return None
    
    def open_annotation_gui(self):
        video_dir = self.get_selected_video()
        if not video_dir:
            messagebox.showwarning("No Selection", "Please select a video first.")
            return
        transcript_path = video_dir / "transcript.json"
        if not transcript_path.exists():
            messagebox.showerror("Not Found", f"transcript.json not found in {video_dir}")
            return
        if self.current_corrector:
            try:
                self.current_corrector.on_close()
            except:
                pass
        self.current_corrector = WaveformCorrector(transcript_path, self.project_manager)
        self.status_bar.config(text=f"Opened annotation: {video_dir.name}")
    
    def open_error_gui(self):
        video_dir = self.get_selected_video()
        if not video_dir:
            messagebox.showwarning("No Selection", "Please select a video first.")
            return
        transcript_path = video_dir / "transcript.json"
        if not transcript_path.exists():
            messagebox.showerror("Not Found", f"transcript.json not found in {video_dir}")
            return
        if self.current_error_classifier:
            try:
                self.current_error_classifier.on_close()
            except:
                pass
        self.current_error_classifier = ErrorClassificationGUI(transcript_path, self.project_manager)
        self.status_bar.config(text=f"Opened error classification: {video_dir.name}")
    
    def on_video_double_click(self, event):
        self.open_annotation_gui()
    
    def mark_all_verified(self):
        if not messagebox.askyesno("Confirm", "Mark ALL Whisper-generated videos as verified?\n\nThis will change their status but won't modify the actual transcripts."):
            return
        count = 0
        for channel_name, video_dirs in self.project_manager.videos.items():
            for video_dir in video_dirs:
                if str(video_dir) in self.project_manager.status_tracker:
                    status = self.project_manager.status_tracker[str(video_dir)]
                    if status.get("whisper_generated", False) and not status["verified"]:
                        self.project_manager.mark_all_verified(video_dir)
                        count += 1
        self.refresh_lists()
        messagebox.showinfo("Complete", f"Marked {count} videos as verified.")
        self.status_bar.config(text=f"Marked {count} videos as verified")
    
    def export_progress(self):
        report_path = self.project_manager.export_progress_report()
        messagebox.showinfo("Export Complete", f"Progress report exported to:\n{report_path}")
        self.status_bar.config(text=f"Exported progress report to {report_path}")
    
    def save_all_transcripts(self):
        saved = self.project_manager.save_all()
        messagebox.showinfo("Save Complete", f"Saved {saved} transcripts.")
        self.refresh_lists()
        self.status_bar.config(text=f"Saved {saved} transcripts")
    
    def run(self):
        self.root.mainloop()

# -------------------------
# Main entry point
# -------------------------
if __name__ == "__main__":
    base_dir = Path("Youtube/preprocess")
    if not base_dir.exists():
        selected_dir = filedialog.askdirectory(title="Select preprocess directory")
        if selected_dir:
            base_dir = Path(selected_dir)
        else:
            print("No directory selected. Exiting.")
            raise SystemExit(0)
    
    browser = TranscriptionBrowser()
    browser.run()