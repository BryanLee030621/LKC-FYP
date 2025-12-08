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
                            
                            # Check if ANY segment has manually verified text (different from Whisper's pattern)
                            # Whisper usually generates complete sentences, while manual edits might be partial
                            has_verified_text = False
                            whisper_generated = False
                            
                            segments = data.get("segments", [])
                            for seg in segments:
                                text = seg.get("text", "").strip()
                                if text:
                                    # Check if this looks like a Whisper-generated transcript
                                    # Whisper usually produces proper sentences with punctuation
                                    if any(punct in text for punct in ['.', '?', '!', ',']) and len(text) > 10:
                                        whisper_generated = True
                                    # If text exists but doesn't look like typical Whisper output,
                                    # or if we've specifically marked it as verified
                                    if seg.get("verified", False) or not whisper_generated:
                                        has_verified_text = True
                                        break
                            
                            mod_time = transcript_path.stat().st_mtime
                            self.status_tracker[str(video_dir)] = {
                                "verified": has_verified_text,
                                "whisper_generated": whisper_generated,
                                "last_modified": mod_time,
                                "segments_verified": 0,
                                "total_segments": len(segments)
                            }
                    except Exception as e:
                        print(f"Error loading {transcript_path}: {e}")
    
    def get_video_info(self, video_dir):
        """Get video information for display"""
        if str(video_dir) not in self.status_tracker:
            return "○ No transcript"
        
        status = self.status_tracker[str(video_dir)]
        
        # Calculate verified segments count
        if str(video_dir) in self.transcripts:
            segments = self.transcripts[str(video_dir)].get("segments", [])
            total = len(segments)
            verified = sum(1 for seg in segments if seg.get("verified", False))
            status["segments_verified"] = verified
            status["total_segments"] = total
        
        # Determine display status
        if status["verified"]:
            display_status = "✓ Verified"
            color = "green"
        elif status["whisper_generated"]:
            display_status = "○ Whisper Generated (Pending)"
            color = "orange"
        else:
            display_status = "○ Empty"
            color = "gray"
        
        # Add segment counts
        display_status = f"{display_status} ({status['segments_verified']}/{status['total_segments']})"
        
        # Add modification time if verified
        if status["verified"]:
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
                    elif status["whisper_generated"]:
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
                    line += f'"{status["whisper_generated"]}","{mod_str}"'
                    lines.append(line)
        
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        
        return report_path

# -------------------------
# Main GUI class
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

        # Text editor for transcript
        txt_frame = tk.LabelFrame(self.root, text="Transcription", padx=10, pady=5)
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

        # load text
        self.text_widget.delete("1.0", tk.END)
        text = seg.get("text") or ""
        self.text_widget.insert("1.0", text)
        self.text_modified = False
        
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
# Main Browser Window
# -------------------------
class TranscriptionBrowser:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Bahasa Rojak Transcription Browser")
        self.root.geometry("1000x700")
        
        # Initialize project manager
        self.project_manager = ProjectManager()
        self.current_corrector = None
        
        self.setup_ui()
        self.refresh_lists()
        
    def setup_ui(self):
        # Header
        header_frame = tk.Frame(self.root, bg="#2c3e50", height=60)
        header_frame.pack(fill="x")
        header_frame.pack_propagate(False)
        
        title = tk.Label(header_frame, text="🎤 Bahasa Rojak Transcription Corrector", 
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
        
        # Now grid INSIDE tree_container
        self.video_tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")
        
        tree_container.grid_rowconfigure(0, weight=1)
        tree_container.grid_columnconfigure(0, weight=1)
    
        self.video_tree.bind("<Double-1>", self.on_video_double_click)
        
        # Right panel: Actions and Info
        right_frame = tk.LabelFrame(main_frame, text="Actions", padx=10, pady=10)
        right_frame.pack(side="right", fill="y", padx=(5, 0))
        
        # Action buttons
        btn_open = tk.Button(right_frame, text="📝 Open Selected", command=self.open_selected_video, 
                           width=20, height=2, bg="#3498db", fg="white", font=("Arial", 10, "bold"))
        btn_open.pack(pady=5)
        
        btn_refresh = tk.Button(right_frame, text="🔄 Refresh List", command=self.refresh_lists,
                              width=20, height=2)
        btn_refresh.pack(pady=5)
        
        btn_export = tk.Button(right_frame, text="📊 Export Progress", command=self.export_progress,
                             width=20, height=2, bg="#27ae60", fg="white")
        btn_export.pack(pady=5)
        
        btn_save_all = tk.Button(right_frame, text="💾 Save All", command=self.save_all_transcripts,
                               width=20, height=2, bg="#e74c3c", fg="white")
        btn_save_all.pack(pady=5)
        
        # Mark all as verified button
        btn_mark_all = tk.Button(right_frame, text="✓ Mark All as Verified", command=self.mark_all_verified,
                               width=20, height=2, bg="#f39c12", fg="white")
        btn_mark_all.pack(pady=5)
        
        # Statistics frame
        stats_frame = tk.LabelFrame(right_frame, text="Statistics", padx=10, pady=10)
        stats_frame.pack(fill="x", pady=20)
        
        self.stats_text = tk.Text(stats_frame, height=8, width=25, font=("Arial", 9))
        self.stats_text.pack(fill="both")
        self.stats_text.config(state=tk.DISABLED)
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(right_frame, variable=self.progress_var, length=200)
        self.progress_bar.pack(pady=10)
        
        self.progress_label = tk.Label(right_frame, text="0% complete", font=("Arial", 9))
        self.progress_label.pack()
    
    def refresh_lists(self):
        """Refresh the channel and video lists"""
        # Clear lists
        self.channel_listbox.delete(0, tk.END)
        self.video_tree.delete(*self.video_tree.get_children())
        
        # Reload project data
        self.project_manager.load_project()
        
        # Populate channels
        for channel in sorted(self.project_manager.channels, key=lambda x: x.name):
            self.channel_listbox.insert(tk.END, channel.name)
        
        # Update statistics
        self.update_statistics()
        
        self.status_bar.config(text=f"Loaded {len(self.project_manager.channels)} channels")
    
    def update_statistics(self):
        """Update statistics display"""
        total_videos = sum(len(videos) for videos in self.project_manager.videos.values())
        
        # Calculate verification statistics
        verified_videos = 0
        whisper_videos = 0
        total_segments = 0
        verified_segments = 0
        
        for status in self.project_manager.status_tracker.values():
            if status["verified"]:
                verified_videos += 1
            elif status["whisper_generated"]:
                whisper_videos += 1
            verified_segments += status["segments_verified"]
            total_segments += status["total_segments"]
        
        # Update progress (based on verified segments)
        if total_segments > 0:
            progress = (verified_segments / total_segments) * 100
            self.progress_var.set(progress)
            self.progress_label.config(text=f"{progress:.1f}% verified ({verified_segments}/{total_segments} segments)")
        else:
            self.progress_var.set(0)
            self.progress_label.config(text="0% verified")
        
        # Update stats text
        self.stats_text.config(state=tk.NORMAL)
        self.stats_text.delete(1.0, tk.END)
        
        stats = f"""📊 Project Statistics
────────────────
Channels: {len(self.project_manager.channels)}
Videos: {total_videos}
✓ Verified: {verified_videos}
○ Whisper Generated: {whisper_videos}
○ Empty: {total_videos - verified_videos - whisper_videos}
────────────────
Total Segments: {total_segments}
✓ Verified Segments: {verified_segments}
○ Pending Segments: {total_segments - verified_segments}"""
        
        self.stats_text.insert(1.0, stats)
        self.stats_text.config(state=tk.DISABLED)
    
    def on_channel_select(self, event):
        """Handle channel selection"""
        selection = self.channel_listbox.curselection()
        if not selection:
            return
        
        channel_name = self.channel_listbox.get(selection[0])
        self.populate_videos(channel_name)
    
    def populate_videos(self, channel_name):
        """Populate videos for selected channel"""
        # Clear video tree
        self.video_tree.delete(*self.video_tree.get_children())
        
        # Get videos for this channel
        video_dirs = self.project_manager.videos.get(channel_name, [])
        
        for video_dir in sorted(video_dirs, key=lambda x: x.name):
            status = self.project_manager.get_video_info(video_dir)
            
            # Determine icon/color based on status
            if "✓ Verified" in status:
                tag = "verified"
            elif "Whisper Generated" in status:
                tag = "whisper"
            else:
                tag = "empty"
            
            # Insert into tree
            item = self.video_tree.insert("", "end", text=video_dir.name, 
                                         values=(status,), tags=(tag,))
        
        # Configure tags for coloring
        self.video_tree.tag_configure("verified", foreground="green")
        self.video_tree.tag_configure("whisper", foreground="orange")
        self.video_tree.tag_configure("empty", foreground="gray")
    
    def filter_videos(self, *args):
        """Filter videos based on search text"""
        search_text = self.search_var.get().lower()
        selection = self.channel_listbox.curselection()
        
        if not selection:
            return
        
        channel_name = self.channel_listbox.get(selection[0])
        video_dirs = self.project_manager.videos.get(channel_name, [])
        
        # Clear and repopulate with filtered results
        self.video_tree.delete(*self.video_tree.get_children())
        
        for video_dir in video_dirs:
            if search_text in video_dir.name.lower():
                status = self.project_manager.get_video_info(video_dir)
                
                # Determine tag
                if "✓ Verified" in status:
                    tag = "verified"
                elif "Whisper Generated" in status:
                    tag = "whisper"
                else:
                    tag = "empty"
                    
                self.video_tree.insert("", "end", text=video_dir.name, 
                                      values=(status,), tags=(tag,))
    
    def get_selected_video(self):
        """Get the currently selected video"""
        selection = self.video_tree.selection()
        if not selection:
            return None
        
        # Get selected channel
        channel_selection = self.channel_listbox.curselection()
        if not channel_selection:
            return None
        
        channel_name = self.channel_listbox.get(channel_selection[0])
        video_name = self.video_tree.item(selection[0])["text"]
        
        # Find the video directory
        for video_dir in self.project_manager.videos[channel_name]:
            if video_dir.name == video_name:
                return video_dir
        
        return None
    
    def open_selected_video(self):
        """Open the selected video in correction tool"""
        video_dir = self.get_selected_video()
        if not video_dir:
            messagebox.showwarning("No Selection", "Please select a video first.")
            return
        
        transcript_path = video_dir / "transcript.json"
        if not transcript_path.exists():
            messagebox.showerror("Not Found", f"transcript.json not found in {video_dir}")
            return
        
        # Close existing corrector if open
        if self.current_corrector:
            try:
                self.current_corrector.on_close()
            except:
                pass
        
        # Open new corrector
        self.current_corrector = WaveformCorrector(transcript_path, self.project_manager)
        self.status_bar.config(text=f"Opened: {video_dir.name}")
    
    def mark_all_verified(self):
        """Mark all Whisper-generated videos as verified (for batch processing)"""
        if not messagebox.askyesno("Confirm", "Mark ALL Whisper-generated videos as verified?\n\nThis will change their status but won't modify the actual transcripts."):
            return
        
        count = 0
        for channel_name, video_dirs in self.project_manager.videos.items():
            for video_dir in video_dirs:
                if str(video_dir) in self.project_manager.status_tracker:
                    status = self.project_manager.status_tracker[str(video_dir)]
                    if status["whisper_generated"] and not status["verified"]:
                        self.project_manager.mark_all_verified(video_dir)
                        count += 1
        
        self.refresh_lists()
        messagebox.showinfo("Complete", f"Marked {count} videos as verified.")
        self.status_bar.config(text=f"Marked {count} videos as verified")
    
    def on_video_double_click(self, event):
        """Handle double-click on video item"""
        self.open_selected_video()
    
    def export_progress(self):
        """Export progress report"""
        report_path = self.project_manager.export_progress_report()
        messagebox.showinfo("Export Complete", f"Progress report exported to:\n{report_path}")
        self.status_bar.config(text=f"Exported progress report to {report_path}")
    
    def save_all_transcripts(self):
        """Save all modified transcripts"""
        saved = self.project_manager.save_all()
        messagebox.showinfo("Save Complete", f"Saved {saved} transcripts.")
        self.refresh_lists()
        self.status_bar.config(text=f"Saved {saved} transcripts")
    
    def run(self):
        """Start the main application"""
        self.root.mainloop()

# -------------------------
# Main entry point
# -------------------------
if __name__ == "__main__":
    # Check if preprocess directory exists
    base_dir = Path("Youtube/preprocess")
    if not base_dir.exists():
        # Ask user to select directory
        selected_dir = filedialog.askdirectory(title="Select preprocess directory")
        if selected_dir:
            base_dir = Path(selected_dir)
        else:
            print("No directory selected. Exiting.")
            raise SystemExit(0)
    
    # Create browser application
    browser = TranscriptionBrowser()
    browser.run()