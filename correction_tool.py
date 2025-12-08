# correction_tool_waveform.py
import json
import os
from pathlib import Path
import tempfile
import math

import numpy as np
from pydub import AudioSegment
import pygame

import tkinter as tk
from tkinter import messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

# reduce SDL video usage (safer in VMs)
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("GDK_SCALE", "1")
os.environ.setdefault("QT_SCALE_FACTOR", "1")

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
# Main GUI class
# -------------------------
class WaveformCorrector:
    def __init__(self, transcript_path: Path):
        self.transcript_path = Path(transcript_path)
        if not self.transcript_path.exists():
            raise FileNotFoundError(self.transcript_path)

        with open(self.transcript_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

        if "segments" not in self.data or not isinstance(self.data["segments"], list):
            raise ValueError("transcript.json must contain a 'segments' list")

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

        # init pygame mixer for playback
        try:
            pygame.mixer.init()
        except Exception as e:
            print("pygame.mixer init warning:", e)

        # Build UI
        self.root = tk.Tk()
        self.root.geometry("1100x600")
        self.root.title("Waveform Corrector")
        self.setup_ui()
        self.load_segment()

    def setup_ui(self):
        top_frame = tk.Frame(self.root)
        top_frame.pack(fill="x", padx=6, pady=6)

        self.lbl_file = tk.Label(top_frame, text="No file", anchor="w")
        self.lbl_file.pack(side="left", padx=4)

        btn_play = tk.Button(top_frame, text="Play file", command=self.play_full)
        btn_play.pack(side="left", padx=4)

        btn_play_sel = tk.Button(top_frame, text="Play selection", command=self.play_selection)
        btn_play_sel.pack(side="left", padx=4)

        btn_split = tk.Button(top_frame, text="Split selection", command=self.split_selection)
        btn_split.pack(side="left", padx=4)

        btn_delete = tk.Button(top_frame, text="Delete selection", command=self.delete_selection)
        btn_delete.pack(side="left", padx=4)

        btn_prev = tk.Button(top_frame, text="Prev", command=self.prev_segment)
        btn_prev.pack(side="left", padx=4)

        btn_next = tk.Button(top_frame, text="Next", command=self.next_segment)
        btn_next.pack(side="left", padx=4)

        btn_save = tk.Button(top_frame, text="Save transcript", command=self.save_transcript)
        btn_save.pack(side="left", padx=12)

        # Time / selection label
        self.lbl_idx = tk.Label(self.root, text="")
        self.lbl_idx.pack(anchor="w", padx=6)

        # Text editor for transcript
        txt_frame = tk.Frame(self.root)
        txt_frame.pack(fill="x", padx=6, pady=(0,6))
        tk.Label(txt_frame, text="Transcription:").pack(anchor="w")
        self.text_widget = tk.Text(txt_frame, height=4, wrap="word")
        self.text_widget.pack(fill="x")

        # Matplotlib waveform area
        self.fig, self.ax = plt.subplots(figsize=(11, 4))
        self.ax.set_ylabel("Amplitude")
        self.ax.set_xlabel("Time (s)")
        plt.tight_layout()

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill="both", expand=True, padx=6, pady=6)

        # connect mouse events
        self.canvas.mpl_connect("button_press_event", self.on_mouse_press)
        self.canvas.mpl_connect("motion_notify_event", self.on_mouse_move)
        self.canvas.mpl_connect("button_release_event", self.on_mouse_release)

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
        self.text_widget.insert("1.0", seg.get("text") or "")

        # update labels
        total = len(self.data["segments"])
        self.lbl_file.config(text=f"{self.transcript_path.parent.name}/{seg_name}")
        self.lbl_idx.config(text=f"Segment {self.current_index+1}/{total} — duration {ms_to_hms(len(self.audio))}")

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
    
        # ---- FIX: ensure equal length ----
        if len(t) != len(y):
            m = min(len(t), len(y))
            t = t[:m]
            y = y[:m]
    
        self.ax.clear()
        self.ax.plot(t, y, linewidth=0.5)
        self.ax.set_xlim(0, duration_s)
        self.ax.set_ylim(-1.05, 1.05)
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Amplitude")
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
        self.selection = (int(min(self.sel_start, self.sel_end) * 1000), int(max(self.sel_start, self.sel_end) * 1000))
        self.update_selection_lines()

    def on_mouse_release(self, event):
        if not self.dragging:
            return
        self.dragging = False
        # finalize selection
        if event.inaxes == self.ax:
            self.sel_end = max(0.0, min(event.xdata, self.ax.get_xlim()[1]))
            self.selection = (int(min(self.sel_start, self.sel_end) * 1000), int(max(self.sel_start, self.sel_end) * 1000))
            self.update_selection_lines()

    def update_selection_lines(self):
        # remove old lines
        for coll in list(self.ax.collections) + list(self.ax.lines):
            # keep only main waveform line (first line), others remove
            pass
        # simpler: redraw plot and overlay selection rectangle
        self.plot_waveform()
        if self.selection and (self.selection[1] - self.selection[0]) > 0:
            start_s = self.selection[0] / 1000.0
            end_s = self.selection[1] / 1000.0
            self.ax.axvspan(start_s, end_s, color="red", alpha=0.25)
            self.ax.text((start_s + end_s) / 2, 0.9, f"{ms_to_hms(self.selection[0])} - {ms_to_hms(self.selection[1])}", 
                         ha="center", va="center", fontsize=9, bbox=dict(alpha=0.6))
        self.canvas.draw_idle()

    # ---------------------------
    # Playback helpers
    # ---------------------------
    def _play_audiosegment(self, audio_segment: AudioSegment):
        # export to temp wav and play with pygame
        fd, tmp = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        tmp_path = Path(tmp)
        try:
            audio_segment.export(str(tmp_path), format="wav")
            if pygame.mixer.get_init():
                pygame.mixer.music.stop()
            pygame.mixer.music.load(str(tmp_path))
            pygame.mixer.music.play()
            # keep reference so it isn't deleted immediately
            self.wav_temp = tmp_path
        except Exception as e:
            messagebox.showerror("Playback error", str(e))

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

    # ---------------------------
    # Edit operations
    # ---------------------------
    def split_selection(self):
        if not self.selection or self.selection[1] <= self.selection[0]:
            messagebox.showinfo("No selection", "Select a region to split around (click-drag).")
            return

        start_ms, end_ms = self.selection
        # left = 0..start, middle = start..end, right = end..end
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

        # export left and right (we discard the middle which may be music)
        export_wav(left, left_path)
        export_wav(right, right_path)

        # update transcript.json:
        # replace current segment entry by two entries:
        cur_seg = self.data["segments"][self.current_index]
        left_seg = {"segment": left_path.name, "text": cur_seg.get("text", "")}
        right_seg = {"segment": right_path.name, "text": ""}  # empty text for user to fill
        # replace in list
        self.data["segments"].pop(self.current_index)
        self.data["segments"].insert(self.current_index, right_seg)
        self.data["segments"].insert(self.current_index, left_seg)

        # save backup of original audio
        backup = self.audio_path.with_suffix(self.audio_path.suffix + ".bak")
        if not backup.exists():
            self.audio_path.rename(backup)

        messagebox.showinfo("Split complete", f"Saved:\n{left_path.name}\n{right_path.name}\n\nOriginal backed up as {backup.name}")
        # reload to show first (left) new segment
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
            self.audio_path.rename(backup)

        # overwrite file with new_audio
        export_wav(new_audio, self.audio_path)

        # update in-memory audio and replot
        self.audio = new_audio
        messagebox.showinfo("Delete complete", f"Deleted region {ms_to_hms(start_ms)} - {ms_to_hms(end_ms)}. Original backed up as {backup.name}")
        self.load_segment()

    # ---------------------------
    # navigation & save
    # ---------------------------
    def save_transcript(self):
        # copy text back to segment
        text = self.text_widget.get("1.0", tk.END).strip()
        self.data["segments"][self.current_index]["text"] = text
        with open(self.transcript_path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)
        messagebox.showinfo("Saved", f"Saved transcript: {self.transcript_path}")

    def next_segment(self):
        # save current text to data
        self.data["segments"][self.current_index]["text"] = self.text_widget.get("1.0", tk.END).strip()
        if self.current_index < len(self.data["segments"]) - 1:
            self.current_index += 1
            self.load_segment()

    def prev_segment(self):
        self.data["segments"][self.current_index]["text"] = self.text_widget.get("1.0", tk.END).strip()
        if self.current_index > 0:
            self.current_index -= 1
            self.load_segment()

    def on_close(self):
        # cleanup temp file
        try:
            if self.wav_temp and Path(self.wav_temp).exists():
                Path(self.wav_temp).unlink()
        except Exception:
            pass
        pygame.mixer.quit()
        self.root.destroy()

# -------------------------
# CLI chooser (safe for VMs)
# -------------------------
def choose_transcript_cli():
    base = Path("Youtube") / "preprocess"
    if not base.exists():
        print("Error: Youtube/preprocess not found. Run from project root.")
        return None
    transcripts = list(base.rglob("transcript.json"))
    if not transcripts:
        print("No transcript.json found.")
        return None

    print("\nFound transcripts:\n")
    for i, p in enumerate(transcripts):
        print(f"[{i+1}] {p}")
    choice = input("\nEnter number to edit: ").strip()
    try:
        idx = int(choice) - 1
        if 0 <= idx < len(transcripts):
            return transcripts[idx]
    except Exception:
        pass
    return None

# -------------------------
# run
# -------------------------
if __name__ == "__main__":
    chosen = choose_transcript_cli()
    if not chosen:
        print("No transcript chosen. Exiting.")
        raise SystemExit(0)

    app = WaveformCorrector(chosen)
    app.root.protocol("WM_DELETE_WINDOW", app.on_close)
    app.root.mainloop()
