"""
app.py
────────────────────────────────────────────────────────────────────────────────
YOLO Detection & Tracking UI
  • Tkinter-based GUI – no extra web frameworks needed
  • Choose model from drop-down
  • Open a video file or use webcam
  • Live annotated preview with FPS / stats sidebar
  • All tunable parameters come from config.yaml
────────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import logging
import queue
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageTk

from detection_engine import DetectionEngine, get_available_reid_models
from system_utils import recommend_models, get_available_trackers

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
)
logger = logging.getLogger("app")

CONFIG_PATH = "config.yaml"

UI_FONT = "DejaVu Sans"
UI_SCALE = 1.4
TITLE_SIZE = 32
SUBTITLE_SIZE = 13
CARD_SIZE = 11
TEXT_SIZE = 13
SMALL_TEXT_SIZE = 11
BUTTON_SIZE = 14
PLACEHOLDER_SIZE = 20

# ── Colour palette (dark theme) ───────────────────────────────────────────────
BG       = "#0d0f14"
PANEL    = "#161a24"
ACCENT   = "#00e5a0"
ACCENT2  = "#0099ff"
FG       = "#e8ecf0"
FG_DIM   = "#6b7280"
DANGER   = "#ff4d6d"
CARD     = "#1e2433"
BORDER   = "#252d3d"


# ─────────────────────────────────────────────────────────────────────────────
#  Video Worker Thread
# ─────────────────────────────────────────────────────────────────────────────

class VideoWorker(threading.Thread):
    """
    Background thread: reads frames from a source, passes them to the engine,
    and pushes results into a queue for the GUI to consume.
    """

    def __init__(self, source, engine: DetectionEngine,
                 out_q: queue.Queue, target_fps: int = 30):
        super().__init__(daemon=True)
        self.source = source
        self.engine = engine
        self.out_q = out_q
        self.target_fps = target_fps
        self._stop_evt = threading.Event()

    def stop(self):
        self._stop_evt.set()

    def run(self):
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            self.out_q.put(("error", f"Cannot open source: {self.source}"))
            return

        frame_delay = 1.0 / self.target_fps if self.target_fps > 0 else 0.0
        self.engine.reset_trails()

        while not self._stop_evt.is_set():
            t0 = time.perf_counter()
            ok, frame = cap.read()
            if not ok:
                self.out_q.put(("done", None))
                break

            try:
                annotated, stats = self.engine.process_frame(frame)
                # Keep queue small – drop old frames if UI is slow
                if self.out_q.qsize() < 3:
                    self.out_q.put(("frame", (annotated, stats)))
            except Exception as exc:
                logger.exception("Frame processing error")
                self.out_q.put(("error", str(exc)))
                break

            elapsed = time.perf_counter() - t0
            if frame_delay > elapsed:
                time.sleep(frame_delay - elapsed)

        cap.release()


# ─────────────────────────────────────────────────────────────────────────────
#  Main Application
# ─────────────────────────────────────────────────────────────────────────────

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("YOLO Vision — Detection & Tracking")
        self.configure(bg=BG)
        self.tk.call("tk", "scaling", UI_SCALE)
        self.minsize(1540, 980)
        self.resizable(True, True)

        # Detect system specs and get recommendations
        self.system_spec, self.recommendations = recommend_models(prioritize="balanced")
        logger.info("System detected:\n%s", self.system_spec.summary())

        # State
        self.cfg = DetectionEngine.load_config(CONFIG_PATH)
        self.engine = DetectionEngine(self.cfg)
        
        # Check which Re-ID models are available
        reid_availability = get_available_reid_models()
        available_reid_models = [m for m, avail in reid_availability.items() if avail]
        unavailable_reid_models = [m for m, avail in reid_availability.items() if not avail]
        logger.info("✓ Available Re-ID models: %s", available_reid_models)
        if unavailable_reid_models:
            logger.warning("⚠ Unavailable Re-ID models (will use fallback): %s", unavailable_reid_models)
        
        # Re-ID model options: OSNet, ResNet, FastReID, TransReID, VitREID
        self._reid_model_names = [
            # OSNet variants
            "osnet_ain_x1_0", "osnet_x1_0", "osnet_x0_25",
            # ResNet variants
            "resnet50", "resnet101",
            # FastReID
            "fastraid", "osnet_ibn",
            # TransReID
            "transreid",
            # VitREID
            "vitreid"
        ]
        
        # Tracker options - only use available trackers in this installation
        self._tracker_names = get_available_trackers()
        
        self._worker: Optional[VideoWorker] = None
        self._frame_q: queue.Queue = queue.Queue(maxsize=5)
        self._running = False
        self._video_source = None
        self._after_id = None

        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ── UI Construction ────────────────────────────────────────────────────────

    def _build_ui(self):
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        # ── Left sidebar ───────────────────────────────────────────────────────
        sidebar = tk.Frame(self, bg=PANEL, width=390)
        sidebar.grid(row=0, column=0, sticky="nsew", padx=0, pady=0)
        sidebar.grid_propagate(False)
        sidebar.columnconfigure(0, weight=1)

        # Logo / title
        tk.Label(sidebar, text="YOLO\nVISION", bg=PANEL, fg=ACCENT,
                 font=(UI_FONT, TITLE_SIZE, "bold"), justify="left",
                 padx=20, pady=26).grid(row=0, column=0, sticky="w")

        tk.Label(sidebar, text="Detection & Tracking Studio",
                 bg=PANEL, fg=FG_DIM, font=(UI_FONT, SUBTITLE_SIZE),
                 padx=20).grid(row=1, column=0, sticky="w")

        self._sep(sidebar, 2)

        # ── System Specs & Recommendations ─────────────────────────────────
        self._card_label(sidebar, 3, "SYSTEM SPECS")
        
        # CPU, RAM, GPU info
        spec_lines = [
            f"CPU: {self.system_spec.cpu_cores} cores @ {self.system_spec.cpu_freq:.1f}GHz",
            f"RAM: {self.system_spec.total_ram_gb:.1f}GB",
            f"GPU: {self.system_spec.gpu_name if self.system_spec.has_gpu else 'None'}"
        ]
        spec_text = "\n".join(spec_lines)
        tk.Label(sidebar, text=spec_text, bg=PANEL, fg=FG,
                 font=(UI_FONT, SMALL_TEXT_SIZE), justify="left",
                 padx=16, pady=6, anchor="w").grid(row=4, column=0, sticky="ew")

        # Recommended models
        self._card_label(sidebar, 5, "RECOMMENDATIONS")
        rec_reid = self.recommendations.get("reid", ["None"])[0] if self.recommendations.get("reid") else "None"
        rec_tracker = self.recommendations.get("tracker", ["bytetrack"])[0] if self.recommendations.get("tracker") else "bytetrack"
        
        rec_text = f"Re-ID: {rec_reid}\nTracker: {rec_tracker}"
        tk.Label(sidebar, text=rec_text, bg=CARD, fg=ACCENT,
                 font=(UI_FONT, SMALL_TEXT_SIZE), justify="left",
                 padx=12, pady=8, anchor="w",
                 relief="flat", bd=0).grid(row=6, column=0, padx=16, pady=(0, 8), sticky="ew")

        self._btn(sidebar, 7, "✓ Apply Recommendations", self._apply_recommendations, ACCENT2)

        self._sep(sidebar, 8)

        # ── Model card ────────────────────────────────────────────────────────
        self._card_label(sidebar, 9, "MODEL")
        model_keys = list(self.cfg.get("models", {}).keys())
        self._model_var = tk.StringVar(value=model_keys[0] if model_keys else "")
        model_menu = ttk.Combobox(
            sidebar, textvariable=self._model_var,
            values=model_keys, state="readonly",
            font=(UI_FONT, TEXT_SIZE), width=30,
        )
        model_menu.grid(row=10, column=0, padx=16, pady=(0, 10), sticky="ew")
        self._style_combobox()

        self._btn(sidebar, 11, "⬡  Load Model", self._load_model, ACCENT)

        self._sep(sidebar, 12)

        # ── Source card ───────────────────────────────────────────────────────
        self._card_label(sidebar, 13, "SOURCE")
        self._source_var = tk.StringVar(value="Webcam (device 0)")
        src_entry = tk.Entry(
            sidebar, textvariable=self._source_var,
            bg=CARD, fg=FG, insertbackground=ACCENT,
            font=(UI_FONT, TEXT_SIZE), bd=0, relief="flat",
            highlightthickness=1, highlightbackground=BORDER,
            highlightcolor=ACCENT, width=34,
        )
        src_entry.grid(row=14, column=0, padx=16, pady=(0, 6), sticky="ew")
        self._btn(sidebar, 15, "📂  Browse Video File", self._browse_file, ACCENT2)

        self._sep(sidebar, 16)

        # ── Detection tweaks ─────────────────────────────────────────────────
        self._card_label(sidebar, 17, "DETECTION")

        conf_frame = tk.Frame(sidebar, bg=PANEL)
        conf_frame.grid(row=18, column=0, padx=16, pady=(0, 4), sticky="ew")
        tk.Label(conf_frame, text="Confidence", bg=PANEL, fg=FG_DIM,
                 font=(UI_FONT, SMALL_TEXT_SIZE)).pack(side="left")
        self._conf_var = tk.DoubleVar(
            value=self.cfg["detection"].get("conf_threshold", 0.35))
        self._conf_label = tk.Label(conf_frame, bg=PANEL, fg=ACCENT,
                                    font=(UI_FONT, SMALL_TEXT_SIZE))
        self._conf_label.pack(side="right")
        self._update_conf_label()
        tk.Scale(
            sidebar, from_=0.05, to=1.0, resolution=0.01,
            orient="horizontal", variable=self._conf_var,
            bg=PANEL, fg=FG, troughcolor=CARD, activebackground=ACCENT,
            highlightthickness=0, bd=0, showvalue=False,
            command=lambda _: self._update_conf_label(),
        ).grid(row=19, column=0, padx=14, pady=(0, 6), sticky="ew")

        iou_frame = tk.Frame(sidebar, bg=PANEL)
        iou_frame.grid(row=20, column=0, padx=16, pady=(0, 4), sticky="ew")
        tk.Label(iou_frame, text="IoU Threshold", bg=PANEL, fg=FG_DIM,
                 font=(UI_FONT, SMALL_TEXT_SIZE)).pack(side="left")
        self._iou_var = tk.DoubleVar(
            value=self.cfg["detection"].get("iou_threshold", 0.45))
        self._iou_label = tk.Label(iou_frame, bg=PANEL, fg=ACCENT,
                                   font=(UI_FONT, SMALL_TEXT_SIZE))
        self._iou_label.pack(side="right")
        self._update_iou_label()
        tk.Scale(
            sidebar, from_=0.1, to=1.0, resolution=0.01,
            orient="horizontal", variable=self._iou_var,
            bg=PANEL, fg=FG, troughcolor=CARD, activebackground=ACCENT,
            highlightthickness=0, bd=0, showvalue=False,
            command=lambda _: self._update_iou_label(),
        ).grid(row=21, column=0, padx=14, pady=(0, 8), sticky="ew")

        # Toggles
        toggle_frame = tk.Frame(sidebar, bg=PANEL)
        toggle_frame.grid(row=22, column=0, padx=16, pady=(0, 10), sticky="ew")
        toggle_frame.columnconfigure((0, 1), weight=1)

        self._trail_var = tk.BooleanVar(value=True)
        self._reid_var  = tk.BooleanVar(value=True)
        self._check(toggle_frame, "Show Trail", self._trail_var, 0, 0)
        self._check(toggle_frame, "Re-ID", self._reid_var, 0, 1)

        # ── Re-ID model selector ───────────────────────────────────────────
        self._card_label(sidebar, 23, "RE-ID MODEL")
        self._reid_model_var = tk.StringVar(
            value=self.cfg.get("reid", {}).get("model", self._reid_model_names[0])
        )
        reid_menu = ttk.Combobox(
            sidebar,
            textvariable=self._reid_model_var,
            values=self._reid_model_names,
            state="readonly",
            font=(UI_FONT, TEXT_SIZE),
            width=34,
        )
        reid_menu.grid(row=24, column=0, padx=16, pady=(0, 10), sticky="ew")
        self._style_combobox()

        self._sep(sidebar, 25)

        # ── Tracker selector ───────────────────────────────────────────────
        self._card_label(sidebar, 26, "TRACKER")
        self._tracker_var = tk.StringVar(
            value=self.cfg.get("tracking", {}).get("tracker", self._tracker_names[0])
        )
        tracker_menu = ttk.Combobox(
            sidebar,
            textvariable=self._tracker_var,
            values=self._tracker_names,
            state="readonly",
            font=(UI_FONT, TEXT_SIZE),
            width=34,
        )
        tracker_menu.grid(row=27, column=0, padx=16, pady=(0, 10), sticky="ew")
        self._style_combobox()

        self._sep(sidebar, 28)

        # ── Control buttons ───────────────────────────────────────────────────
        ctrl = tk.Frame(sidebar, bg=PANEL)
        ctrl.grid(row=29, column=0, padx=14, pady=8, sticky="ew")
        ctrl.columnconfigure((0, 1), weight=1)

        self._start_btn = self._btn_grid(ctrl, "▶  START", self._start, ACCENT, 0, 0)
        self._stop_btn  = self._btn_grid(ctrl, "■  STOP",  self._stop,  DANGER, 0, 1,
                                         state="disabled")

        self._sep(sidebar, 30)

        # ── Stats panel ───────────────────────────────────────────────────────
        self._card_label(sidebar, 31, "LIVE STATS")
        self._stat_labels: dict[str, tk.Label] = {}
        for i, (key, label_text) in enumerate([
            ("fps",        "FPS"),
            ("detections", "Detections"),
            ("tracks",     "Active Tracks"),
            ("model",      "Detection Model"),
            ("reid_model",  "Re-ID Model"),
        ]):
            row = 32 + i
            f = tk.Frame(sidebar, bg=PANEL)
            f.grid(row=row, column=0, padx=16, pady=1, sticky="ew")
            tk.Label(f, text=label_text, bg=PANEL, fg=FG_DIM,
                   font=(UI_FONT, SMALL_TEXT_SIZE), width=14, anchor="w").pack(side="left")
            lbl = tk.Label(f, text="—", bg=PANEL, fg=ACCENT,
                       font=(UI_FONT, TEXT_SIZE, "bold"), anchor="e")
            lbl.pack(side="right")
            self._stat_labels[key] = lbl

        # spacer
        tk.Frame(sidebar, bg=PANEL).grid(row=99, column=0, pady=12)

        # ── Video canvas ──────────────────────────────────────────────────────
        canvas_frame = tk.Frame(self, bg=BG)
        canvas_frame.grid(row=0, column=1, sticky="nsew", padx=0, pady=0)
        canvas_frame.rowconfigure(0, weight=1)
        canvas_frame.columnconfigure(0, weight=1)

        self.canvas = tk.Canvas(canvas_frame, bg="#0a0c10", bd=0,
                                highlightthickness=0, relief="flat")
        self.canvas.grid(row=0, column=0, sticky="nsew",
                         padx=12, pady=12)

        # Status bar
        self._status_var = tk.StringVar(value="Ready – load a model and press START")
        tk.Label(canvas_frame, textvariable=self._status_var,
                 bg=PANEL, fg=FG_DIM, font=(UI_FONT, SMALL_TEXT_SIZE),
                 anchor="w", padx=16, pady=6,
                 ).grid(row=1, column=0, sticky="ew")

        self._draw_placeholder()

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _draw_placeholder(self):
        self.canvas.update_idletasks()
        w = self.canvas.winfo_width() or 700
        h = self.canvas.winfo_height() or 500
        self.canvas.delete("all")
        self.canvas.create_text(w // 2, h // 2,
                                 text="No Video\nLoad a model and press ▶ START",
                                 fill=FG_DIM, font=(UI_FONT, PLACEHOLDER_SIZE),
                                 justify="center")

    def _sep(self, parent, row):
        tk.Frame(parent, bg=BORDER, height=1).grid(
            row=row, column=0, sticky="ew", padx=12, pady=8)

    def _card_label(self, parent, row, text):
        tk.Label(parent, text=text, bg=PANEL, fg=FG_DIM,
                 font=(UI_FONT, SMALL_TEXT_SIZE, "bold"),
                 padx=16, pady=2, anchor="w",
                 ).grid(row=row, column=0, sticky="w")

    def _btn(self, parent, row, text, cmd, colour):
        b = tk.Button(
            parent, text=text, command=cmd,
            bg=colour, fg=BG, activebackground=FG, activeforeground=BG,
            font=(UI_FONT, BUTTON_SIZE, "bold"), bd=0, relief="flat",
            padx=10, pady=8, cursor="hand2",
        )
        b.grid(row=row, column=0, padx=16, pady=(0, 8), sticky="ew")
        return b

    def _btn_grid(self, parent, text, cmd, colour, r, c, state="normal"):
        b = tk.Button(
            parent, text=text, command=cmd,
            bg=colour, fg=BG, activebackground=FG, activeforeground=BG,
            font=(UI_FONT, TEXT_SIZE, "bold"), bd=0, relief="flat",
            padx=8, pady=7, cursor="hand2", state=state,
        )
        b.grid(row=r, column=c, padx=3, pady=2, sticky="ew")
        return b

    def _check(self, parent, text, var, r, c):
        cb = tk.Checkbutton(
            parent, text=text, variable=var,
            bg=PANEL, fg=FG, selectcolor=CARD,
            activebackground=PANEL, activeforeground=ACCENT,
            font=(UI_FONT, SMALL_TEXT_SIZE), cursor="hand2",
        )
        cb.grid(row=r, column=c, sticky="w", pady=2)

    def _style_combobox(self):
        style = ttk.Style(self)
        style.theme_use("default")
        style.configure("TCombobox",
                         fieldbackground=CARD, background=CARD,
                         foreground=FG, selectbackground=ACCENT,
                         selectforeground=BG,
                         borderwidth=0, relief="flat",
                         arrowsize=22, padding=8)
        style.map("TCombobox", fieldbackground=[("readonly", CARD)],
                  foreground=[("readonly", FG)])

    def _update_conf_label(self):
        self._conf_label.config(text=f"{self._conf_var.get():.2f}")
        if self.engine:
            self.engine.conf = self._conf_var.get()

    def _update_iou_label(self):
        self._iou_label.config(text=f"{self._iou_var.get():.2f}")
        if self.engine:
            self.engine.iou = self._iou_var.get()

    # ── Actions ───────────────────────────────────────────────────────────────

    def _load_model(self):
        key = self._model_var.get()
        if not key:
            messagebox.showwarning("No model", "Please select a model first.")
            return
        
        reid_model = self._reid_model_var.get()
        if not reid_model:
            messagebox.showwarning("No Re-ID model", "Please select a Re-ID model first.")
            return
            
        self._set_status(f"Loading detection model {key} with Re-ID {reid_model} …")
        self.update_idletasks()
        try:
            # Load Re-ID model first
            self.engine.set_reid_model(reid_model)
            self._stat_labels["reid_model"].config(text=reid_model)
            
            # Load detection model
            self.engine.load_model(key)
            self._stat_labels["model"].config(text=key)
            self._set_status(f"✓ Model '{key}' + Re-ID '{reid_model}' loaded – press ▶ START")
        except (FileNotFoundError, ValueError) as exc:
            messagebox.showerror("Model Error", f"Failed to load model:\n\n{str(exc)}")
            self._set_status("Model load failed.")
        except Exception as exc:
            messagebox.showerror("Unexpected Error", f"Error loading model:\n\n{str(exc)}")
            self._set_status("Model load failed.")

    def _apply_recommendations(self):
        """Apply recommended Re-ID model and tracker."""
        if not self.recommendations.get("reid") or not self.recommendations.get("tracker"):
            messagebox.showwarning("No recommendations", "Could not generate recommendations for your system.")
            return

        rec_reid = self.recommendations["reid"][0]
        rec_tracker = self.recommendations["tracker"][0]

        self._reid_model_var.set(rec_reid)
        self._tracker_var.set(rec_tracker)

        self._set_status(f"✓ Applied recommendations: Re-ID={rec_reid}, Tracker={rec_tracker}")
        messagebox.showinfo(
            "Recommendations Applied",
            f"Set for optimal performance on your system:\n\n"
            f"Re-ID Model: {rec_reid}\n"
            f"Tracker: {rec_tracker}\n\n"
            f"Load a detection model and press ▶ START"
        )

    def _browse_file(self):
        path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv *.flv"),
                ("All files", "*.*"),
            ]
        )
        if path:
            self._source_var.set(path)

    def _resolve_source(self):
        raw = self._source_var.get().strip()
        if raw.lower().startswith("webcam") or raw == "0":
            return 0
        try:
            return int(raw)
        except ValueError:
            return raw

    def _start(self):
        if self.engine._model is None:
            messagebox.showwarning("No Model", "Please load a model first.")
            return
        if self._running:
            return

        # Apply live toggle settings
        self.engine._reid_enabled = self._reid_var.get()
        self.engine.show_trail = self._trail_var.get()
        self.engine.set_reid_model(self._reid_model_var.get())
        self.engine.set_tracker(self._tracker_var.get())

        source = self._resolve_source()
        target_fps = self.cfg.get("video", {}).get("target_fps", 30)

        self._frame_q = queue.Queue(maxsize=5)
        self._worker = VideoWorker(source, self.engine, self._frame_q, target_fps)
        self._worker.start()
        self._running = True

        self._start_btn.config(state="disabled")
        self._stop_btn.config(state="normal")
        self._set_status("Running …")
        self._poll_frames()

    def _stop(self):
        self._running = False
        if self._worker:
            self._worker.stop()
            self._worker = None
        if self._after_id:
            self.after_cancel(self._after_id)
            self._after_id = None
        self._start_btn.config(state="normal")
        self._stop_btn.config(state="disabled")
        self._set_status("Stopped.")
        self._draw_placeholder()

    def _poll_frames(self):
        if not self._running:
            return
        try:
            while True:
                kind, payload = self._frame_q.get_nowait()
                if kind == "frame":
                    frame, stats = payload
                    self._display_frame(frame)
                    self._update_stats(stats)
                elif kind == "done":
                    self._stop()
                    self._set_status("Video ended.")
                    return
                elif kind == "error":
                    messagebox.showerror("Error", payload)
                    self._stop()
                    return
        except queue.Empty:
            pass
        self._after_id = self.after(10, self._poll_frames)

    def _display_frame(self, bgr: np.ndarray):
        self.canvas.update_idletasks()
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw < 2 or ch < 2:
            return

        h, w = bgr.shape[:2]
        scale = min(cw / w, ch / h)
        nw, nh = int(w * scale), int(h * scale)

        resized = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        img = ImageTk.PhotoImage(Image.fromarray(rgb))

        self.canvas.delete("all")
        x_off = (cw - nw) // 2
        y_off = (ch - nh) // 2
        self.canvas.create_image(x_off, y_off, anchor="nw", image=img)
        self.canvas._img_ref = img  # prevent GC

    def _update_stats(self, stats: dict):
        self._stat_labels["fps"].config(text=f"{stats.get('fps', 0):.1f}")
        self._stat_labels["detections"].config(text=str(stats.get("detections", 0)))
        self._stat_labels["tracks"].config(text=str(stats.get("tracks", 0)))

    def _set_status(self, msg: str):
        self._status_var.set(msg)

    def _on_close(self):
        self._stop()
        self.destroy()


# ─────────────────────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = App()
    app.mainloop()
