import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import cv2
from PIL import Image, ImageTk
import numpy as np
import os

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

from src.stego_video import embed_message, extract_message
from src.video_io import read_video_frames, color_histogram_video


BG = "#0b1220"
CARD = "#111827"
TEXT = "white"
ACCENT = "#2563eb"


class StegoApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Video Steganography")
        self.root.geometry("1200x720")
        self.root.configure(bg=BG)

        self.video_frames = []
        self.current_frame = 0
        self.playing = False
        self.video_selected = False

        self.setup_ui()

    # ================= UI =================
    def setup_ui(self):
        container = tk.Frame(self.root, bg=BG)
        container.pack(fill="both", expand=True)

        title = tk.Label(container, text="Video Steganography", font=("Arial", 24, "bold"), bg=BG, fg=TEXT)
        title.pack(pady=10)

        notebook = ttk.Notebook(container)
        notebook.pack(fill="both", expand=True)

        self.embed_tab = tk.Frame(notebook, bg=BG)
        self.extract_tab = tk.Frame(notebook, bg=BG)

        notebook.add(self.embed_tab, text="Encode")
        notebook.add(self.extract_tab, text="Decode")

        self.build_embed_ui()
        self.build_extract_ui()

    # ================= CARD =================
    def card(self, parent):
        frame = tk.Frame(parent, bg=CARD, bd=0, highlightthickness=1, highlightbackground="#1f2937")
        return frame

    # ================= EMBED =================
    def build_embed_ui(self):
        frame = tk.Frame(self.embed_tab, bg=BG)
        frame.pack(fill="both", expand=True, padx=20, pady=10)

        # ---------- UPLOAD ----------
        upload = self.card(frame)
        upload.pack(fill="x", pady=10)

        tk.Label(upload, text="Upload Video", bg=CARD, fg=TEXT).pack(anchor="w", padx=10, pady=5)

        self.select_btn = tk.Button(upload, text="Browse files", command=self.toggle_video)
        self.select_btn.pack(pady=5)

        self.video_label = tk.Label(upload, text="No file selected", bg=CARD, fg="gray")
        self.video_label.pack(pady=5)

        # ---------- MESSAGE ----------
        msg_card = self.card(frame)
        msg_card.pack(fill="x", pady=10)

        tk.Label(msg_card, text="Secret Message", bg=CARD, fg=TEXT).pack(anchor="w", padx=10)

        self.message_entry = tk.Text(msg_card, height=4)
        self.message_entry.pack(fill="x", padx=10, pady=5)

        # ---------- OPTIONS ----------
        opt = self.card(frame)
        opt.pack(fill="x", pady=10)

        self.encrypt_var = tk.BooleanVar()
        tk.Checkbutton(opt, text="Use Encryption", variable=self.encrypt_var, bg=CARD, fg=TEXT).pack(anchor="w", padx=10)

        self.key_entry = tk.Entry(opt)
        self.key_entry.pack(fill="x", padx=10, pady=5)

        self.random_var = tk.BooleanVar()
        tk.Checkbutton(opt, text="Random Mode", variable=self.random_var, bg=CARD, fg=TEXT).pack(anchor="w", padx=10)

        self.stego_key_entry = tk.Entry(opt)
        self.stego_key_entry.pack(fill="x", padx=10, pady=5)

        tk.Button(frame, text="Encode", bg=ACCENT, fg="white", command=self.run_embed).pack(pady=10)

        # ---------- METRICS ----------
        self.metrics_label = tk.Label(frame, text="", bg=BG, fg=TEXT)
        self.metrics_label.pack()

        # ---------- VIDEO ----------
        self.video_frame = self.card(frame)
        self.video_frame.pack(fill="both", expand=True, pady=10)

        self.canvas = tk.Label(self.video_frame)
        self.canvas.pack()

        controls = tk.Frame(self.video_frame, bg=CARD)
        controls.pack()

        tk.Button(controls, text="Play", command=self.play_video).pack(side="left")
        tk.Button(controls, text="Pause", command=self.pause_video).pack(side="left")

        # ---------- CHART ----------
        self.chart_frame = tk.Frame(frame, bg=BG)
        self.chart_frame.pack(fill="both", expand=True)

    # ================= VIDEO =================
    def toggle_video(self):
        if not self.video_selected:
            path = filedialog.askopenfilename(filetypes=[("Video", "*.avi *.mp4")])
            if path:
                self.video_path = path
                self.video_label.config(text=os.path.basename(path))
                self.load_video(path)
                self.play_video()
                self.video_selected = True
                self.select_btn.config(text="Clear")
        else:
            self.video_label.config(text="No file selected")
            self.canvas.config(image="")
            self.video_frames = []
            self.video_selected = False
            self.select_btn.config(text="Browse files")

    def load_video(self, path):
        self.video_frames, self.fps = read_video_frames(path)
        self.current_frame = 0

    def play_video(self):
        self.playing = True
        self.update_frame()

    def pause_video(self):
        self.playing = False

    def update_frame(self):
        if not self.playing or not self.video_frames:
            return

        frame = cv2.cvtColor(self.video_frames[self.current_frame], cv2.COLOR_BGR2RGB)
        img = ImageTk.PhotoImage(Image.fromarray(frame))

        self.canvas.config(image=img)
        self.canvas.image = img

        self.current_frame = (self.current_frame + 1) % len(self.video_frames)
        self.root.after(int(1000/self.fps), self.update_frame)

    # ================= EMBED LOGIC =================
    def run_embed(self):
        try:
            if not hasattr(self, "video_path"):
                raise Exception("Video belum dipilih")

            text = self.message_entry.get("1.0", tk.END).strip()

            if not text:
                raise Exception("Pesan kosong")

            key = int(self.key_entry.get(), 0) if self.encrypt_var.get() else None
            stego_key = int(self.stego_key_entry.get()) if self.random_var.get() else None

            output_path = filedialog.asksaveasfilename(defaultextension=".avi")

            result = embed_message(
                self.video_path,
                output_path,
                text.encode(),
                True,
                ".txt",
                "message.txt",
                self.encrypt_var.get(),
                key,
                self.random_var.get(),
                stego_key
            )

            self.metrics_label.config(
                text=f"MSE: {result['mse_avg']:.4f} | PSNR: {result['psnr_avg']:.2f}"
            )

            cover_frames, _ = read_video_frames(self.video_path)
            stego_frames, _ = read_video_frames(output_path)

            self.show_histogram(cover_frames, stego_frames, result["mse_list"])

        except Exception as e:
            messagebox.showerror("Error", str(e))

    # ================= HISTOGRAM =================
    def show_histogram(self, cover_frames, stego_frames, mse_list):
        hb_c, hg_c, hr_c = color_histogram_video(cover_frames)
        hb_s, hg_s, hr_s = color_histogram_video(stego_frames)

        fig, ax = plt.subplots(1, 3, figsize=(8, 3))
        fig.suptitle(f"MSE avg={np.mean(mse_list):.4f}")

        ax[0].plot(hb_c); ax[0].plot(hb_s, linestyle="--"); ax[0].set_title("B")
        ax[1].plot(hg_c); ax[1].plot(hg_s, linestyle="--"); ax[1].set_title("G")
        ax[2].plot(hr_c); ax[2].plot(hr_s, linestyle="--"); ax[2].set_title("R")

        for w in self.chart_frame.winfo_children():
            w.destroy()

        canvas = FigureCanvasTkAgg(fig, master=self.chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack()

    # ================= EXTRACT =================
    def build_extract_ui(self):
        frame = tk.Frame(self.extract_tab, bg=BG)
        frame.pack(fill="both", expand=True, padx=20, pady=10)

        self.stego_label = tk.Label(frame, text="No file selected", bg=BG, fg="gray")
        self.stego_label.pack()

        tk.Button(frame, text="Browse Video", command=self.select_stego).pack()

        self.extract_key = tk.Entry(frame)
        self.extract_key.pack()

        self.extract_stego_key = tk.Entry(frame)
        self.extract_stego_key.pack()

        tk.Button(frame, text="Decode", bg=ACCENT, fg="white", command=self.run_extract).pack(pady=10)

        self.output_text = tk.Text(frame, height=10, bg="#111", fg="white")
        self.output_text.pack(fill="both", expand=True)

    def select_stego(self):
        self.stego_path = filedialog.askopenfilename(filetypes=[("Video", "*.avi *.mp4")])
        self.stego_label.config(text=os.path.basename(self.stego_path))

    def run_extract(self):
        try:
            self.output_text.delete("1.0", tk.END)
            self.output_text.insert(tk.END, "⏳ Extracting...\n")

            result = extract_message(self.stego_path)

            if result["is_text"]:
                text = result["message"].decode(errors="replace")
                self.output_text.insert(tk.END, f"✅ SUCCESS\n\n{text}")
            else:
                path = f"tests_output/output{result['extension']}"
                with open(path, "wb") as f:
                    f.write(result["message"])
                self.output_text.insert(tk.END, f"Saved to {path}")

        except Exception as e:
            self.output_text.insert(tk.END, f"❌ ERROR: {e}")


if __name__ == "__main__":
    root = tk.Tk()
    app = StegoApp(root)
    root.mainloop()