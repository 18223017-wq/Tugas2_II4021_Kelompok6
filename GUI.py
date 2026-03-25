import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
import threading
import cv2
from PIL import Image, ImageTk
import numpy as np
import os
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

from src.stego_video import embed_message, extract_message
from src.video_io import read_video_frames, mse_psnr_video, color_histogram_video

class StegoApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Video Steganography Dashboard")
        self.root.geometry("1200x700")
        self.root.configure(bg="#1e1e1e")

        self.video_frames = []
        self.current_frame = 0
        self.playing = False

        self.setup_ui()

    def setup_ui(self):
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill="both", expand=True)

        self.embed_tab = tk.Frame(notebook, bg="#1e1e1e")
        self.extract_tab = tk.Frame(notebook, bg="#1e1e1e")

        notebook.add(self.embed_tab, text="Embed")
        notebook.add(self.extract_tab, text="Extract")

        self.build_embed_ui()
        self.build_extract_ui()

    def show_histogram(self, cover_frames, stego_frames):
        from src.video_io import color_histogram_video

        hb_c, hg_c, hr_c = color_histogram_video(cover_frames)
        hb_s, hg_s, hr_s = color_histogram_video(stego_frames)

        fig, ax = plt.subplots(1, 3, figsize=(10, 3))

        ax[0].plot(hb_c, color='blue', label='Cover')
        ax[0].plot(hb_s, color='cyan', linestyle='--', label='Stego')
        ax[0].set_title("Blue Channel")

        ax[1].plot(hg_c, color='green')
        ax[1].plot(hg_s, color='lime', linestyle='--')
        ax[1].set_title("Green Channel")

        ax[2].plot(hr_c, color='red')
        ax[2].plot(hr_s, color='orange', linestyle='--')
        ax[2].set_title("Red Channel")

        for a in ax:
            a.legend()
            a.set_facecolor("#1e1e1e")

        fig.patch.set_facecolor("#1e1e1e")

        canvas = FigureCanvasTkAgg(fig, master=self.chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack()

    def show_metrics_chart(self, mse_list, psnr_list):
        fig, ax = plt.subplots(1, 2, figsize=(10, 3))

        ax[0].plot(mse_list)
        ax[0].set_title("MSE per Frame")

        ax[1].plot(psnr_list)
        ax[1].set_title("PSNR per Frame")

        for a in ax:
            a.set_facecolor("#1e1e1e")

        fig.patch.set_facecolor("#1e1e1e")

        canvas = FigureCanvasTkAgg(fig, master=self.chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack()

    def build_embed_ui(self):
        frame = self.embed_tab

        tk.Button(frame, text="Select Video", command=self.select_video).pack()
        self.video_label = tk.Label(frame, text="No video selected", bg="#1e1e1e", fg="white")
        self.video_label.pack()

        tk.Label(frame, text="Secret Message", bg="#1e1e1e", fg="white").pack()
        self.message_entry = tk.Text(frame, height=5)
        self.message_entry.pack()

        self.encrypt_var = tk.BooleanVar()
        tk.Checkbutton(frame, text="Use Encryption", variable=self.encrypt_var).pack()

        self.key_entry = tk.Entry(frame)
        self.key_entry.pack()

        self.random_var = tk.BooleanVar()
        tk.Checkbutton(frame, text="Random Mode", variable=self.random_var).pack()

        self.stego_key_entry = tk.Entry(frame)
        self.stego_key_entry.pack()

        tk.Button(frame, text="Embed", command=self.run_embed).pack(pady=10)

        self.metrics_label = tk.Label(frame, text="", bg="#1e1e1e", fg="white")
        self.metrics_label.pack()
        #visualisasi metrics
        self.chart_frame = tk.Frame(frame, bg="#1e1e1e")
        self.chart_frame.pack(pady=10)

        # video canvas
        self.canvas = tk.Label(frame)
        self.canvas.pack()

        controls = tk.Frame(frame, bg="#1e1e1e")
        controls.pack()

        tk.Button(controls, text="Play", command=self.play_video).pack(side="left")
        tk.Button(controls, text="Pause", command=self.pause_video).pack(side="left")

    def build_extract_ui(self):
        frame = self.extract_tab

        self.stego_label = tk.Label(frame, text="No file selected", bg="#1e1e1e", fg="white")
        self.stego_label.pack()

        tk.Button(frame, text="Select Stego Video", command=self.select_stego).pack()

        self.extract_label = tk.Label(frame, text="", bg="#1e1e1e", fg="white")
        self.extract_label.pack()

        tk.Label(frame, text="A5/1 Key (64-bit int / hex)", bg="#1e1e1e", fg="white").pack()
        self.extract_key = tk.Entry(frame)
        self.extract_key.pack()

        tk.Label(frame, text="Stego Key (seed, harus sama seperti embed)", bg="#1e1e1e", fg="white").pack()
        self.extract_stego_key = tk.Entry(frame)
        self.extract_stego_key.pack()

        tk.Button(frame, text="Extract", command=self.run_extract).pack(pady=10)

        self.output_text = tk.Text(frame, height=10, bg="#111", fg="white")
        self.output_text.pack()

        self.output_text.config(state='disabled')

    def set_output_text(self, text):
        self.output_text.config(state="normal")
        self.output_text.delete("1.0", tk.END)
        self.output_text.insert(tk.END, text)
        self.output_text.config(state="disabled")

    def select_video(self):
        self.video_path = filedialog.askopenfilename(filetypes=[("Video", "*.avi *.mp4")])
        self.video_label.config(text=self.video_path)
        #load videonya (kyk media player)
        if self.video_path:
            self.load_video(self.video_path)
            self.play_video()

    def run_embed(self):
        try:
            key = int(self.key_entry.get(), 0) if self.encrypt_var.get() else None
            stego_key = int(self.stego_key_entry.get()) if self.random_var.get() else None

            output_path = filedialog.asksaveasfilename(defaultextension=".avi")

            message_text = self.message_entry.get("1.0", tk.END).strip()

            if message_text.strip() != "":
                msg = message_text.strip().encode()
                is_text = True
                extension = ".txt"
                filename = "message.txt"

            else:
                # ===== FILE MODE =====
                file_path = filedialog.askopenfilename(title="Select file to hide")

                if not file_path:
                    raise ValueError("Tidak ada pesan yang dimasukkan")

                with open(file_path, "rb") as f:
                    msg = f.read()

                is_text = False
                extension = os.path.splitext(file_path)[1]
                filename = os.path.basename(file_path)

            result = embed_message(
            cover_path=self.video_path,
            output_path=output_path,
            message=msg,
            is_text=is_text,
            extension=extension,      
            filename=filename,       
            use_encryption=self.encrypt_var.get(),
            a51_key=key,
            use_random=self.random_var.get(),
            stego_key=stego_key
        )

            self.metrics_label.config(
                text=f"MSE: {result['mse_avg']:.4f} | PSNR: {result['psnr_avg']:.2f}"
            )

            self.load_video(output_path)

        except Exception as e:
            messagebox.showerror("Error", str(e))

        # load cover & stego frames
        cover_frames, _ = read_video_frames(self.video_path)
        stego_frames, _ = read_video_frames(output_path)

        # clear chart lama
        for widget in self.chart_frame.winfo_children():
            widget.destroy()

        # tampilkan chart
        self.show_metrics_chart(result["mse_avg"] if isinstance(result["mse_avg"], list) else [], result["psnr_per_frame"])
        self.show_histogram(cover_frames, stego_frames)

    def select_stego(self):
        self.stego_path = filedialog.askopenfilename(filetypes=[("Video", "*.avi *.mp4")])
        self.stego_label.config(text=self.stego_path)

    def run_extract(self):
        try:
            key = int(self.extract_key.get(), 0) if self.extract_key.get() else None
            stego_key = int(self.extract_stego_key.get()) if self.extract_stego_key.get() else None

            result = extract_message(
                stego_path=self.stego_path,
                a51_key=key,
                stego_key=stego_key
            )

            os.makedirs("tests_output", exist_ok=True)

            base_name = os.path.splitext(os.path.basename(self.stego_path))[0]

            if result["is_text"]:
                text = result["message"].decode(errors="replace")

                # tampilkan di GUI
                self.set_output_text(text)

                # save .txt
                save_path = os.path.join("tests_output", base_name + ".txt")
                with open(save_path, "w", encoding="utf-8") as f:
                    f.write(text)

            else:
                ext = result["extension"] if result["extension"] else ".bin"
                filename = base_name + ext
                save_path = os.path.join("tests_output", filename)

                with open(save_path, "wb") as f:
                    f.write(result["message"])

                # 🔥 tampilkan status di textbox (BUKAN popup)
                self.set_output_text(f"{filename} berhasil disimpan di folder tests_output")

        except Exception as e:
            self.set_output_text(f"ERROR: {str(e)}")

    def load_video(self, path):
        self.video_frames, self.fps = read_video_frames(path)
        self.current_frame = 0
        self.playing = False

    def play_video(self):
        self.playing = True
        self.update_frame()

    def pause_video(self):
        self.playing = False

    def update_frame(self):
        if not self.playing:
            return

        frame = self.video_frames[self.current_frame]
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        img = Image.fromarray(frame)
        imgtk = ImageTk.PhotoImage(img)

        self.canvas.imgtk = imgtk
        self.canvas.configure(image=imgtk)

        self.current_frame = (self.current_frame + 1) % len(self.video_frames)

        self.root.after(int(1000/self.fps), self.update_frame)

if __name__ == "__main__":
    root = tk.Tk()
    app = StegoApp(root)
    root.mainloop()