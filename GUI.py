import os
os.environ['TK_SILENCE_DEPRECATION'] = '1'

import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path
import cv2
from PIL import Image, ImageTk

from src.stego_video import embed_message, extract_message
from src.video_io import read_video_frames


# 🎨 THEME
BG = "#FFF6F9"
CARD = "#FFFFFF"
PRIMARY = "#FFB6C1"
SECONDARY = "#B5EAD7"
TEXT = "#333333"


class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Steganografi Video AVI 🌸")
        self.root.configure(bg=BG)
        self.root.geometry("1000x650")

        self.frames = []
        self.idx = 0
        self.playing = False

        self.cover = ""
        self.message = ""
        self.output = ""

        # ===== HEADER =====
        header = tk.Label(root, text="Steganografi Video AVI",
                          font=("Helvetica", 20, "bold"),
                          bg=BG, fg=TEXT)
        header.pack(pady=10)

        # ===== MAIN CONTAINER =====
        container = tk.Frame(root, bg=BG)
        container.pack(fill="both", expand=True, padx=10, pady=5)

        container.grid_columnconfigure(0, weight=1)
        container.grid_columnconfigure(1, weight=2)

        # ===== LEFT PANEL =====
        left = tk.Frame(container, bg=CARD, bd=1, relief="solid")
        left.grid(row=0, column=0, sticky="nsew", padx=10, pady=5)

        # ===== RIGHT PANEL =====
        right = tk.Frame(container, bg=CARD, bd=1, relief="solid")
        right.grid(row=0, column=1, sticky="nsew", padx=10, pady=5)

        # ================= LEFT CONTENT =================
        self.build_left(left)

        # ================= RIGHT CONTENT =================
        self.build_right(right)

        # ===== OUTPUT LOG =====
        self.log = tk.Text(root, height=8, bg=CARD)
        self.log.pack(fill="x", padx=10, pady=10)

    # ================= LEFT UI =================
    def build_left(self, parent):
        tk.Label(parent, text="Embed Settings",
                 font=("Helvetica", 14, "bold"),
                 bg=CARD).pack(pady=10)

        self.btn(parent, "Load Video", self.load_video)
        self.btn(parent, "Load Message", self.load_message)
        self.btn(parent, "Save Output", self.save_output)

        # Encryption
        self.enc = tk.BooleanVar()
        tk.Checkbutton(parent, text="Use Encryption (A5/1)",
                       variable=self.enc, bg=CARD).pack(anchor="w", padx=10)

        self.key_entry = tk.Entry(parent)
        self.key_entry.pack(fill="x", padx=10, pady=5)

        # Random
        self.rand = tk.BooleanVar()
        tk.Checkbutton(parent, text="Random Embedding",
                       variable=self.rand, bg=CARD).pack(anchor="w", padx=10)

        self.stego_entry = tk.Entry(parent)
        self.stego_entry.pack(fill="x", padx=10, pady=5)

        # Buttons
        tk.Button(parent, text="EMBED", bg=SECONDARY,
                  command=self.embed).pack(fill="x", padx=10, pady=10)

        tk.Button(parent, text="EXTRACT", bg=PRIMARY,
                  command=self.extract).pack(fill="x", padx=10, pady=5)

    # ================= RIGHT UI =================
    def build_right(self, parent):
        self.video_label = tk.Label(parent, bg="black", width=500, height=350)
        self.video_label.pack(pady=10)

        control = tk.Frame(parent, bg=CARD)
        control.pack()

        tk.Button(control, text="▶ Play", command=self.play).grid(row=0, column=0, padx=5)
        tk.Button(control, text="⏸ Pause", command=self.pause).grid(row=0, column=1, padx=5)

    # ================= BUTTON STYLE =================
    def btn(self, parent, text, cmd):
        tk.Button(parent, text=text, bg=PRIMARY,
                  command=cmd).pack(fill="x", padx=10, pady=3)

    # ================= VIDEO =================
    def load_video(self):
        self.cover = filedialog.askopenfilename(filetypes=[("AVI", "*.avi")])
        if not self.cover:
            return

        self.frames, _ = read_video_frames(self.cover)

        if not self.frames:
            messagebox.showerror("Error", "Video gagal dibaca")
            return

        self.idx = 0
        self.show_frame()

    def show_frame(self):
        if not self.frames:
            return

        frame = self.frames[self.idx]

        # resize biar fit
        frame = cv2.resize(frame, (500, 300))

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = ImageTk.PhotoImage(Image.fromarray(img))

        self.video_label.configure(image=img)
        self.video_label.image = img

    def play(self):
        if not self.frames:
            messagebox.showwarning("Warning", "Load video dulu!")
            return

        self.playing = True
        self.loop()

    def pause(self):
        self.playing = False

    def loop(self):
        if not self.playing or not self.frames:
            return

        self.idx = (self.idx + 1) % len(self.frames)
        self.show_frame()
        self.root.after(30, self.loop)

    # ================= EMBED =================
    def embed(self):
        try:
            if not self.cover or not self.message or not self.output:
                raise Exception("Lengkapi semua input dulu!")

            with open(self.message, "rb") as f:
                msg = f.read()

            result = embed_message(
                cover_path=self.cover,
                output_path=self.output,
                message=msg,
                is_text=self.message.endswith(".txt"),
                extension=Path(self.message).suffix,
                filename=Path(self.message).name,
                use_encryption=self.enc.get(),
                a51_key=int(self.key_entry.get(), 16) if self.enc.get() else None,
                use_random=self.rand.get(),
                stego_key=int(self.stego_entry.get()) if self.rand.get() else None
            )

            self.log.insert(tk.END,
                f"✔ Embed OK | PSNR: {result['psnr_avg']:.2f} | MSE: {result['mse_avg']:.4f}\n")

            self.frames, _ = read_video_frames(self.output)
            self.idx = 0
            self.show_frame()

        except Exception as e:
            messagebox.showerror("Error", str(e))

    # ================= EXTRACT =================
    def extract(self):
        try:
            path = filedialog.askopenfilename(filetypes=[("AVI", "*.avi")])
            if not path:
                return

            result = extract_message(
                stego_path=path,
                a51_key=int(self.key_entry.get(), 16) if self.enc.get() else None,
                stego_key=int(self.stego_entry.get()) if self.rand.get() else None
            )

            if result["is_text"]:
                self.log.insert(tk.END, f"📩 {result['message'].decode()}\n")
            else:
                save = filedialog.asksaveasfilename(
                    defaultextension=result["extension"],
                    initialfile=result["filename"]
                )
                with open(save, "wb") as f:
                    f.write(result["message"])

                self.log.insert(tk.END, f"✔ File saved: {save}\n")

        except Exception as e:
            messagebox.showerror("Error", str(e))


if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()