import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import threading
import os
from function import analyze_video


class VideoAnalysisApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("📹 Pose-Based Video Analysis")
        
        # Make it full screen and resizable
        self.geometry(f"{self.winfo_screenwidth()}x{self.winfo_screenheight()}")
        self.resizable(True, True)

        self.video_path = None
        self.graph_path = None
        self.annotated_video_path = None
        self.graph_img = None  # keep ref to PhotoImage

        self.create_widgets()

    def create_widgets(self):
        # Title Label
        tk.Label(self, text="📹 Pose-Based Video Analysis", font=("Arial", 20, "bold")).pack(pady=10)

        # Upload Button
        upload_btn = tk.Button(self, text="📁 Select Video File", command=self.browse_video, font=("Arial", 14))
        upload_btn.pack(pady=5)

        # Status Label
        self.status_label = tk.Label(self, text="No video selected.", font=("Arial", 12))
        self.status_label.pack(pady=5)

        # Frame for summary and graph
        self.content_frame = tk.Frame(self)
        self.content_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Summary Frame
        self.summary_frame = tk.LabelFrame(self.content_frame, text="📊 Analysis Summary", font=("Arial", 14))
        self.summary_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=5)

        self.summary_text = tk.Text(self.summary_frame, font=("Arial", 12), state=tk.DISABLED)
        self.summary_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Graph Frame
        self.graph_frame = tk.LabelFrame(self.content_frame, text="📈 Knee Angles Over Time", font=("Arial", 14))
        self.graph_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=5)

        self.graph_label = tk.Label(self.graph_frame)
        self.graph_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Video Preview Frame
        self.video_frame = tk.LabelFrame(self, text="🎥 Annotated Video Preview", font=("Arial", 14))
        self.video_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        self.play_btn = tk.Button(self.video_frame, text="▶ Play Annotated Video", state=tk.DISABLED, command=self.play_video)
        self.play_btn.pack(pady=10)



    def browse_video(self):
        filetypes = [("Video files", "*.mp4 *.avi *.mov *.mkv *.webm"), ("All files", "*.*")]
        path = filedialog.askopenfilename(title="Select Video File", filetypes=filetypes)
        if path:
            self.video_path = path
            self.status_label.config(text=f"Selected video:\n{os.path.basename(path)}")
            self.clear_results()
            self.run_analysis_thread()

    def clear_results(self):
        # Clear previous summary, graph, and disable play button
        self.summary_text.config(state=tk.NORMAL)
        self.summary_text.delete("1.0", tk.END)
        self.summary_text.config(state=tk.DISABLED)
        self.graph_label.config(image="")
        self.graph_img = None
        self.play_btn.config(state=tk.DISABLED)

    def run_analysis_thread(self):
        # Run analyze_video in background thread to avoid freezing UI
        self.status_label.config(text="Analyzing video... Please wait.")
        thread = threading.Thread(target=self.run_analysis)
        thread.start()

    def run_analysis(self):
        try:
            results, graph_path, annotated_video_path, csv_output_path = analyze_video(self.video_path)
        except Exception as e:
            self.status_label.config(text="Error during analysis.")
            messagebox.showerror("Error", f"Failed to analyze video:\n{e}")
            return

        # Update UI on main thread
        self.after(0, self.display_results, results, graph_path, annotated_video_path)

    def display_results(self, results, graph_path, annotated_video_path):
        self.status_label.config(text="✅ Analysis complete!")
        self.graph_path = graph_path
        self.annotated_video_path = annotated_video_path

        # Display summary
        self.summary_text.config(state=tk.NORMAL)
        self.summary_text.delete("1.0", tk.END)
        units = {
            "avg_left_knee_angle": "°",
            "avg_right_knee_angle": "°",
            "max_foot_spread": "units",
            "straight_walk_frames": "frames"
        }
        for key, value in results.items():
            label = key.replace("_", " ").title()
            unit = units.get(key, "")
            self.summary_text.insert(tk.END, f"{label}: {value} {unit}\n")
        self.summary_text.config(state=tk.DISABLED)

        # Display graph image
        if graph_path and os.path.exists(graph_path):
            img = Image.open(graph_path)
            img.thumbnail((600, 400))
            self.graph_img = ImageTk.PhotoImage(img)
            self.graph_label.config(image=self.graph_img)
        else:
            self.graph_label.config(text="Graph image not available")

        # Enable play button if annotated video exists
        if annotated_video_path and os.path.exists(annotated_video_path):
            self.play_btn.config(state=tk.NORMAL)
        else:
            self.play_btn.config(state=tk.DISABLED)

    def play_video(self):
        if self.annotated_video_path and os.path.exists(self.annotated_video_path):
            try:
                # Open video with default system player (Windows)
                os.startfile(self.annotated_video_path)
            except Exception as e:
                messagebox.showerror("Error", f"Could not open video:\n{e}")
        else:
            messagebox.showwarning("Warning", "Annotated video not found.")


if __name__ == "__main__":
    app = VideoAnalysisApp()
    app.mainloop()