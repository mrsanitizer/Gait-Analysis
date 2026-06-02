import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import threading
import queue
import os
import cv2
from function import analyze_video

class VideoAnalysisApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("📹 Pose-Based Video Analysis")
        
        self.geometry(f"{self.winfo_screenwidth()}x{self.winfo_screenheight()}")
        self.resizable(True, True)

        self.video_path = None
        self.graph_path = None
        self.annotated_video_path = None
        self.graph_img = None  

        # NEW: Thread-safe queue for background communication
        self.result_queue = queue.Queue()

        self.create_widgets()

    def create_widgets(self):
        tk.Label(self, text="📹 Pose-Based Video Analysis", font=("Arial", 20, "bold")).pack(pady=10)

        upload_btn = tk.Button(self, text="📁 Select Video File", command=self.browse_video, font=("Arial", 14))
        upload_btn.pack(pady=5)

        self.status_label = tk.Label(self, text="No video selected.", font=("Arial", 12))
        self.status_label.pack(pady=5)

        self.content_frame = tk.Frame(self)
        self.content_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.summary_frame = tk.LabelFrame(self.content_frame, text="📊 Analysis Summary", font=("Arial", 14))
        self.summary_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=5)

        self.summary_text = tk.Text(self.summary_frame, font=("Arial", 12), state=tk.DISABLED)
        self.summary_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.graph_frame = tk.LabelFrame(self.content_frame, text="📈 Joint Angles Over Time", font=("Arial", 14))
        self.graph_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=5)

        self.graph_label = tk.Label(self.graph_frame)
        self.graph_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

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
        self.summary_text.config(state=tk.NORMAL)
        self.summary_text.delete("1.0", tk.END)
        self.summary_text.config(state=tk.DISABLED)
        self.graph_label.config(image="")
        self.graph_img = None
        self.play_btn.config(state=tk.DISABLED)

    def run_analysis_thread(self):
        self.status_label.config(text="Analyzing video... Please wait.")
        
        # Start the background work
        thread = threading.Thread(target=self.run_analysis, daemon=True)
        thread.start()
        
        # Start the main UI thread's polling loop
        self.check_queue()

    def run_analysis(self):
        try:
            results, graph_path, annotated_video_path, csv_output_path = analyze_video(self.video_path)
            # Put success data in the box (error is None)
            self.result_queue.put((results, graph_path, annotated_video_path, None))
        except Exception as e:
            # Put error data in the box
            self.result_queue.put((None, None, None, str(e)))

    def check_queue(self):
        try:
            # Check the box without freezing
            results, graph_path, annotated_video_path, error = self.result_queue.get_nowait()
            
            if error:
                self.show_error(error)
            else:
                self.display_results(results, graph_path, annotated_video_path)
                
        except queue.Empty:
            # Box is empty, check again in 100 milliseconds
            self.after(100, self.check_queue)

    def show_error(self, error_msg):
        self.status_label.config(text="Error during analysis.")
        messagebox.showerror("Error", f"Failed to analyze video:\n{error_msg}")

    def display_results(self, results, graph_path, annotated_video_path):
        self.status_label.config(text="✅ Analysis complete!")
        self.graph_path = graph_path
        self.annotated_video_path = annotated_video_path

        self.summary_text.config(state=tk.NORMAL)
        self.summary_text.delete("1.0", tk.END)
        
        units = {
            "avg_left_knee_flexion": "°", "avg_right_knee_flexion": "°",
            "avg_left_hip_flexion": "°", "avg_right_hip_flexion": "°",
            "avg_left_ankle_up": "°", "avg_right_ankle_up": "°",
            "avg_left_ankle_down": "°", "avg_right_ankle_down": "°"
        }
        
        if results:
            for key, value in results.items():
                label = key.replace("_", " ").title()
                unit = units.get(key, "")
                self.summary_text.insert(tk.END, f"{label}: {value}{unit}\n")
        self.summary_text.config(state=tk.DISABLED)

        if graph_path and os.path.exists(graph_path):
            img = Image.open(graph_path)
            img.thumbnail((600, 400))
            self.graph_img = ImageTk.PhotoImage(img)
            self.graph_label.config(image=self.graph_img)
        else:
            self.graph_label.config(text="Graph image not available")

        if annotated_video_path and os.path.exists(annotated_video_path):
            self.play_btn.config(state=tk.NORMAL)
        else:
            self.play_btn.config(state=tk.DISABLED)

    def play_video(self):
        if not self.annotated_video_path or not os.path.exists(self.annotated_video_path):
            messagebox.showwarning("Warning", "Annotated video not found.")
            return

        vid_window = tk.Toplevel(self)
        vid_window.title("Annotated Video Playback")
        vid_label = tk.Label(vid_window)
        vid_label.pack(padx=10, pady=10)

        cap = cv2.VideoCapture(self.annotated_video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        delay = int(1000 / fps) if fps > 0 else 30

        def stream():
            ret, frame = cap.read()
            if ret:
                cv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(cv_img)
                img.thumbnail((800, 600)) 
                
                imgtk = ImageTk.PhotoImage(image=img)
                vid_label.imgtk = imgtk
                vid_label.configure(image=imgtk)
                
                vid_window.after(delay, stream)
            else:
                cap.release()
                vid_window.destroy()

        stream()

if __name__ == "__main__":
    app = VideoAnalysisApp()
    app.mainloop()