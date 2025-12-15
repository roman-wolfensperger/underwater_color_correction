#!/usr/bin/env python3
"""
Graphical User Interface for Underwater Video Color Correction
Uses Tkinter for cross-platform GUI with live preview
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
import threading
import sys
import os
import cv2
import numpy as np
from PIL import Image, ImageTk

# Import the color correction script
try:
    from underwater_color_correction import UnderwaterColorCorrector
except ImportError:
    print("Error: underwater_color_correction.py must be in the same directory")
    sys.exit(1)


class UnderwaterColorCorrectionGUI:
    """
    GUI application for underwater video color correction.
    Provides intuitive interface with live preview.
    """
    
    def __init__(self, root):
        """
        Initialize the GUI application.
        
        Args:
            root: Tkinter root window
        """
        self.root = root
        self.root.title("Underwater Video Color Correction")
        self.root.geometry("1200x700")
        self.root.resizable(True, True)
        
        # Variables
        self.input_file = tk.StringVar()
        self.output_file = tk.StringVar()
        self.intensity_var = tk.DoubleVar(value=1.0)
        self.contrast_var = tk.DoubleVar(value=2.0)
        self.denoise_var = tk.BooleanVar(value=False)
        self.processing = False
        
        # Preview variables
        self.original_frame = None
        self.preview_frame = None
        self.photo_original = None
        self.photo_preview = None
        
        # Create GUI elements
        self.create_widgets()
        
        # Center window on screen
        self.center_window()
        
        # Bind slider updates to preview
        self.intensity_var.trace_add('write', self.on_parameter_change)
        self.contrast_var.trace_add('write', self.on_parameter_change)
    
    def center_window(self):
        """Center the window on the screen."""
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f'{width}x{height}+{x}+{y}')
    
    def create_widgets(self):
        """Create all GUI widgets."""
        # Main container
        main_container = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left panel (controls)
        left_frame = ttk.Frame(main_container, width=400)
        main_container.add(left_frame, weight=0)
        
        # Right panel (preview)
        right_frame = ttk.Frame(main_container)
        main_container.add(right_frame, weight=1)
        
        # Create left panel widgets
        self.create_control_panel(left_frame)
        
        # Create right panel widgets
        self.create_preview_panel(right_frame)
    
    def create_control_panel(self, parent):
        """Create the control panel with all settings."""
        # Main frame with scrollbar capability
        canvas = tk.Canvas(parent)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        main_frame = scrollable_frame
        
        # Title
        title_label = ttk.Label(
            main_frame, 
            text="🌊 Underwater Color Correction",
            font=('Helvetica', 14, 'bold')
        )
        title_label.pack(pady=(10, 20))
        
        # ========== FILE SELECTION SECTION ==========
        file_frame = ttk.LabelFrame(main_frame, text="Video Files", padding="10")
        file_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        # Input file
        ttk.Label(file_frame, text="Input Video:").grid(
            row=0, column=0, sticky=tk.W, pady=5
        )
        input_entry = ttk.Entry(
            file_frame, 
            textvariable=self.input_file, 
            width=30
        )
        input_entry.grid(row=1, column=0, padx=5, pady=5, sticky=tk.EW)
        
        browse_input_btn = ttk.Button(
            file_frame, 
            text="Browse...", 
            command=self.browse_input
        )
        browse_input_btn.grid(row=1, column=1, padx=5, pady=5)
        
        # Load preview button
        load_preview_btn = ttk.Button(
            file_frame,
            text="📷 Load Preview",
            command=self.load_preview_frame
        )
        load_preview_btn.grid(row=2, column=0, columnspan=2, pady=5)
        
        # Output file
        ttk.Label(file_frame, text="Output Video:").grid(
            row=3, column=0, sticky=tk.W, pady=5
        )
        output_entry = ttk.Entry(
            file_frame, 
            textvariable=self.output_file, 
            width=30
        )
        output_entry.grid(row=4, column=0, padx=5, pady=5, sticky=tk.EW)
        
        browse_output_btn = ttk.Button(
            file_frame, 
            text="Browse...", 
            command=self.browse_output
        )
        browse_output_btn.grid(row=4, column=1, padx=5, pady=5)
        
        file_frame.columnconfigure(0, weight=1)
        
        # ========== CORRECTION PARAMETERS SECTION ==========
        params_frame = ttk.LabelFrame(
            main_frame, 
            text="Correction Parameters", 
            padding="10"
        )
        params_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        # Intensity slider
        ttk.Label(params_frame, text="Color Correction Intensity:").pack(
            anchor=tk.W, pady=(5, 0)
        )
        
        intensity_frame = ttk.Frame(params_frame)
        intensity_frame.pack(fill=tk.X, pady=5)
        
        self.intensity_slider = ttk.Scale(
            intensity_frame,
            from_=0.0,
            to=2.0,
            orient=tk.HORIZONTAL,
            variable=self.intensity_var,
            command=self.update_intensity_label
        )
        self.intensity_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        
        self.intensity_label = ttk.Label(intensity_frame, text="1.00", width=4)
        self.intensity_label.pack(side=tk.LEFT)
        
        ttk.Label(params_frame, text="(0.0 = No Correction, 2.0 = Aggressive)", 
                 font=('Helvetica', 8)).pack(anchor=tk.W)
        
        # Contrast slider
        ttk.Label(params_frame, text="Contrast Enhancement:").pack(
            anchor=tk.W, pady=(10, 0)
        )
        
        contrast_frame = ttk.Frame(params_frame)
        contrast_frame.pack(fill=tk.X, pady=5)
        
        self.contrast_slider = ttk.Scale(
            contrast_frame,
            from_=0.0,
            to=4.0,
            orient=tk.HORIZONTAL,
            variable=self.contrast_var,
            command=self.update_contrast_label
        )
        self.contrast_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        
        self.contrast_label = ttk.Label(contrast_frame, text="2.00", width=4)
        self.contrast_label.pack(side=tk.LEFT)
        
        ttk.Label(params_frame, text="(0.0 = No Enhancement, 4.0 = Maximum)", 
                 font=('Helvetica', 8)).pack(anchor=tk.W)
        
        # Denoise checkbox
        denoise_check = ttk.Checkbutton(
            params_frame,
            text="Enable Denoising (slower but cleaner)",
            variable=self.denoise_var
        )
        denoise_check.pack(anchor=tk.W, pady=10)
        
        # ========== PROGRESS SECTION ==========
        progress_frame = ttk.LabelFrame(
            main_frame, 
            text="Processing Status", 
            padding="10"
        )
        progress_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        self.progress_bar = ttk.Progressbar(
            progress_frame,
            mode='determinate',
            length=300
        )
        self.progress_bar.pack(fill=tk.X, pady=5)
        
        self.status_label = ttk.Label(
            progress_frame, 
            text="Ready to process video",
            font=('Helvetica', 9)
        )
        self.status_label.pack(pady=5)
        
        # ========== ACTION BUTTONS ==========
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        self.process_btn = ttk.Button(
            button_frame,
            text="🎬 Start Correction",
            command=self.start_processing
        )
        self.process_btn.pack(fill=tk.X, pady=2)
        
        reset_btn = ttk.Button(
            button_frame,
            text="Reset Parameters",
            command=self.reset_parameters
        )
        reset_btn.pack(fill=tk.X, pady=2)
        
        # ========== INFO SECTION ==========
        info_text = "⚠️ FFmpeg required for audio preservation"
        info_label = ttk.Label(
            main_frame,
            text=info_text,
            font=('Helvetica', 8),
            foreground='#666666'
        )
        info_label.pack(pady=10)
    
    def create_preview_panel(self, parent):
        """Create the preview panel with before/after comparison."""
        # Title
        preview_title = ttk.Label(
            parent,
            text="Live Preview",
            font=('Helvetica', 14, 'bold')
        )
        preview_title.pack(pady=10)
        
        # Container for both previews
        preview_container = ttk.Frame(parent)
        preview_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Original frame
        original_frame = ttk.LabelFrame(preview_container, text="Original", padding="5")
        original_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        self.original_canvas = tk.Canvas(original_frame, bg='#2b2b2b', width=400, height=300)
        self.original_canvas.pack(fill=tk.BOTH, expand=True)
        
        # Corrected frame
        corrected_frame = ttk.LabelFrame(preview_container, text="Corrected", padding="5")
        corrected_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        self.preview_canvas = tk.Canvas(corrected_frame, bg='#2b2b2b', width=400, height=300)
        self.preview_canvas.pack(fill=tk.BOTH, expand=True)
        
        # Instructions label
        instructions = ttk.Label(
            parent,
            text="Click 'Load Preview' to extract a frame from the video and see live corrections",
            font=('Helvetica', 9),
            foreground='#666666'
        )
        instructions.pack(pady=5)
    
    def update_intensity_label(self, value):
        """Update intensity label when slider moves."""
        self.intensity_label.config(text=f"{float(value):.2f}")
    
    def update_contrast_label(self, value):
        """Update contrast label when slider moves."""
        self.contrast_label.config(text=f"{float(value):.2f}")
    
    def on_parameter_change(self, *args):
        """Called when any parameter changes - updates preview."""
        if self.original_frame is not None:
            self.update_preview()
    
    def browse_input(self):
        """Open file dialog to select input video."""
        filename = filedialog.askopenfilename(
            title="Select Input Video",
            filetypes=[
                ("MP4 files", "*.mp4"),
                ("All files", "*.*")
            ]
        )
        
        if filename:
            self.input_file.set(filename)
            
            # Auto-suggest output filename
            if not self.output_file.get():
                input_path = Path(filename)
                output_path = input_path.parent / f"{input_path.stem}_corrected.mp4"
                self.output_file.set(str(output_path))
            
            # Clear current preview
            self.original_frame = None
            self.preview_frame = None
            self.clear_preview()
    
    def browse_output(self):
        """Open file dialog to select output location."""
        filename = filedialog.asksaveasfilename(
            title="Save Output Video As",
            defaultextension=".mp4",
            filetypes=[
                ("MP4 files", "*.mp4"),
                ("All files", "*.*")
            ]
        )
        
        if filename:
            self.output_file.set(filename)
    
    def load_preview_frame(self):
        """Load a frame from the video for preview."""
        if not self.input_file.get():
            messagebox.showerror("Error", "Please select an input video first")
            return
        
        input_path = Path(self.input_file.get())
        if not input_path.exists():
            messagebox.showerror("Error", "Input file does not exist")
            return
        
        try:
            # Open video
            cap = cv2.VideoCapture(str(input_path))
            
            if not cap.isOpened():
                messagebox.showerror("Error", "Cannot open video file")
                return
            
            # Get total frames and jump to middle
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            middle_frame = total_frames // 2
            cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame)
            
            # Read frame
            ret, frame = cap.read()
            cap.release()
            
            if not ret:
                messagebox.showerror("Error", "Could not read frame from video")
                return
            
            # Store original frame
            self.original_frame = frame.copy()
            
            # Display original and update preview
            self.display_original()
            self.update_preview()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load preview:\n{str(e)}")
    
    def display_original(self):
        """Display the original frame."""
        if self.original_frame is None:
            return
        
        # Convert to RGB for display
        frame_rgb = cv2.cvtColor(self.original_frame, cv2.COLOR_BGR2RGB)
        
        # Resize to fit canvas
        canvas_width = self.original_canvas.winfo_width()
        canvas_height = self.original_canvas.winfo_height()
        
        if canvas_width <= 1 or canvas_height <= 1:
            canvas_width = 400
            canvas_height = 300
        
        frame_resized = self.resize_frame(frame_rgb, canvas_width, canvas_height)
        
        # Convert to PhotoImage
        image = Image.fromarray(frame_resized)
        self.photo_original = ImageTk.PhotoImage(image)
        
        # Display on canvas
        self.original_canvas.delete("all")
        self.original_canvas.create_image(
            canvas_width // 2, 
            canvas_height // 2, 
            image=self.photo_original
        )
    
    def update_preview(self):
        """Update the preview with current parameters."""
        if self.original_frame is None:
            return
        
        try:
            # Create a temporary corrector with current parameters
            # We use a dummy path since we're only processing a frame
            temp_input = self.input_file.get() or "dummy.mp4"
            corrector = UnderwaterColorCorrector(
                temp_input,
                "dummy_output.mp4",
                self.intensity_var.get(),
                self.contrast_var.get()
            )
            
            # Process the frame
            corrected = corrector.process_frame(self.original_frame.copy(), False)
            
            # Convert to RGB for display
            frame_rgb = cv2.cvtColor(corrected, cv2.COLOR_BGR2RGB)
            
            # Resize to fit canvas
            canvas_width = self.preview_canvas.winfo_width()
            canvas_height = self.preview_canvas.winfo_height()
            
            if canvas_width <= 1 or canvas_height <= 1:
                canvas_width = 400
                canvas_height = 300
            
            frame_resized = self.resize_frame(frame_rgb, canvas_width, canvas_height)
            
            # Convert to PhotoImage
            image = Image.fromarray(frame_resized)
            self.photo_preview = ImageTk.PhotoImage(image)
            
            # Display on canvas
            self.preview_canvas.delete("all")
            self.preview_canvas.create_image(
                canvas_width // 2, 
                canvas_height // 2, 
                image=self.photo_preview
            )
            
        except Exception as e:
            print(f"Preview update error: {e}")
    
    def resize_frame(self, frame, max_width, max_height):
        """Resize frame to fit within max dimensions while maintaining aspect ratio."""
        height, width = frame.shape[:2]
        
        # Calculate scaling factor
        scale = min(max_width / width, max_height / height)
        
        if scale < 1:
            new_width = int(width * scale)
            new_height = int(height * scale)
            frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        return frame
    
    def clear_preview(self):
        """Clear both preview canvases."""
        self.original_canvas.delete("all")
        self.preview_canvas.delete("all")
        self.photo_original = None
        self.photo_preview = None
    
    def reset_parameters(self):
        """Reset all parameters to default values."""
        self.intensity_var.set(1.0)
        self.contrast_var.set(2.0)
        self.denoise_var.set(False)
        self.update_intensity_label(1.0)
        self.update_contrast_label(2.0)
    
    def validate_inputs(self):
        """
        Validate user inputs before processing.
        
        Returns:
            True if inputs are valid, False otherwise
        """
        # Check input file
        if not self.input_file.get():
            messagebox.showerror("Error", "Please select an input video file")
            return False
        
        input_path = Path(self.input_file.get())
        if not input_path.exists():
            messagebox.showerror("Error", "Input file does not exist")
            return False
        
        if input_path.suffix.lower() != '.mp4':
            messagebox.showerror("Error", "Input file must be MP4 format")
            return False
        
        # Check output file
        if not self.output_file.get():
            messagebox.showerror("Error", "Please specify an output file location")
            return False
        
        # Check if output file already exists
        output_path = Path(self.output_file.get())
        if output_path.exists():
            response = messagebox.askyesno(
                "Overwrite File?",
                f"The file {output_path.name} already exists.\nDo you want to overwrite it?"
            )
            if not response:
                return False
        
        return True
    
    def update_progress(self, progress, frame_count, total_frames):
        """
        Update progress bar and status label.
        
        Args:
            progress: Progress percentage (0-100)
            frame_count: Current frame number
            total_frames: Total number of frames
        """
        self.progress_bar['value'] = progress
        self.status_label.config(
            text=f"Processing: {frame_count}/{total_frames} frames ({progress:.1f}%)"
        )
        self.root.update_idletasks()
    
    def process_video_thread(self):
        """Worker thread for video processing."""
        try:
            # Create corrector instance
            corrector = UnderwaterColorCorrector(
                self.input_file.get(),
                self.output_file.get(),
                self.intensity_var.get(),
                self.contrast_var.get()
            )
            
            # Process video with progress callback
            corrector.process_video(
                apply_denoise=self.denoise_var.get(),
                show_progress=False,
                progress_callback=self.update_progress
            )
            
            # Processing complete
            self.root.after(0, self.processing_complete)
            
        except Exception as e:
            self.root.after(0, lambda: self.processing_error(str(e)))
    
    def start_processing(self):
        """Start video processing in a separate thread."""
        if not self.validate_inputs():
            return
        
        # Update UI state
        self.processing = True
        self.process_btn.config(state=tk.DISABLED)
        self.progress_bar['value'] = 0
        self.status_label.config(text="Initializing...")
        
        # Start processing thread
        thread = threading.Thread(target=self.process_video_thread, daemon=True)
        thread.start()
    
    def processing_complete(self):
        """Handle successful processing completion."""
        self.processing = False
        self.process_btn.config(state=tk.NORMAL)
        self.progress_bar['value'] = 100
        self.status_label.config(text="✅ Processing complete!")
        
        messagebox.showinfo(
            "Success",
            f"Video correction completed successfully!\n\nOutput saved to:\n{self.output_file.get()}"
        )
    
    def processing_error(self, error_message):
        """
        Handle processing error.
        
        Args:
            error_message: Error description
        """
        self.processing = False
        self.process_btn.config(state=tk.NORMAL)
        self.progress_bar['value'] = 0
        self.status_label.config(text="❌ Processing failed")
        
        messagebox.showerror(
            "Processing Error",
            f"An error occurred during processing:\n\n{error_message}"
        )


def main():
    """Main entry point for GUI application."""
    # Create root window
    root = tk.Tk()
    
    # Set theme (try to use modern theme if available)
    try:
        style = ttk.Style()
        style.theme_use('clam')  # Modern theme
    except:
        pass
    
    # Create application
    app = UnderwaterColorCorrectionGUI(root)
    
    # Start GUI event loop
    root.mainloop()


if __name__ == "__main__":
    main()
