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
        self.color_mask_var = tk.StringVar(value='all')
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
        self.color_mask_var.trace_add('write', self.on_parameter_change)
        
        # Bind window resize event
        self.root.bind('<Configure>', self.on_window_resize)
        self.resize_timer = None
    
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
            from_=-2.0,
            to=2.0,
            orient=tk.HORIZONTAL,
            variable=self.intensity_var,
            command=self.update_intensity_label
        )
        self.intensity_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        
        self.intensity_label = ttk.Label(intensity_frame, text="1.00", width=5)
        self.intensity_label.pack(side=tk.LEFT)
        
        ttk.Label(params_frame, text="(Negative: Remove Red, Positive: Add Red)", 
                 font=('Helvetica', 8)).pack(anchor=tk.W)
        
        # Contrast slider
        ttk.Label(params_frame, text="Contrast Enhancement:").pack(
            anchor=tk.W, pady=(10, 0)
        )
        
        contrast_frame = ttk.Frame(params_frame)
        contrast_frame.pack(fill=tk.X, pady=5)
        
        self.contrast_slider = ttk.Scale(
            contrast_frame,
            from_=-2.0,
            to=4.0,
            orient=tk.HORIZONTAL,
            variable=self.contrast_var,
            command=self.update_contrast_label
        )
        self.contrast_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        
        self.contrast_label = ttk.Label(contrast_frame, text="2.00", width=5)
        self.contrast_label.pack(side=tk.LEFT)
        
        ttk.Label(params_frame, text="(Negative: Reduce Contrast, Positive: Enhance Contrast)", 
                 font=('Helvetica', 8)).pack(anchor=tk.W)
        
        # Color mask selector
        ttk.Label(params_frame, text="Target Colors for Red Enhancement:").pack(
            anchor=tk.W, pady=(10, 0)
        )
        
        color_mask_frame = ttk.Frame(params_frame)
        color_mask_frame.pack(fill=tk.X, pady=5)
        
        color_options = [
            ('All Colors', 'all'),
            ('Blue Only', 'blue'),
            ('Green Only', 'green'),
            ('Cyan/Turquoise', 'cyan')
        ]
        
        for i, (label, value) in enumerate(color_options):
            rb = ttk.Radiobutton(
                color_mask_frame,
                text=label,
                variable=self.color_mask_var,
                value=value
            )
            rb.pack(anchor=tk.W, padx=20)
        
        ttk.Label(params_frame, text="(Selective correction on specific underwater color casts)", 
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
    
    def on_window_resize(self, event):
        """Called when window is resized - debounced refresh."""
        # Only handle root window resize events
        if event.widget != self.root:
            return
        
        # Cancel previous timer if exists
        if self.resize_timer is not None:
            self.root.after_cancel(self.resize_timer)
        
        # Set new timer to refresh after 200ms of no resize events
        self.resize_timer = self.root.after(200, self.refresh_preview_display)
    
    def browse_input(self):
        """Open file dialog to select input video or image."""
        filename = filedialog.askopenfilename(
            title="Select Input Video or Image",
            filetypes=[
                ("Supported files", "*.mp4 *.jpg *.jpeg *.JPG *.JPEG *.png *.PNG"),
                ("MP4 files", "*.mp4"),
                ("JPEG files", "*.jpg *.jpeg *.JPG *.JPEG"),
                ("PNG files", "*.png *.PNG"),
                ("All files", "*.*")
            ]
        )
        
        if filename:
            self.input_file.set(filename)
            
            # Auto-suggest output filename
            if not self.output_file.get():
                input_path = Path(filename)
                if input_path.suffix.lower() == '.mp4':
                    output_path = input_path.parent / f"{input_path.stem}_corrected.mp4"
                else:
                    # For images, keep the same extension
                    output_path = input_path.parent / f"{input_path.stem}_corrected{input_path.suffix}"
                self.output_file.set(str(output_path))
            
            # Clear current preview
            self.original_frame = None
            self.preview_frame = None
            self.clear_preview()
    
    def browse_output(self):
        """Open file dialog to select output location."""
        # Determine default extension based on input
        default_ext = ".mp4"
        filetypes = [
            ("Supported files", "*.mp4 *.jpg *.jpeg *.png"),
            ("MP4 files", "*.mp4"),
            ("JPEG files", "*.jpg *.jpeg"),
            ("PNG files", "*.png"),
            ("All files", "*.*")
        ]
        
        if self.input_file.get():
            input_ext = Path(self.input_file.get()).suffix.lower()
            if input_ext in ['.jpg', '.jpeg', '.png']:
                default_ext = input_ext
        
        filename = filedialog.asksaveasfilename(
            title="Save Output As",
            defaultextension=default_ext,
            filetypes=filetypes
        )
        
        if filename:
            self.output_file.set(filename)
    
    def load_preview_frame(self):
        """Load a frame from the video or load the image for preview."""
        if not self.input_file.get():
            messagebox.showerror("Error", "Please select an input file first")
            return
        
        input_path = Path(self.input_file.get())
        if not input_path.exists():
            messagebox.showerror("Error", "Input file does not exist")
            return
        
        try:
            # Check if it's an image or video
            if input_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                # Load image directly
                frame = cv2.imread(str(input_path))
                
                if frame is None:
                    messagebox.showerror("Error", "Could not read image file")
                    return
                
                # Store original frame
                self.original_frame = frame.copy()
            else:
                # Load from video
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
            self.refresh_preview_display()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load preview:\n{str(e)}")
    
    def refresh_preview_display(self):
        """Refresh both original and corrected preview displays."""
        if self.original_frame is None:
            return
        
        self.display_original()
        self.update_preview()
    
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
            # Use dummy paths since we're only processing a frame
            temp_input = self.input_file.get() if self.input_file.get() else "dummy.mp4"
            
            # Check if file exists before creating corrector
            if not Path(temp_input).exists():
                # For preview only, create a minimal corrector-like object
                # We'll manually apply the corrections
                self.update_preview_manual()
                return
            
            corrector = UnderwaterColorCorrector(
                temp_input,
                "dummy_output.mp4",
                self.intensity_var.get(),
                self.contrast_var.get(),
                self.color_mask_var.get()
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
            import traceback
            traceback.print_exc()
    
    def update_preview_manual(self):
        """Fallback method to update preview without file access."""
        if self.original_frame is None:
            return
        
        try:
            # Manually create a simple preview without full corrector
            frame = self.original_frame.copy()
            intensity = self.intensity_var.get()
            contrast = self.contrast_var.get()
            
            # Simple color correction without file dependency
            if intensity > 0.0:
                # Basic white balance in LAB
                lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
                avg_a = np.average(lab[:, :, 1])
                avg_b = np.average(lab[:, :, 2])
                lab[:, :, 1] = lab[:, :, 1] - ((avg_a - 128) * (1.1 * intensity))
                lab[:, :, 2] = lab[:, :, 2] - ((avg_b - 128) * (1.1 * intensity))
                frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
                
                # Basic red enhancement
                b, g, r = cv2.split(frame)
                r = cv2.convertScaleAbs(r, alpha=1.0 + (0.3 * intensity), beta=10 * intensity)
                b = cv2.convertScaleAbs(b, alpha=1.0 - (0.1 * intensity), beta=-5 * intensity)
                g = cv2.convertScaleAbs(g, alpha=1.0 - (0.05 * intensity), beta=-3 * intensity)
                frame = cv2.merge([b, g, r])
            
            if contrast > 0.0:
                # Basic contrast enhancement
                lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=contrast, tileGridSize=(8, 8))
                l = clahe.apply(l)
                frame = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
            
            # Convert to RGB for display
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
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
            print(f"Manual preview update error: {e}")
    
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
        self.color_mask_var.set('all')
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
        
        if input_path.suffix.lower() not in ['.mp4', '.jpg', '.jpeg', '.png']:
            messagebox.showerror("Error", "Input file must be MP4, JPG, JPEG, or PNG format")
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
        """Worker thread for video/image processing."""
        try:
            # Create corrector instance
            corrector = UnderwaterColorCorrector(
                self.input_file.get(),
                self.output_file.get(),
                self.intensity_var.get(),
                self.contrast_var.get(),
                self.color_mask_var.get()
            )
            
            # Process based on file type
            if corrector.is_image:
                corrector.process_image(
                    show_progress=False,
                    apply_denoise=self.denoise_var.get()
                )
            else:
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
            f"Processing completed successfully!\n\nOutput saved to:\n{self.output_file.get()}"
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
