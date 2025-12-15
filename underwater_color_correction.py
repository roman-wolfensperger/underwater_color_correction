#!/usr/bin/env python3
"""
Underwater Video Color Correction Script
Optimized for GoPro Hero 3 footage with red physical filter
Python 3.12 compatible - WITH AUDIO PRESERVATION
"""

import cv2
import numpy as np
from pathlib import Path
import argparse
from typing import Tuple, Optional
import sys
import subprocess
import tempfile
import shutil


class UnderwaterColorCorrector:
    """
    Handles color correction for underwater footage shot with red filter.
    Compensates for blue-green dominance and enhances natural colors.
    """
    
    def __init__(self, input_path: str, output_path: str, 
                 intensity: float = 1.0, contrast: float = 2.0):
        """
        Initialize the color corrector.
        
        Args:
            input_path: Path to input MP4 video
            output_path: Path for output corrected video
            intensity: Correction intensity (0.0-2.0), default 1.0, 0.0 = no correction
            contrast: Contrast enhancement level (0.0-4.0), default 2.0
        """
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.intensity = np.clip(intensity, 0.0, 2.0)
        self.contrast = np.clip(contrast, 0.0, 4.0)
        
        # Validate input file
        if not self.input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        if self.input_path.suffix.lower() != '.mp4':
            raise ValueError("Input file must be MP4 format")
    
    def apply_white_balance(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply automatic white balance correction.
        Compensates for blue-green cast from water absorption.
        
        Args:
            frame: Input BGR frame
            
        Returns:
            White-balanced frame
        """
        # If intensity is 0, return original frame
        if self.intensity == 0.0:
            return frame
        
        # Convert to LAB color space for better color manipulation
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        
        # Calculate average values for each channel
        avg_a = np.average(lab[:, :, 1])
        avg_b = np.average(lab[:, :, 2])
        
        # Adjust A and B channels to neutralize color cast
        lab[:, :, 1] = lab[:, :, 1] - ((avg_a - 128) * (1.1 * self.intensity))
        lab[:, :, 2] = lab[:, :, 2] - ((avg_b - 128) * (1.1 * self.intensity))
        
        # Convert back to BGR
        balanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        return balanced
    
    def enhance_red_channel(self, frame: np.ndarray) -> np.ndarray:
        """
        Enhance red channel to compensate for water absorption.
        Red light is absorbed most in water, even with physical filter.
        
        Args:
            frame: Input BGR frame
            
        Returns:
            Red-enhanced frame
        """
        # If intensity is 0, return original frame
        if self.intensity == 0.0:
            return frame
        
        # Split channels
        b, g, r = cv2.split(frame)
        
        # Enhance red channel (water absorbs red first)
        r_enhanced = cv2.convertScaleAbs(r, alpha=1.0 + (0.3 * self.intensity), beta=10 * self.intensity)
        
        # Slightly reduce blue/green dominance
        b_reduced = cv2.convertScaleAbs(b, alpha=1.0 - (0.1 * self.intensity), beta=-5 * self.intensity)
        g_reduced = cv2.convertScaleAbs(g, alpha=1.0 - (0.05 * self.intensity), beta=-3 * self.intensity)
        
        # Merge channels back
        enhanced = cv2.merge([b_reduced, g_reduced, r_enhanced])
        return enhanced
    
    def apply_contrast_enhancement(self, frame: np.ndarray) -> np.ndarray:
        """
        Enhance contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization).
        Improves visibility in low-contrast underwater scenes.
        
        Args:
            frame: Input BGR frame
            
        Returns:
            Contrast-enhanced frame
        """
        # Convert to LAB for luminance-only processing
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel with adjustable contrast parameter
        clahe = cv2.createCLAHE(clipLimit=self.contrast, tileGridSize=(8, 8))
        l_clahe = clahe.apply(l)
        
        # Merge and convert back
        lab_clahe = cv2.merge([l_clahe, a, b])
        enhanced = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
        return enhanced
    
    def apply_gamma_correction(self, frame: np.ndarray, gamma: float = 1.2) -> np.ndarray:
        """
        Apply gamma correction to brighten mid-tones.
        Compensates for light loss at depth.
        
        Args:
            frame: Input BGR frame
            gamma: Gamma value (>1 brightens, <1 darkens)
            
        Returns:
            Gamma-corrected frame
        """
        # If intensity is 0, return original frame
        if self.intensity == 0.0:
            return frame
        
        # Adjust gamma based on intensity (linear interpolation)
        # intensity=0 -> gamma=1.0 (no change)
        # intensity=1 -> gamma=1.2 (standard)
        # intensity=2 -> gamma=1.4 (strong)
        adjusted_gamma = 1.0 + ((gamma - 1.0) * self.intensity)
        
        # Build lookup table
        inv_gamma = 1.0 / adjusted_gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 
                         for i in range(256)]).astype("uint8")
        
        # Apply gamma correction
        return cv2.LUT(frame, table)
    
    def denoise_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply gentle denoising to reduce sensor noise.
        Useful for low-light underwater conditions.
        
        Args:
            frame: Input BGR frame
            
        Returns:
            Denoised frame
        """
        # Use fastNlMeansDenoisingColored for color images
        denoised = cv2.fastNlMeansDenoisingColored(frame, None, 6, 6, 7, 21)
        return denoised
    
    def process_frame(self, frame: np.ndarray, apply_denoise: bool = False) -> np.ndarray:
        """
        Apply full color correction pipeline to a single frame.
        
        Args:
            frame: Input BGR frame
            apply_denoise: Whether to apply denoising (slower)
            
        Returns:
            Fully corrected frame
        """
        # If both intensity and contrast are 0, return original frame
        # This avoids unnecessary color space conversions and rounding errors
        if self.intensity == 0.0 and self.contrast == 0.0 and not apply_denoise:
            return frame.copy()
        
        # Step 1: White balance correction (skipped if intensity=0)
        if self.intensity > 0.0:
            frame = self.apply_white_balance(frame)
        
        # Step 2: Enhance red channel (skipped if intensity=0)
        if self.intensity > 0.0:
            frame = self.enhance_red_channel(frame)
        
        # Step 3: Contrast enhancement (skipped if contrast=0)
        if self.contrast > 0.0:
            frame = self.apply_contrast_enhancement(frame)
        
        # Step 4: Gamma correction (skipped if intensity=0)
        if self.intensity > 0.0:
            frame = self.apply_gamma_correction(frame)
        
        # Step 5: Optional denoising (can be slow)
        if apply_denoise:
            frame = self.denoise_frame(frame)
        
        return frame
    
    def check_ffmpeg(self) -> bool:
        """
        Check if FFmpeg is available on the system.
        
        Returns:
            True if FFmpeg is available, False otherwise
        """
        try:
            subprocess.run(['ffmpeg', '-version'], 
                         stdout=subprocess.PIPE, 
                         stderr=subprocess.PIPE, 
                         check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False
    
    def extract_audio(self, temp_audio_path: str) -> bool:
        """
        Extract audio from input video using FFmpeg.
        
        Args:
            temp_audio_path: Path to save extracted audio
            
        Returns:
            True if audio was extracted successfully, False otherwise
        """
        try:
            cmd = [
                'ffmpeg',
                '-i', str(self.input_path),
                '-vn',  # No video
                '-acodec', 'copy',  # Copy audio codec
                '-y',  # Overwrite output file
                temp_audio_path
            ]
            
            subprocess.run(cmd, stdout=subprocess.PIPE, 
                         stderr=subprocess.PIPE, check=True)
            return True
        except subprocess.CalledProcessError:
            return False
    
    def merge_audio_video(self, temp_video_path: str) -> bool:
        """
        Merge processed video with original audio using FFmpeg.
        
        Args:
            temp_video_path: Path to temporary video without audio
            
        Returns:
            True if merge was successful, False otherwise
        """
        try:
            cmd = [
                'ffmpeg',
                '-i', temp_video_path,
                '-i', str(self.input_path),
                '-c:v', 'copy',  # Copy video codec
                '-c:a', 'aac',   # Encode audio to AAC
                '-map', '0:v:0',  # Video from first input
                '-map', '1:a:0?',  # Audio from second input (optional)
                '-shortest',  # Finish when shortest stream ends
                '-y',  # Overwrite output file
                str(self.output_path)
            ]
            
            subprocess.run(cmd, stdout=subprocess.PIPE, 
                         stderr=subprocess.PIPE, check=True)
            return True
        except subprocess.CalledProcessError:
            return False
    
    def process_video(self, apply_denoise: bool = False, 
                     show_progress: bool = True,
                     progress_callback: Optional[callable] = None):
        """
        Process entire video file with color correction and preserve audio.
        
        Args:
            apply_denoise: Whether to apply denoising (significantly slower)
            show_progress: Display progress information
            progress_callback: Optional callback function for progress updates
        """
        # Special case: if no correction needed, just copy the file directly
        if self.intensity == 0.0 and self.contrast == 0.0 and not apply_denoise:
            if show_progress:
                print("No corrections needed (intensity=0, contrast=0)")
                print("Copying input file to output location...")
            
            # Check if FFmpeg is available for lossless copy
            has_ffmpeg = self.check_ffmpeg()
            
            if has_ffmpeg:
                # Use FFmpeg to copy streams without re-encoding
                try:
                    cmd = [
                        'ffmpeg',
                        '-i', str(self.input_path),
                        '-c', 'copy',  # Copy all streams without re-encoding
                        '-y',  # Overwrite output file
                        str(self.output_path)
                    ]
                    subprocess.run(cmd, stdout=subprocess.PIPE, 
                                 stderr=subprocess.PIPE, check=True)
                    
                    if show_progress:
                        print(f"File copied successfully to: {self.output_path}")
                    return
                except subprocess.CalledProcessError:
                    if show_progress:
                        print("FFmpeg copy failed, falling back to manual copy...")
            
            # Fallback: simple file copy
            import shutil
            shutil.copy2(str(self.input_path), str(self.output_path))
            
            if show_progress:
                print(f"File copied successfully to: {self.output_path}")
            return
        
        # Check if FFmpeg is available for audio preservation
        has_ffmpeg = self.check_ffmpeg()
        
        if not has_ffmpeg:
            if show_progress:
                print("WARNING: FFmpeg not found. Audio will not be preserved.")
                print("Install FFmpeg to preserve audio track.")
                print("-" * 50)
        
        # Open input video
        cap = cv2.VideoCapture(str(self.input_path))
        
        if not cap.isOpened():
            raise IOError(f"Cannot open video file: {self.input_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if show_progress:
            print(f"Input video: {self.input_path.name}")
            print(f"Resolution: {width}x{height}")
            print(f"FPS: {fps}")
            print(f"Total frames: {total_frames}")
            print(f"Correction intensity: {self.intensity}")
            print(f"Contrast enhancement: {self.contrast}")
            print(f"Denoising: {'Enabled' if apply_denoise else 'Disabled'}")
            print(f"Audio preservation: {'Enabled' if has_ffmpeg else 'Disabled'}")
            print("-" * 50)
        
        # Determine output path (temporary if using FFmpeg)
        if has_ffmpeg:
            temp_video = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
            video_output_path = temp_video.name
            temp_video.close()
        else:
            video_output_path = str(self.output_path)
        
        # Create output video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                # Process frame
                corrected_frame = self.process_frame(frame, apply_denoise)
                
                # Write to output
                out.write(corrected_frame)
                
                frame_count += 1
                
                # Calculate progress
                progress = (frame_count / total_frames) * 100
                
                # Call progress callback if provided
                if progress_callback:
                    progress_callback(progress, frame_count, total_frames)
                
                # Show progress
                if show_progress and frame_count % 30 == 0:
                    print(f"Processing: {frame_count}/{total_frames} frames ({progress:.1f}%)", 
                          end='\r')
        
        finally:
            # Release resources
            cap.release()
            out.release()
            
            if show_progress:
                print(f"\nVideo processing complete!")
            
            # Merge audio if FFmpeg is available
            if has_ffmpeg:
                if show_progress:
                    print("Merging audio track...")
                
                success = self.merge_audio_video(video_output_path)
                
                # Clean up temporary file
                try:
                    Path(video_output_path).unlink()
                except:
                    pass
                
                if success:
                    if show_progress:
                        print("Audio merged successfully!")
                else:
                    if show_progress:
                        print("Warning: Could not merge audio. Output has no audio.")
            
            if show_progress:
                print(f"Output saved to: {self.output_path}")


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Color correction for underwater GoPro Hero 3 footage with red filter",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python underwater_color_correction.py input.mp4 output.mp4
  
  # With custom intensity and contrast
  python underwater_color_correction.py input.mp4 output.mp4 --intensity 1.5 --contrast 3.0
  
  # Enable denoising (slower but cleaner)
  python underwater_color_correction.py input.mp4 output.mp4 --denoise
  
Note: FFmpeg must be installed to preserve audio track.
        """
    )
    
    parser.add_argument('input', help='Input MP4 video file')
    parser.add_argument('output', help='Output MP4 video file')
    parser.add_argument(
        '--intensity', 
        type=float, 
        default=1.0,
        help='Correction intensity (0.0-2.0, default: 1.0, 0.0 = no correction)'
    )
    parser.add_argument(
        '--contrast',
        type=float,
        default=2.0,
        help='Contrast enhancement level (0.0-4.0, default: 2.0)'
    )
    parser.add_argument(
        '--denoise',
        action='store_true',
        help='Enable denoising (slower processing)'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress progress output'
    )
    
    args = parser.parse_args()
    
    try:
        # Create corrector instance
        corrector = UnderwaterColorCorrector(
            args.input, 
            args.output, 
            args.intensity,
            args.contrast
        )
        
        # Process video
        corrector.process_video(
            apply_denoise=args.denoise,
            show_progress=not args.quiet
        )
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
