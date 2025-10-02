#!/usr/bin/env python3
"""
Video processing script - Extract last N frames from video
Read video and extract the last specified number of frames, save as filename_last_n_frames
"""

import os
import sys
import argparse
import cv2
from pathlib import Path


def extract_last_n_frames(input_video_path, n_frames, output_path=None):
    """
    Extract the last n frames from a video and save as a new video file
    
    Args:
        input_video_path (str): Input video file path
        n_frames (int): Number of frames to extract
        output_path (str, optional): Output path, auto-generated if None
    
    Returns:
        str: Output file path
    """
    # Check if input file exists
    if not os.path.exists(input_video_path):
        raise FileNotFoundError(f"Input video file not found: {input_video_path}")
    
    # Open video file
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {input_video_path}")
    
    try:
        # Get video information
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Video information:")
        print(f"  Total frames: {total_frames}")
        print(f"  FPS: {fps}")
        print(f"  Resolution: {width}x{height}")
        
        # Check if requested frames exceed total frames
        if n_frames > total_frames:
            print(f"Warning: Requested frames ({n_frames}) exceed total frames ({total_frames})")
            n_frames = total_frames
        
        # Calculate starting frame position
        start_frame = max(0, total_frames - n_frames)
        
        # Set output file path
        if output_path is None:
            input_path = Path(input_video_path)
            output_filename = f"{input_path.stem}_last_{n_frames}_frames{input_path.suffix}"
            output_path = input_path.parent / output_filename
        
        print(f"Starting extraction from frame {start_frame}, extracting {n_frames} frames")
        print(f"Output file: {output_path}")
        
        # Set video encoder
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        # Move to start position
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        # Read and write frames
        frames_written = 0
        while frames_written < n_frames:
            ret, frame = cap.read()
            if not ret:
                print(f"Warning: Cannot read more frames at frame {start_frame + frames_written}")
                break
            
            out.write(frame)
            frames_written += 1
            
            # Show progress
            if frames_written % 10 == 0 or frames_written == n_frames:
                progress = (frames_written / n_frames) * 100
                print(f"Progress: {frames_written}/{n_frames} ({progress:.1f}%)")
        
        print(f"Successfully extracted {frames_written} frames to {output_path}")
        return str(output_path)
        
    finally:
        # Release resources
        cap.release()
        if 'out' in locals():
            out.release()
        cv2.destroyAllWindows()


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Extract the last specified number of frames from video",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            Usage examples:
            python crop_video.py input.mp4 30                    # Extract last 30 frames
            python crop_video.py input.mp4 50 -o output.mp4      # Extract last 50 frames to specified file
            python crop_video.py *.mp4 25                        # Batch process multiple files
            """
    )
    
    parser.add_argument(
        "input_video", 
        nargs='+',
        help="Input video file path(s) (supports wildcards)"
    )
    
    parser.add_argument(
        "n_frames", 
        type=int,
        help="Number of frames to extract"
    )
    
    parser.add_argument(
        "-o", "--output", 
        help="Output file path (only effective when processing single file)"
    )
    
    args = parser.parse_args()
    
    # Validate frame number parameter
    if args.n_frames <= 0:
        print("Error: Number of frames must be greater than 0")
        sys.exit(1)
    
    # Process input files
    input_files = []
    for pattern in args.input_video:
        if os.path.exists(pattern):
            input_files.append(pattern)
        else:
            # If not a direct file path, might be a wildcard pattern
            from glob import glob
            matched_files = glob(pattern)
            if matched_files:
                input_files.extend(matched_files)
            else:
                print(f"Warning: No files found matching pattern: {pattern}")
    
    if not input_files:
        print("Error: No input files found")
        sys.exit(1)
    
    # Check output parameter
    if len(input_files) > 1 and args.output:
        print("Warning: Ignoring --output parameter when processing multiple files")
        args.output = None
    
    # Process each file
    success_count = 0
    for input_file in input_files:
        try:
            print(f"\nProcessing file: {input_file}")
            output_file = extract_last_n_frames(
                input_file, 
                args.n_frames, 
                args.output
            )
            success_count += 1
        except Exception as e:
            print(f"Error: Failed to process file {input_file}: {e}")
            continue
    
    print(f"\nCompleted! Successfully processed {success_count}/{len(input_files)} files")


if __name__ == "__main__":
    main()