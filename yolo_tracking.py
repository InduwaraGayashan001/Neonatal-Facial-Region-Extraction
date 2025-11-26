"""
Method 2: YOLO Built-in Tracking
Uses YOLO's native tracking capabilities for efficient processing.
"""

import os
import time
from ultralytics import YOLO

def process_video_with_yolo_tracking(video_path, model_path, output_dir="method2_output", 
                                    imgsz=320, conf=0.3, vid_stride=2):
    """Process video using YOLO's built-in tracking."""
    
    print("="*70)
    print("METHOD 2: YOLO BUILT-IN TRACKING")
    print("="*70)
    
    # Load model
    print(f"Loading model: {model_path}")
    model = YOLO(model_path)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nProcessing Configuration:")
    print(f"  Input: {video_path}")
    print(f"  Image Size: {imgsz}")
    print(f"  Confidence: {conf}")
    print(f"  Frame Stride: {vid_stride}")
    print(f"  Output: {output_dir}")
    
    # Start tracking
    start_time = time.time()
    
    results = model.track(
        source=video_path,
        save=True,
        project=output_dir,
        name="yolo_tracked",
        persist=True,
        show_labels=True,
        show_conf=True,
        line_width=2,
        # Optimization parameters
        imgsz=imgsz,
        conf=conf,
        iou=0.7,
        half=True,
        device=0,
        verbose=False,
        stream_buffer=True,
        vid_stride=vid_stride
    )
    
    total_time = time.time() - start_time
    
    # Print results
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"Processing Time: {total_time:.2f} seconds")
    print(f"Output saved to: {output_dir}/yolo_tracked")
    
    return {
        'total_time': total_time,
        'output_dir': output_dir
    }

if __name__ == "__main__":
    # Default configuration
    process_video_with_yolo_tracking(
        video_path="vid2.avi",
        model_path="runs/segment/train/weights/best.pt",
        output_dir="method2_output",
        imgsz=320,
        conf=0.3,
        vid_stride=2
    )
