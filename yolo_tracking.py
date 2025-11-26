"""
Method 2: YOLO Built-in Tracking
Uses YOLO's native tracking capabilities for efficient processing.
"""

import os
import time
import cv2
import numpy as np
from ultralytics import YOLO

def process_video_with_yolo_tracking(video_path, model_path, output_dir="method2_output", 
                                    imgsz=320, conf=0.3, vid_stride=2, binary_mask_only=True):
    """Process video using YOLO's built-in tracking.
    
    Args:
        binary_mask_only: If True, outputs only white mask on black background (no colors, no text)
    """
    
    print("="*70)
    print("METHOD 2: YOLO BUILT-IN TRACKING")
    if binary_mask_only:
        print("Output Mode: Binary Mask Only (No Text/Colors)")
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
    
    if binary_mask_only:
        # Custom processing for binary mask output
        import cv2
        import numpy as np
        
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "yolo_tracked_binary.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        results_gen = model.track(
            source=video_path,
            stream=True,
            persist=True,
            imgsz=imgsz,
            conf=conf,
            iou=0.7,
            half=True,
            device=0,
            verbose=False,
            vid_stride=vid_stride
        )
        
        frame_count = 0
        for results in results_gen:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            
            mask = np.zeros((height, width), dtype=np.uint8)
            if results.masks is not None and len(results.masks) > 0:
                mask_xy = results.masks.xy[0]
                pts = np.array(mask_xy, dtype=np.int32).reshape((-1, 1, 2))
                cv2.fillPoly(mask, [pts], 255)
            # Apply mask to original frame to show only face region
            masked_frame = cv2.bitwise_and(frame, frame, mask=mask)
            out.write(masked_frame)
        
        cap.release()
        out.release()
    else:
        # Standard YOLO tracking with visualization
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
        video_path="video/vid2.avi",
        model_path="runs/segment/train/weights/best.pt",
        output_dir="method2_output",
        imgsz=320,
        conf=0.3,
        vid_stride=1
    )
