"""
Method 2: YOLO Built-in Tracking
Uses YOLO's native tracking capabilities for efficient processing.
"""

import os
import time
import subprocess
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
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "yolo_tracked_binary.mp4")
        
        # Detect input codec and choose appropriate output
        input_size_mb = os.path.getsize(video_path) / (1024 * 1024)
        uncompressed_size_mb = (width * height * 3 * total_frames) / (1024 * 1024)
        is_compressed = input_size_mb < (uncompressed_size_mb * 0.5)
        
        if is_compressed:
            print(f"Input is compressed ({input_size_mb:.1f} MB), using HuffYUV lossless codec...")
            output_path_avi = output_path.rsplit('.', 1)[0] + '.avi'
            ffmpeg_cmd = [
                'ffmpeg', '-y',
                '-f', 'rawvideo', '-vcodec', 'rawvideo',
                '-s', f'{width}x{height}', '-pix_fmt', 'bgr24', '-r', str(fps),
                '-i', '-', '-an',
                '-vcodec', 'huffyuv', '-pix_fmt', 'rgb24',
                output_path_avi
            ]
            output_path = output_path_avi
        else:
            print(f"Input is uncompressed ({input_size_mb:.1f} MB), using rawvideo output...")
            output_path_avi = output_path.rsplit('.', 1)[0] + '.avi'
            ffmpeg_cmd = [
                'ffmpeg', '-y',
                '-f', 'rawvideo', '-vcodec', 'rawvideo',
                '-s', f'{width}x{height}', '-pix_fmt', 'bgr24', '-r', str(fps),
                '-i', '-', '-an',
                '-vcodec', 'rawvideo', '-pix_fmt', 'bgr24',
                output_path_avi
            ]
            output_path = output_path_avi
        
        ffmpeg_process = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        
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
            ffmpeg_process.stdin.write(masked_frame.tobytes())
        
        cap.release()
        ffmpeg_process.stdin.close()
        ffmpeg_process.wait()
        
        # Print file size comparison
        input_size = os.path.getsize(video_path)
        output_size = os.path.getsize(output_path) if os.path.exists(output_path) else 0
        input_size_mb = input_size / (1024 * 1024)
        output_size_mb = output_size / (1024 * 1024)
        size_ratio = (output_size / input_size * 100) if input_size > 0 else 0
        print(f"\nFile Sizes:")
        print(f"  Input:  {input_size_mb:.2f} MB")
        print(f"  Output: {output_size_mb:.2f} MB ({size_ratio:.1f}% of input)")
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
