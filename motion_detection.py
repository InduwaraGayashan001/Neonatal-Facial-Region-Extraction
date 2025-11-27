"""
Method 3: Motion-Triggered Detection (with fallbacks)
Detects faces only when motion is detected, with safety fallbacks.
"""

import cv2
import numpy as np
import time
import os
import subprocess
from ultralytics import YOLO

def detect_face(model, frame, conf_threshold=0.25):
    """Detect all faces in a single frame."""
    results = model.predict(frame, conf=conf_threshold, verbose=False)[0]
    
    if results.masks is None or len(results.masks) == 0:
        return [], []
    
    # Get all face masks and confidences
    mask_polygons = [mask_xy for mask_xy in results.masks.xy]
    confidences = [conf.item() for conf in results.boxes.conf]
    
    return mask_polygons, confidences

def draw_mask(frame, mask_polygon, confidence, color=(0, 255, 0), status_text=""):
    """Draw polygon mask on frame."""
    if mask_polygon is None or len(mask_polygon) == 0:
        return frame
    
    result = frame.copy()
    pts = np.array(mask_polygon, dtype=np.int32).reshape((-1, 1, 2))
    
    # Draw filled polygon
    overlay = result.copy()
    cv2.fillPoly(overlay, [pts], color)
    cv2.addWeighted(overlay, 0.3, result, 0.7, 0, result)
    
    # Add text
    if len(mask_polygon) > 0:
        x_min = int(np.min(mask_polygon[:, 0]))
        y_min = int(np.min(mask_polygon[:, 1]))
        cv2.putText(result, f"{status_text} {confidence:.2f}", (x_min, y_min - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    return result

def process_motion_detection(video_path, model_path, output_path, motion_threshold=5.0, 
                 fallback_interval=60, conf_threshold=0.25, binary_mask_only=False):
    """Process video with motion-triggered detection and fallbacks.
    
    Args:
        binary_mask_only: If True, outputs only white mask on black background (no colors, no text)
    """
    
    print("="*70)
    print("METHOD 3: MOTION-TRIGGERED DETECTION (WITH FALLBACKS)")
    if binary_mask_only:
        print("Output Mode: Binary Mask Only (No Text/Colors)")
    print("="*70)
    
    # Load model
    print(f"Loading model: {model_path}")
    model = YOLO(model_path)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video: {width}x{height}, {fps} FPS, {total_frames} frames")
    print(f"Motion Threshold: {motion_threshold}%")
    print(f"Fallback Interval: {fallback_interval} frames")
    
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
    
    # Motion detection setup
    prev_gray_roi = None
    
    # Tracking variables
    last_mask = None
    last_confidence = 0.0
    last_detection_frame = 0
    last_bbox = None  # Store bounding box of last detected face region
    
    # Stats
    start_time = time.time()
    frame_count = 0
    detection_count = 0
    motion_count = 0
    fallback_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Detect motion only in face region if available
        motion_detected = False
        motion_percentage = 0.0
        
        if last_bbox is not None:
            # Extract ROI (Region of Interest) with padding
            x1, y1, x2, y2 = last_bbox
            
            # Add padding (10% on each side)
            pad_x = int((x2 - x1) * 0.10)
            pad_y = int((y2 - y1) * 0.10)
            
            # Apply padding with boundary checks
            roi_x1 = max(0, x1 - pad_x)
            roi_y1 = max(0, y1 - pad_y)
            roi_x2 = min(width, x2 + pad_x)
            roi_y2 = min(height, y2 + pad_y)
            
            # Extract ROI from current frame
            roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]
            
            if roi.size > 0:
                gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                gray_roi_blurred = cv2.GaussianBlur(gray_roi, (17, 17), 0)
                
                # Compare with previous frame ROI
                if prev_gray_roi is not None and prev_gray_roi.shape == gray_roi_blurred.shape:
                    # Calculate frame difference in ROI
                    frame_diff = cv2.absdiff(prev_gray_roi, gray_roi_blurred)
                    _, thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)
                    
                    # Count motion pixels in ROI
                    motion_pixels = cv2.countNonZero(thresh)
                    roi_area = (roi_x2 - roi_x1) * (roi_y2 - roi_y1)
                    motion_percentage = (motion_pixels / roi_area) * 100
                    
                    motion_detected = motion_percentage > motion_threshold
                    if motion_detected:
                        motion_count += 1
                
                prev_gray_roi = gray_roi_blurred.copy()
        else:
            # No previous face region, detect on first frame
            motion_detected = True
        
        # Fallback check
        frames_since_detection = frame_count - last_detection_frame
        fallback_triggered = frames_since_detection >= fallback_interval
        
        # Decide whether to detect
        should_detect = (last_mask is None) or motion_detected or fallback_triggered
        
        if should_detect:
            # Run detection
            mask_polygons, confidences = detect_face(model, frame, conf_threshold)
            detection_count += len(mask_polygons) if mask_polygons else 0
            
            if len(mask_polygons) > 0:
                last_mask = mask_polygons
                last_confidence = confidences
                last_detection_frame = frame_count
                
                # Calculate and store bounding box for motion detection (use first face)
                first_polygon = mask_polygons[0]
                x_coords = first_polygon[:, 0]
                y_coords = first_polygon[:, 1]
                last_bbox = (int(x_coords.min()), int(y_coords.min()), 
                            int(x_coords.max()), int(y_coords.max()))
                
                color = (0, 255, 0)  # Green for new detection
                status = "DETECTING"
                if fallback_triggered:
                    fallback_count += 1
            else:
                mask_polygons = last_mask if last_mask else []
                confidences = [c * 0.5 for c in last_confidence] if last_confidence else []
                color = (0, 0, 255)  # Red for failed detection
                status = "FAILED"
        else:
            # Use cached mask
            mask_polygons = last_mask if last_mask else []
            confidences = last_confidence if last_confidence else []
            color = (0, 255, 255)  # Yellow for tracking
            status = "TRACKING"
        
        # Create output frame based on mode
        if binary_mask_only:
            # Binary mask: only show extracted regions, rest is black
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            for mask_polygon in mask_polygons:
                if mask_polygon is not None and len(mask_polygon) > 0:
                    pts = np.array(mask_polygon, dtype=np.int32).reshape((-1, 1, 2))
                    cv2.fillPoly(mask, [pts], 255)
            # Apply mask to original frame to show only face regions
            masked_frame = cv2.bitwise_and(frame, frame, mask=mask)
            annotated_frame = masked_frame
        else:
            # Draw detection with colors and text
            annotated_frame = draw_mask(frame, mask_polygons, confidences, color, status)
            
            # Draw motion detection ROI if available
            if last_bbox is not None:
                x1, y1, x2, y2 = last_bbox
                pad_x = int((x2 - x1) * 0.2)
                pad_y = int((y2 - y1) * 0.2)
                roi_x1 = max(0, x1 - pad_x)
                roi_y1 = max(0, y1 - pad_y)
                roi_x2 = min(width, x2 + pad_x)
                roi_y2 = min(height, y2 + pad_y)
                
                # Draw ROI rectangle (cyan for motion region)
                roi_color = (255, 255, 0) if motion_detected else (128, 128, 128)
                cv2.rectangle(annotated_frame, (roi_x1, roi_y1), (roi_x2, roi_y2), roi_color, 2)
                cv2.putText(annotated_frame, "Motion ROI", (roi_x1, roi_y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, roi_color, 1)
            
            # Add frame info
            info_text = f"Frame: {frame_count}/{total_frames} | Face Motion: {motion_percentage:.1f}%"
            cv2.putText(annotated_frame, info_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        ffmpeg_process.stdin.write(annotated_frame.tobytes())
        
        # Progress update
        if frame_count % 50 == 0:
            progress = (frame_count / total_frames) * 100
            print(f"Progress: {progress:.1f}% | Detections: {detection_count} | Motion: {motion_count}")
    
    # Cleanup
    cap.release()
    ffmpeg_process.stdin.close()
    ffmpeg_process.wait()
    
    # Calculate stats
    total_time = time.time() - start_time
    detection_rate = detection_count / frame_count if frame_count > 0 else 0
    motion_rate = motion_count / frame_count if frame_count > 0 else 0
    processing_fps = frame_count / total_time if total_time > 0 else 0
    
    # Get file sizes
    input_size = os.path.getsize(video_path)
    output_size = os.path.getsize(output_path) if os.path.exists(output_path) else 0
    input_size_mb = input_size / (1024 * 1024)
    output_size_mb = output_size / (1024 * 1024)
    size_ratio = (output_size / input_size * 100) if input_size > 0 else 0
    
    # Print results
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"Total Frames: {frame_count}")
    print(f"Detections: {detection_count} ({detection_rate:.1%})")
    print(f"Motion Events: {motion_count} ({motion_rate:.1%})")
    print(f"Fallback Triggers: {fallback_count}")
    print(f"Processing Time: {total_time:.2f} seconds")
    print(f"Processing Speed: {processing_fps:.1f} FPS")
    print(f"\nFile Sizes:")
    print(f"  Input:  {input_size_mb:.2f} MB")
    print(f"  Output: {output_size_mb:.2f} MB ({size_ratio:.1f}% of input)")
    print(f"Output saved to: {output_path}")

if __name__ == "__main__":
    # Default configuration
    process_motion_detection(
        video_path="vide.avi",
        model_path="runs/segment/train/weights/best.pt",
        output_path="method3_output.mp4",
        motion_threshold=5.0,
        fallback_interval=60,
        conf_threshold=0.25
    )
