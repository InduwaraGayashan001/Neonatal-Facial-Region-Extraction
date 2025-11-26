"""
Method 3: Motion-Triggered Detection (with fallbacks)
Detects faces only when motion is detected, with safety fallbacks.
"""

import cv2
import numpy as np
import time
import os
from ultralytics import YOLO

def detect_face(model, frame, conf_threshold=0.25):
    """Detect face in a single frame."""
    results = model.predict(frame, conf=conf_threshold, verbose=False)[0]
    
    if results.masks is None or len(results.masks) == 0:
        return None, 0.0
    
    mask_xy = results.masks.xy[0]
    confidence = results.boxes.conf[0].item()
    
    return mask_xy, confidence

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

def process_video(video_path, model_path, output_path, motion_threshold=5.0, 
                 fallback_interval=60, conf_threshold=0.25):
    """Process video with motion-triggered detection and fallbacks."""
    
    print("="*70)
    print("METHOD 3: MOTION-TRIGGERED DETECTION (WITH FALLBACKS)")
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
    
    # Setup output
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
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
            mask_polygon, confidence = detect_face(model, frame, conf_threshold)
            detection_count += 1
            
            if mask_polygon is not None:
                last_mask = mask_polygon
                last_confidence = confidence
                last_detection_frame = frame_count
                
                # Calculate and store bounding box for motion detection
                x_coords = mask_polygon[:, 0]
                y_coords = mask_polygon[:, 1]
                last_bbox = (int(x_coords.min()), int(y_coords.min()), 
                            int(x_coords.max()), int(y_coords.max()))
                
                color = (0, 255, 0)  # Green for new detection
                status = "DETECTING"
                if fallback_triggered:
                    fallback_count += 1
            else:
                mask_polygon = last_mask
                confidence = last_confidence * 0.5
                color = (0, 0, 255)  # Red for failed detection
                status = "FAILED"
        else:
            # Use cached mask
            mask_polygon = last_mask
            confidence = last_confidence
            color = (0, 255, 255)  # Yellow for tracking
            status = "TRACKING"
        
        # Draw detection
        annotated_frame = draw_mask(frame, mask_polygon, confidence, color, status)
        
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
        
        out.write(annotated_frame)
        
        # Progress update
        if frame_count % 50 == 0:
            progress = (frame_count / total_frames) * 100
            print(f"Progress: {progress:.1f}% | Detections: {detection_count} | Motion: {motion_count}")
    
    # Cleanup
    cap.release()
    out.release()
    
    # Calculate stats
    total_time = time.time() - start_time
    detection_rate = detection_count / frame_count if frame_count > 0 else 0
    motion_rate = motion_count / frame_count if frame_count > 0 else 0
    processing_fps = frame_count / total_time if total_time > 0 else 0
    
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
    print(f"Output saved to: {output_path}")

if __name__ == "__main__":
    # Default configuration
    process_video(
        video_path="vid2.avi",
        model_path="runs/segment/train/weights/best.pt",
        output_path="method3_output.mp4",
        motion_threshold=1.0,
        fallback_interval=60,
        conf_threshold=0.25
    )
