"""
Method 4: Interval-based Detection
Detects faces at regular intervals and tracks between detections.
"""

import cv2
import numpy as np
import time
import os
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

def process_interval_tracking(video_path, model_path, output_path, detection_interval=30, conf_threshold=0.25, binary_mask_only=True):
    """Process video with interval-based detection.
    
    Args:
        binary_mask_only: If True, outputs only white mask on black background (no colors, no text)
    """
    
    print("="*70)
    print("METHOD 4: INTERVAL-BASED DETECTION")
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
    print(f"Detection Interval: {detection_interval} frames")
    
    # Setup output
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Tracking variables
    last_mask = None
    last_confidence = 0.0
    frames_since_detection = 0
    
    # Stats
    start_time = time.time()
    frame_count = 0
    detection_count = 0
    tracking_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        frames_since_detection += 1
        
        # Decide whether to detect
        should_detect = (last_mask is None) or (frames_since_detection >= detection_interval)
        
        if should_detect:
            # Run detection
            mask_polygons, confidences = detect_face(model, frame, conf_threshold)
            detection_count += len(mask_polygons) if mask_polygons else 0
            frames_since_detection = 0
            
            if len(mask_polygons) > 0:
                last_mask = mask_polygons
                last_confidence = confidences
                color = (0, 255, 0)  # Green for detection
                status = "DETECTING"
            else:
                mask_polygons = last_mask if last_mask else []
                confidences = [c * 0.5 for c in last_confidence] if last_confidence else []
                color = (0, 0, 255)  # Red for failed detection
                status = "FAILED"
        else:
            # Use cached mask
            mask_polygons = last_mask if last_mask else []
            confidences = last_confidence if last_confidence else []
            tracking_count += 1
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
            
            # Add frame info
            next_detection = detection_interval - frames_since_detection
            info_text = f"Frame: {frame_count}/{total_frames} | Next: {next_detection}"
            cv2.putText(annotated_frame, info_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        out.write(annotated_frame)
        
        # Progress update
        if frame_count % 50 == 0:
            progress = (frame_count / total_frames) * 100
            print(f"Progress: {progress:.1f}% | Detections: {detection_count} | Tracking: {tracking_count}")
    
    # Cleanup
    cap.release()
    out.release()
    
    # Calculate stats
    total_time = time.time() - start_time
    detection_rate = detection_count / frame_count if frame_count > 0 else 0
    tracking_rate = tracking_count / frame_count if frame_count > 0 else 0
    processing_fps = frame_count / total_time if total_time > 0 else 0
    
    # Print results
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"Total Frames: {frame_count}")
    print(f"Detections: {detection_count} ({detection_rate:.1%})")
    print(f"Tracking Frames: {tracking_count} ({tracking_rate:.1%})")
    print(f"Processing Time: {total_time:.2f} seconds")
    print(f"Processing Speed: {processing_fps:.1f} FPS")
    print(f"Speedup Factor: {1/detection_rate:.1f}x")
    print(f"Output saved to: {output_path}")

if __name__ == "__main__":
    # Default configuration
    process_interval_tracking(
        video_path="video/vid2.avi",
        model_path="runs/segment/train/weights/best.pt",
        output_path="method4_output.mp4",
        detection_interval=30,
        conf_threshold=0.25
    )
