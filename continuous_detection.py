"""
Method 1: Continuous Detection (Every Frame)
Detects faces in every single frame for maximum accuracy.
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

def draw_mask(frame, mask_polygons, confidences, color=(0, 255, 0)):
    """Draw multiple polygon masks on frame."""
    if not mask_polygons or len(mask_polygons) == 0:
        return frame
    
    result = frame.copy()
    
    # Draw each mask
    for mask_polygon, confidence in zip(mask_polygons, confidences):
        if mask_polygon is None or len(mask_polygon) == 0:
            continue
            
        pts = np.array(mask_polygon, dtype=np.int32).reshape((-1, 1, 2))
        
        # Draw filled polygon
        overlay = result.copy()
        cv2.fillPoly(overlay, [pts], color)
        cv2.addWeighted(overlay, 0.3, result, 0.7, 0, result)
        
        # Add confidence text
        if len(mask_polygon) > 0:
            x_min = int(np.min(mask_polygon[:, 0]))
            y_min = int(np.min(mask_polygon[:, 1]))
            cv2.putText(result, f"Face: {confidence:.2f}", (x_min, y_min - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    return result

def process_continuous_detection(video_path, model_path, output_path, conf_threshold=0.25, binary_mask_only=True):
    """Process video with continuous detection.
    
    Args:
        binary_mask_only: If True, outputs only white mask on black background (no colors, no text)
    """
    
    print("="*70)
    print("METHOD 1: CONTINUOUS DETECTION (EVERY FRAME)")
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
    
    # Setup output
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Process frames
    start_time = time.time()
    frame_count = 0
    detection_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Detect all faces in current frame
        mask_polygons, confidences = detect_face(model, frame, conf_threshold)
        
        if len(mask_polygons) > 0:
            detection_count += len(mask_polygons)
        
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
            annotated_frame = draw_mask(frame, mask_polygons, confidences)
            
            # Add frame info
            status = f"Frame: {frame_count}/{total_frames} | DETECTING"
            cv2.putText(annotated_frame, status, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        out.write(annotated_frame)
        
        # Progress update
        if frame_count % 50 == 0:
            progress = (frame_count / total_frames) * 100
            print(f"Progress: {progress:.1f}% | Detections: {detection_count}")
    
    # Cleanup
    cap.release()
    out.release()
    
    # Calculate stats
    total_time = time.time() - start_time
    detection_rate = detection_count / frame_count if frame_count > 0 else 0
    processing_fps = frame_count / total_time if total_time > 0 else 0
    
    # Print results
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"Total Frames: {frame_count}")
    print(f"Detections: {detection_count} ({detection_rate:.1%})")
    print(f"Processing Time: {total_time:.2f} seconds")
    print(f"Processing Speed: {processing_fps:.1f} FPS")
    print(f"Output saved to: {output_path}")

if __name__ == "__main__":
    # Default configuration
    process_continuous_detection(
        video_path="video/vid2.avi",
        model_path="runs/segment/train/weights/best.pt",
        output_path="method1_output.mp4",
        conf_threshold=0.25
    )
