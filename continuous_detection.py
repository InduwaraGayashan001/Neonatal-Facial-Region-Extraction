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
    """Detect face in a single frame."""
    results = model.predict(frame, conf=conf_threshold, verbose=False)[0]
    
    if results.masks is None or len(results.masks) == 0:
        return None, 0.0
    
    # Get the first (largest) face mask
    mask_xy = results.masks.xy[0]
    confidence = results.boxes.conf[0].item()
    
    return mask_xy, confidence

def draw_mask(frame, mask_polygon, confidence, color=(0, 255, 0)):
    """Draw polygon mask on frame."""
    if mask_polygon is None or len(mask_polygon) == 0:
        return frame
    
    result = frame.copy()
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

def process_video(video_path, model_path, output_path, conf_threshold=0.25):
    """Process video with continuous detection."""
    
    print("="*70)
    print("METHOD 1: CONTINUOUS DETECTION (EVERY FRAME)")
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
        
        # Detect face in current frame
        mask_polygon, confidence = detect_face(model, frame, conf_threshold)
        
        if mask_polygon is not None:
            detection_count += 1
        
        # Draw detection
        annotated_frame = draw_mask(frame, mask_polygon, confidence)
        
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
    process_video(
        video_path="vid2.avi",
        model_path="runs/segment/train/weights/best.pt",
        output_path="method1_output.mp4",
        conf_threshold=0.25
    )
