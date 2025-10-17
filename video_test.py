"""
Neonatal Region Extraction on Video using YOLO11 Segmentation
Processes video frames to detect and segment neonatal regions in real-time.
"""

import cv2
import numpy as np
import argparse
import os
import time
from ultralytics import YOLO
from datetime import datetime
import json

# === Neonatal Region Detection (matches data.yaml) ===
NAMES = ["Region"]

REASON_MAP = {
    0: "Region",
}

# Bright, high-contrast boundary colors (BGR)
COLORS = {
    0: (0, 255, 0),  # green
}

# Per-class thresholds if you want recall-first behavior
CLASS_THRESH = {0: 0.5}

def create_face_mask(img, masks_xy, classes, confs):
    """Create binary mask showing only face regions, everything else black."""
    # Create a black frame of the same size
    masked_frame = np.zeros_like(img)
    
    for poly, cls, conf in zip(masks_xy, classes, confs):
        if len(poly) == 0:
            continue
            
        # Create binary mask for this polygon
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        pts = np.array(poly, dtype=np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(mask, [pts], 255)
        
        # Apply mask to original image and add to result
        face_region = cv2.bitwise_and(img, img, mask=mask)
        masked_frame = cv2.bitwise_or(masked_frame, face_region)
    
    return masked_frame

def draw_polygon_masks(img, masks_xy, classes, confs, thickness=2, halo=0, label_scale=0.3):
    """Draw filled polygon masks without text labels or boundary lines."""
    out = img.copy()
    for poly, cls, conf in zip(masks_xy, classes, confs):
        if len(poly) == 0:
            continue
            
        pts = np.array(poly, dtype=np.int32).reshape(-1, 1, 2)
        color = COLORS.get(int(cls), (0, 255, 0))  # Default green for region
        
        # Draw only filled polygon mask with some transparency
        overlay = out.copy()
        cv2.fillPoly(overlay, [pts], color)
        cv2.addWeighted(overlay, 0.3, out, 0.7, 0, out)

    return out

def process_video(video_path, model, args):
    """Process video frame by frame for face detection."""
    
    # Start timing
    start_time = time.time()
    
    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video Properties:")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps}")
    print(f"  Total Frames: {total_frames}")
    print(f"  Duration: {total_frames/fps:.2f} seconds")
    
    # Setup output video writer if saving is enabled
    out_writer = None
    masked_writer = None
    if args.save_video:
        output_path = os.path.join(args.out, "detected_video.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        print(f"Output video will be saved to: {output_path}")
        
        # Setup masked video writer
        masked_path = os.path.join(args.out, "masked_faces_video.mp4")
        masked_writer = cv2.VideoWriter(masked_path, fourcc, fps, (width, height))
        print(f"Masked faces video will be saved to: {masked_path}")
    
    # Setup frame saving directory
    frames_dir = None
    masked_frames_dir = None
    if args.save_frames:
        frames_dir = os.path.join(args.out, "frames")
        os.makedirs(frames_dir, exist_ok=True)
        print(f"Detected frames will be saved to: {frames_dir}")
        
        masked_frames_dir = os.path.join(args.out, "masked_frames")
        os.makedirs(masked_frames_dir, exist_ok=True)
        print(f"Masked frames will be saved to: {masked_frames_dir}")
    
    # Detection statistics
    detection_stats = {
        "total_frames": 0,
        "frames_with_faces": 0,
        "total_faces_detected": 0,
        "detections_per_frame": [],
        "processing_times": [],
        "inference_times": []
    }
    
    # Process frame by frame
    frame_count = 0
    infer_conf = (min(CLASS_THRESH.values()) if args.use_class_thresh else args.conf)
    
    print(f"\nProcessing video...")
    print(f"Using confidence threshold: {infer_conf}")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_start_time = time.time()
        frame_count += 1
        detection_stats["total_frames"] = frame_count
        
        # Run inference on frame
        inference_start = time.time()
        results = model.predict(frame, conf=infer_conf, iou=args.iou, imgsz=args.imgsz, verbose=False)[0]
        inference_time = time.time() - inference_start
        detection_stats["inference_times"].append(inference_time)
        
        # Extract detection data
        classes = results.boxes.cls.cpu().numpy().astype(int).tolist() if results.boxes is not None else []
        confs = results.boxes.conf.cpu().numpy().tolist() if results.boxes is not None else []
        masks_xy = results.masks.xy if (results.masks is not None) else []
        
        # Filter detections for face class only and apply confidence thresholds
        valid_indices = []
        for i, (cls, conf) in enumerate(zip(classes, confs)):
            if cls == 0:  # Region class
                if args.use_class_thresh:
                    if conf >= CLASS_THRESH.get(cls, args.conf):
                        valid_indices.append(i)
                else:
                    if conf >= args.conf:
                        valid_indices.append(i)
        
        # Filter arrays based on valid indices
        filtered_classes = [classes[i] for i in valid_indices]
        filtered_confs = [confs[i] for i in valid_indices]
        filtered_masks = [masks_xy[i] for i in valid_indices if i < len(masks_xy)]
        
        # Select largest face if multiple detected
        if len(filtered_classes) > 1 and len(filtered_masks) > 0:
            areas = []
            for mask in filtered_masks:
                if len(mask) > 0:
                    # Calculate polygon area
                    pts = np.array(mask, dtype=np.float32)
                    area = cv2.contourArea(pts)
                    areas.append(area)
                else:
                    areas.append(0)
            
            if areas:
                max_idx = np.argmax(areas)
                filtered_classes = [filtered_classes[max_idx]]
                filtered_confs = [filtered_confs[max_idx]]
                filtered_masks = [filtered_masks[max_idx]]
        
        # Update statistics
        num_faces = len(filtered_classes)
        detection_stats["total_faces_detected"] += num_faces
        detection_stats["detections_per_frame"].append(num_faces)
        
        if num_faces > 0:
            detection_stats["frames_with_faces"] += 1
        
        # Create both detection overlay and binary masked frames
        processed_frame = frame.copy()
        masked_frame = np.zeros_like(frame)  # Default black frame
        
        if num_faces > 0:
            # Create detection overlay with polygon outlines
            processed_frame = draw_polygon_masks(
                frame, filtered_masks, filtered_classes, filtered_confs,
                thickness=args.outline_thickness,
                halo=args.halo,
                label_scale=args.label_scale
            )
            
            # Create binary masked frame (only face regions visible)
            masked_frame = create_face_mask(frame, filtered_masks, filtered_classes, filtered_confs)
        
        # Save processed frame to videos
        if out_writer is not None:
            out_writer.write(processed_frame)
        if masked_writer is not None:
            masked_writer.write(masked_frame)
        
        # Save frames if faces detected and frame saving is enabled
        if args.save_frames and num_faces > 0:
            # Save detection overlay frame
            frame_filename = f"frame_{frame_count:06d}_faces_{num_faces}.jpg"
            frame_path = os.path.join(frames_dir, frame_filename)
            cv2.imwrite(frame_path, processed_frame)
            
            # Save masked frame
            masked_filename = f"masked_frame_{frame_count:06d}_faces_{num_faces}.jpg"
            masked_path = os.path.join(masked_frames_dir, masked_filename)
            cv2.imwrite(masked_path, masked_frame)
        
        # Calculate frame processing time
        frame_processing_time = time.time() - frame_start_time
        detection_stats["processing_times"].append(frame_processing_time)
        
        # Display progress
        if frame_count % 30 == 0 or frame_count == total_frames:
            progress = (frame_count / total_frames) * 100
            avg_inference_time = np.mean(detection_stats["inference_times"][-30:]) if detection_stats["inference_times"] else 0
            avg_processing_time = np.mean(detection_stats["processing_times"][-30:]) if detection_stats["processing_times"] else 0
            print(f"Progress: {frame_count}/{total_frames} frames ({progress:.1f}%) - Faces: {num_faces} - Inference: {avg_inference_time*1000:.1f}ms - Total: {avg_processing_time*1000:.1f}ms")
        
        # Real-time display (optional)
        if args.show_realtime:
            # Resize for display if frame is too large
            display_frame = processed_frame
            if width > 1280:
                scale = 1280 / width
                new_width = int(width * scale)
                new_height = int(height * scale)
                display_frame = cv2.resize(processed_frame, (new_width, new_height))
            
            cv2.imshow('Neonatal Face Detection', display_frame)
            
            # Press 'q' to quit, 's' to save current frame
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("User requested stop.")
                break
            elif key == ord('s') and num_faces > 0:
                save_path = os.path.join(args.out, f"manual_save_frame_{frame_count}.jpg")
                cv2.imwrite(save_path, processed_frame)
                print(f"Saved frame to: {save_path}")
    
    # Cleanup
    cap.release()
    if out_writer is not None:
        out_writer.release()
    if masked_writer is not None:
        masked_writer.release()
    if args.show_realtime:
        cv2.destroyAllWindows()
    
    # Calculate processing time statistics
    end_time = time.time()
    total_processing_time = end_time - start_time
    
    # Calculate timing statistics
    detection_stats["detection_rate"] = detection_stats["frames_with_faces"] / detection_stats["total_frames"] if detection_stats["total_frames"] > 0 else 0
    detection_stats["avg_faces_per_frame"] = detection_stats["total_faces_detected"] / detection_stats["total_frames"] if detection_stats["total_frames"] > 0 else 0
    
    # Add timing statistics
    detection_stats["total_processing_time"] = total_processing_time
    detection_stats["avg_inference_time"] = np.mean(detection_stats["inference_times"]) if detection_stats["inference_times"] else 0
    detection_stats["avg_frame_processing_time"] = np.mean(detection_stats["processing_times"]) if detection_stats["processing_times"] else 0
    detection_stats["fps_achieved"] = detection_stats["total_frames"] / total_processing_time if total_processing_time > 0 else 0
    detection_stats["real_time_factor"] = detection_stats["fps_achieved"] / fps if fps > 0 else 0
    
    stats_path = os.path.join(args.out, "detection_statistics.json")
    with open(stats_path, 'w') as f:
        json.dump({
            "video_file": video_path,
            "processing_timestamp": datetime.utcnow().isoformat() + "Z",
            "model_weights": args.weights,
            "detection_statistics": detection_stats,
            "processing_parameters": {
                "confidence_threshold": infer_conf,
                "iou_threshold": args.iou,
                "image_size": args.imgsz,
                "use_class_thresholds": args.use_class_thresh
            }
        }, f, indent=2)
    
    # Print final statistics
    print(f"\n" + "="*60)
    print("VIDEO PROCESSING COMPLETE")
    print("="*60)
    print(f"Total frames processed: {detection_stats['total_frames']}")
    print(f"Frames with faces: {detection_stats['frames_with_faces']}")
    print(f"Detection rate: {detection_stats['detection_rate']:.1%}")
    print(f"Total faces detected: {detection_stats['total_faces_detected']}")
    print(f"Average faces per frame: {detection_stats['avg_faces_per_frame']:.2f}")
    print(f"\n--- PERFORMANCE METRICS ---")
    print(f"Total processing time: {detection_stats['total_processing_time']:.2f} seconds")
    print(f"Average inference time per frame: {detection_stats['avg_inference_time']*1000:.1f} ms")
    print(f"Average total processing time per frame: {detection_stats['avg_frame_processing_time']*1000:.1f} ms")
    print(f"Processing FPS achieved: {detection_stats['fps_achieved']:.1f}")
    print(f"Real-time factor: {detection_stats['real_time_factor']:.2f}x")
    print(f"Video FPS: {fps}")
    if detection_stats['real_time_factor'] >= 1.0:
        print("Processing is faster than real-time!")
    else:
        print(f"Processing is {1/detection_stats['real_time_factor']:.1f}x slower than real-time")
    print(f"Statistics saved to: {stats_path}")

def main():
    parser = argparse.ArgumentParser(description="Neonatal Region Extraction on Video")
    parser.add_argument("--weights", required=True, help="Path to trained model weights (.pt file)")
    parser.add_argument("--video", required=True, help="Path to input video file")
    parser.add_argument("--out", default="video_output", help="Output directory")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size")
    parser.add_argument("--use_class_thresh", action="store_true", help="Use per-class thresholds")
    parser.add_argument("--outline_thickness", type=int, default=2, help="Outline thickness")
    parser.add_argument("--halo", type=int, default=0, help="Halo size")
    parser.add_argument("--label_scale", type=float, default=0.3, help="Label scale")
    parser.add_argument("--save_video", action="store_true", help="Save processed video")
    parser.add_argument("--save_frames", action="store_true", help="Save frames with detections")
    parser.add_argument("--show_realtime", action="store_true", help="Show real-time processing")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.out, exist_ok=True)
    
    # Load model
    print(f"Loading model from: {args.weights}")
    model = YOLO(args.weights)
    
    # Process video
    process_video(args.video, model, args)

if __name__ == "__main__":
    main()