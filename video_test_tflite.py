"""
Neonatal Region Extraction on Video using specific TFLite models
Processes video frames to detect and segment neonatal regions in real-time using TFLite.
"""

import cv2
import numpy as np
import argparse
import os
import time
import json
from datetime import datetime
import tensorflow as tf

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

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=False, scaleFill=False, scaleup=True, stride=32):
    """Resize and pad image to new_shape with stride-aligned dimensions."""
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)

def preprocess_image(img, imgsz=640):
    """Preprocess image for TFLite inference."""
    img_resized, ratio, (dw, dh) = letterbox(img, new_shape=(imgsz, imgsz))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_normalized = img_rgb.astype(np.float32) / 255.0
    img_expanded = np.expand_dims(img_normalized, axis=0)
    return img_expanded, ratio, (dw, dh)

def non_max_suppression(boxes, scores, iou_threshold=0.45):
    """Apply Non-Maximum Suppression."""
    if len(boxes) == 0:
        return []
    
    # Convert to format expected by NMS: [y1, x1, y2, x2]
    boxes_nms = np.array([[b[1], b[0], b[3], b[2]] for b in boxes])
    
    # Apply NMS
    selected_indices = tf.image.non_max_suppression(
        boxes_nms,
        scores,
        max_output_size=100,
        iou_threshold=iou_threshold
    )
    
    return selected_indices.numpy()

def process_mask(mask_coeffs, mask_protos, box, original_shape, imgsz=640):
    """
    Generate segmentation mask from coefficients and prototypes.
    """
    # Matrix multiply coefficients with prototypes: [160, 160, 32] @ [32] -> [160, 160]
    mask = np.matmul(mask_protos, mask_coeffs)
    
    # Apply sigmoid activation
    mask = 1 / (1 + np.exp(-mask))
    
    # Resize mask to input size (640x640)
    mask = cv2.resize(mask, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)
    
    # Crop mask to bounding box region (in normalized space)
    x1, y1, x2, y2 = box
    
    # Convert box coordinates to mask space
    x1_mask = int(x1 / original_shape[1] * imgsz)
    y1_mask = int(y1 / original_shape[0] * imgsz)
    x2_mask = int(x2 / original_shape[1] * imgsz)
    y2_mask = int(y2 / original_shape[0] * imgsz)
    
    # Clip to valid range
    x1_mask = max(0, min(x1_mask, imgsz))
    y1_mask = max(0, min(y1_mask, imgsz))
    x2_mask = max(0, min(x2_mask, imgsz))
    y2_mask = max(0, min(y2_mask, imgsz))
    
    # Threshold mask to binary
    mask_binary = (mask > 0.5).astype(np.uint8)
    
    # Resize to original image size
    mask_resized = cv2.resize(mask_binary, (original_shape[1], original_shape[0]), 
                              interpolation=cv2.INTER_NEAREST)
    
    # Find contours
    contours, _ = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        return []
    
    # Get the largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Simplify polygon
    epsilon = 0.001 * cv2.arcLength(largest_contour, True)
    polygon = cv2.approxPolyDP(largest_contour, epsilon, True)
    
    # Convert to list of [x, y] points
    polygon_xy = polygon.reshape(-1, 2).tolist()
    
    return polygon_xy

def postprocess_tflite(outputs, original_shape, ratio, padding, conf_threshold=0.25, iou_threshold=0.45, imgsz=640):
    """
    Postprocess TFLite outputs to get boxes, masks, classes, and confidences.
    """
    # Get detection output and transpose: [1, 37, 8400] -> [8400, 37]
    predictions = outputs[0][0].T  # Transpose to [num_predictions, features]
    
    # Get mask prototypes if available
    mask_protos = outputs[1][0] if len(outputs) > 1 else None  # [160, 160, 32]
    
    boxes = []
    scores = []
    classes = []
    mask_coeffs_list = []
    
    num_detections = predictions.shape[0]
    num_classes = predictions.shape[1] - 4 - 32  # Total - box coords - mask coeffs
    
    for i in range(num_detections):
        detection = predictions[i]
        
        # Extract box coordinates (first 4 values: cx, cy, w, h)
        cx, cy, w, h = detection[:4]
        
        # Extract class confidences
        class_scores = detection[4:4+num_classes]
        class_id = 0 if num_classes == 1 else np.argmax(class_scores)
        confidence = float(class_scores[class_id]) if num_classes > 0 else float(class_scores[0])
        
        # Extract mask coefficients (last 32 values)
        mask_coeffs = detection[-32:]
        
        if confidence >= conf_threshold:
            # Box coordinates are NORMALIZED (0-1), convert to pixel space first
            cx_px = cx * imgsz
            cy_px = cy * imgsz
            w_px = w * imgsz
            h_px = h * imgsz
            
            # Convert from center format to corner format (in pixel space)
            x1 = cx_px - w_px / 2
            y1 = cy_px - h_px / 2
            x2 = cx_px + w_px / 2
            y2 = cy_px + h_px / 2
            
            # Remove padding
            x1 = x1 - padding[0]
            y1 = y1 - padding[1]
            x2 = x2 - padding[0]
            y2 = y2 - padding[1]
            
            # Scale by ratio to get original image coordinates
            x1 = x1 / ratio[0]
            y1 = y1 / ratio[1]
            x2 = x2 / ratio[0]
            y2 = y2 / ratio[1]
            
            # Clip to image boundaries
            x1 = max(0, min(x1, original_shape[1]))
            y1 = max(0, min(y1, original_shape[0]))
            x2 = max(0, min(x2, original_shape[1]))
            y2 = max(0, min(y2, original_shape[0]))
            
            if x2 > x1 and y2 > y1:
                boxes.append([x1, y1, x2, y2])
                scores.append(confidence)
                classes.append(int(class_id))
                mask_coeffs_list.append(mask_coeffs)
    
    # Apply NMS
    if len(boxes) > 0:
        keep_indices = non_max_suppression(boxes, scores, iou_threshold)
        boxes = [boxes[i] for i in keep_indices]
        scores = [scores[i] for i in keep_indices]
        classes = [classes[i] for i in keep_indices]
        mask_coeffs_list = [mask_coeffs_list[i] for i in keep_indices]
    
    # Generate masks from coefficients and prototypes
    masks = []
    if mask_protos is not None and len(boxes) > 0:
        for box, mask_coeffs in zip(boxes, mask_coeffs_list):
            polygon = process_mask(mask_coeffs, mask_protos, box, original_shape, imgsz)
            masks.append(polygon)
    
    return boxes, scores, classes, masks

def create_face_mask(img, masks_xy, classes, confs):
    """Create binary mask showing only face regions, everything else black."""
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

def process_video(video_path, interpreter, args):
    """Process video frame by frame for region detection using TFLite."""
    
    # Input/Output Details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

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
        output_path = os.path.join(args.out, "detected_video_tflite.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        print(f"Output video will be saved to: {output_path}")
        
        # Setup masked video writer
        masked_path = os.path.join(args.out, "masked_faces_video_tflite.mp4")
        masked_writer = cv2.VideoWriter(masked_path, fourcc, fps, (width, height))
        print(f"Masked video will be saved to: {masked_path}")
    
    # Setup frame saving directory
    frames_dir = None
    masked_frames_dir = None
    if args.save_frames:
        frames_dir = os.path.join(args.out, "frames_tflite")
        os.makedirs(frames_dir, exist_ok=True)
        print(f"Detected frames will be saved to: {frames_dir}")
        
        masked_frames_dir = os.path.join(args.out, "masked_frames_tflite")
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
    
    print(f"\nProcessing video (TFLite)...")
    print(f"Using confidence threshold: {infer_conf}")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_start_time = time.time()
        frame_count += 1
        detection_stats["total_frames"] = frame_count
        
        # Preprocess Frame
        original_shape = frame.shape[:2]
        input_data, ratio, padding = preprocess_image(frame, args.imgsz)
        
        # Run Inference
        inference_start = time.time()
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        
        # Get Outputs
        outputs = [interpreter.get_tensor(output_details[i]['index']) 
                   for i in range(len(output_details))]
        
        inference_time = time.time() - inference_start
        detection_stats["inference_times"].append(inference_time)
        
        # Postprocess
        boxes, confs, classes, masks_xy = postprocess_tflite(
            outputs, original_shape, ratio, padding, 
            infer_conf, args.iou, args.imgsz
        )
        
        # Filter detections for face class only if using class thresholds
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
            display_frame = processed_frame
            if width > 1280:
                scale = 1280 / width
                new_width = int(width * scale)
                new_height = int(height * scale)
                display_frame = cv2.resize(processed_frame, (new_width, new_height))
            
            cv2.imshow('Neonatal Region Detection (TFLite)', display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("User requested stop.")
                break
    
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
    
    detection_stats["detection_rate"] = detection_stats["frames_with_faces"] / detection_stats["total_frames"] if detection_stats["total_frames"] > 0 else 0
    detection_stats["avg_faces_per_frame"] = detection_stats["total_faces_detected"] / detection_stats["total_frames"] if detection_stats["total_frames"] > 0 else 0
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
            "inference_engine": "TFLite",
            "detection_statistics": detection_stats,
            "processing_parameters": {
                "confidence_threshold": infer_conf,
                "iou_threshold": args.iou,
                "image_size": args.imgsz,
                "use_class_thresholds": args.use_class_thresh
            }
        }, f, indent=2)
    
    print(f"\n" + "="*60)
    print("VIDEO PROCESSING COMPLETE (TFLite)")
    print("="*60)
    print(f"Total frames processed: {detection_stats['total_frames']}")
    print(f"Frames with detections: {detection_stats['frames_with_faces']}")
    print(f"Detection rate: {detection_stats['detection_rate']:.1%}")
    print(f"\n--- PERFORMANCE METRICS ---")
    print(f"Total processing time: {detection_stats['total_processing_time']:.2f} seconds")
    print(f"Average inference time per frame: {detection_stats['avg_inference_time']*1000:.1f} ms")
    print(f"Average total processing time per frame: {detection_stats['avg_frame_processing_time']*1000:.1f} ms")
    print(f"Processing FPS achieved: {detection_stats['fps_achieved']:.1f}")
    print(f"Real-time factor: {detection_stats['real_time_factor']:.2f}x")
    if detection_stats['real_time_factor'] >= 1.0:
        print("Processing is faster than real-time!")
    else:
        print(f"Processing is {1/detection_stats['real_time_factor']:.1f}x slower than real-time")
    print(f"Statistics saved to: {stats_path}")

def main():
    parser = argparse.ArgumentParser(description="Neonatal Region Extraction on Video (TFLite)")
    parser.add_argument("--weights", required=True, help="Path to TFLite model (.tflite)")
    parser.add_argument("--video", required=True, help="Path to input video file")
    parser.add_argument("--out", default="video_output_tflite", help="Output directory")
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
    
    os.makedirs(args.out, exist_ok=True)
    
    # Load TFLite model
    print(f"Loading TFLite model from: {args.weights}")
    interpreter = tf.lite.Interpreter(model_path=args.weights)
    interpreter.allocate_tensors()
    
    process_video(args.video, interpreter, args)

if __name__ == "__main__":
    main()
