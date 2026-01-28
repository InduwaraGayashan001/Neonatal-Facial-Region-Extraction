"""
Neonatal Region Extraction using TFLite
Processes images using TensorFlow Lite model for inference.
"""

import argparse
import os
import glob
import json
import cv2
import numpy as np
from datetime import datetime
import tensorflow as tf
from collections import defaultdict

# === Neonatal Face Detection (matches data.yaml) ===
NAMES = ["Region"]

REASON_MAP = {
    0: "Region",
}

# Bright, high-contrast boundary colors (BGR)
COLORS = {
    0: (0, 255, 0),  # green
}

def image_label_from_classes(classes):
    """For Region detection, return 'Region Detected' if any regions found, else 'No Region'"""
    if len(classes) > 0:
        return "Region Detected"
    return "No Region"

def draw_polygon_masks(img, masks_xy, classes, confs, thickness=2, halo=0, label_scale=0.3):
    """Draw filled polygon masks with transparency."""
    out = img.copy()
    for poly, cls, conf in zip(masks_xy, classes, confs):
        if len(poly) == 0:
            continue
            
        pts = np.array(poly, dtype=np.int32).reshape(-1, 1, 2)
        color = COLORS.get(int(cls), (0, 255, 0))  # Default green
        
        # Draw filled polygon mask with transparency
        overlay = out.copy()
        cv2.fillPoly(overlay, [pts], color)
        cv2.addWeighted(overlay, 0.3, out, 0.7, 0, out)
        
        # Draw polygon outline
        cv2.polylines(out, [pts], True, color, thickness)

    return out

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
    
    Args:
        mask_coeffs: Mask coefficients [32]
        mask_protos: Mask prototypes [160, 160, 32]
        box: Bounding box [x1, y1, x2, y2] in original image space
        original_shape: (height, width) of original image
        imgsz: Input image size
    
    Returns:
        polygon_xy: List of [x, y] points defining the mask polygon
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
    
    YOLO v8/v11 TFLite format:
    - Output 0: [batch, features, num_predictions] - needs transpose
    - Output 1: [batch, mask_h, mask_w, mask_coeffs] - mask prototypes
    
    Features layout: [cx, cy, w, h, class_conf_0, ..., class_conf_n, mask_coeff_0, ..., mask_coeff_31]
    """
    # Get detection output and transpose: [1, 37, 8400] -> [8400, 37]
    predictions = outputs[0][0].T  # Transpose to [num_predictions, features]
    
    # Get mask prototypes if available
    mask_protos = outputs[1][0] if len(outputs) > 1 else None  # [160, 160, 32]
    
    boxes = []
    scores = []
    classes = []
    mask_coeffs_list = []
    
    # Parse each detection
    # Format: [cx, cy, w, h, class_0_conf, ..., mask_coeff_0, ...]
    # With 37 features: 4 (box) + 1 (class) + 32 (mask coeffs)
    
    num_detections = predictions.shape[0]
    num_classes = predictions.shape[1] - 4 - 32  # Total - box coords - mask coeffs
    
    for i in range(num_detections):
        detection = predictions[i]
        
        # Extract box coordinates (first 4 values: cx, cy, w, h)
        cx, cy, w, h = detection[:4]
        
        # Extract class confidences (next num_classes values)
        class_scores = detection[4:4+num_classes]
        class_id = 0 if num_classes == 1 else np.argmax(class_scores)
        confidence = float(class_scores[class_id]) if num_classes > 0 else float(class_scores[0])
        
        # Extract mask coefficients (last 32 values)
        mask_coeffs = detection[-32:]
        
        if confidence >= conf_threshold:
            # Box coordinates are NORMALIZED (0-1), convert to pixel space first
            # Multiply by input size to get pixel coordinates
            cx_px = cx * imgsz
            cy_px = cy * imgsz
            w_px = w * imgsz
            h_px = h * imgsz
            
            # Convert from center format to corner format (in pixel space)
            x1 = cx_px - w_px / 2
            y1 = cy_px - h_px / 2
            x2 = cx_px + w_px / 2
            y2 = cy_px + h_px / 2
            
            # Remove padding (if any)
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
            
            # Only add if box is valid
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

def load_yolo_labels(label_path, img_width, img_height):
    """Load YOLO format labels (normalized xywh) and convert to xyxy format."""
    boxes = []
    classes = []
    
    if not os.path.exists(label_path):
        return boxes, classes
    
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            
            class_id = int(parts[0])
            x_center = float(parts[1]) * img_width
            y_center = float(parts[2]) * img_height
            w = float(parts[3]) * img_width
            h = float(parts[4]) * img_height
            
            x1 = x_center - w / 2
            y1 = y_center - h / 2
            x2 = x_center + w / 2
            y2 = y_center + h / 2
            
            boxes.append([x1, y1, x2, y2])
            classes.append(class_id)
    
    return boxes, classes

def calculate_mask_iou(mask1_poly, mask2_poly, img_shape):
    """Calculate IoU between two polygon masks."""
    if len(mask1_poly) == 0 or len(mask2_poly) == 0:
        return 0.0
    
    # Create binary masks
    mask1 = np.zeros(img_shape, dtype=np.uint8)
    mask2 = np.zeros(img_shape, dtype=np.uint8)
    
    # Fill polygons
    pts1 = np.array(mask1_poly, dtype=np.int32).reshape(-1, 1, 2)
    pts2 = np.array(mask2_poly, dtype=np.int32).reshape(-1, 1, 2)
    
    cv2.fillPoly(mask1, [pts1], 1)
    cv2.fillPoly(mask2, [pts2], 1)
    
    # Calculate IoU
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    
    return intersection / union if union > 0 else 0.0

def calculate_iou(box1, box2):
    """Calculate IoU between two boxes [x1, y1, x2, y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0

def calculate_metrics(predictions, ground_truths, iou_threshold=0.5):
    """
    Calculate precision, recall, and mAP metrics.
    
    Args:
        predictions: List of (image_id, box, confidence, class) tuples
        ground_truths: Dict mapping image_id to list of (box, class) tuples
        iou_threshold: IoU threshold for considering a detection as correct
    
    Returns:
        dict: Metrics including precision, recall, mAP@0.5
    """
    # Sort predictions by confidence (descending)
    predictions.sort(key=lambda x: x[2], reverse=True)
    
    true_positives = []
    false_positives = []
    confidences = []
    num_ground_truths = sum(len(gt) for gt in ground_truths.values())
    
    matched_gt = defaultdict(set)  # Track which GTs have been matched
    
    for pred in predictions:
        img_id, pred_box, conf, pred_class = pred
        confidences.append(conf)
        
        if img_id not in ground_truths:
            false_positives.append(1)
            true_positives.append(0)
            continue
        
        best_iou = 0
        best_gt_idx = -1
        
        # Find best matching ground truth
        for gt_idx, (gt_box, gt_class) in enumerate(ground_truths[img_id]):
            if pred_class != gt_class:
                continue
            
            iou = calculate_iou(pred_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx
        
        # Check if match is valid and hasn't been matched before
        if best_iou >= iou_threshold and best_gt_idx not in matched_gt[img_id]:
            true_positives.append(1)
            false_positives.append(0)
            matched_gt[img_id].add(best_gt_idx)
        else:
            true_positives.append(0)
            false_positives.append(1)
    
    # Calculate cumulative TP and FP
    tp_cumsum = np.cumsum(true_positives)
    fp_cumsum = np.cumsum(false_positives)
    
    # Calculate precision and recall
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-10)
    recalls = tp_cumsum / (num_ground_truths + 1e-10)
    
    # Calculate AP using 11-point interpolation
    ap = 0
    for t in np.linspace(0, 1, 11):
        precisions_at_recall = precisions[recalls >= t]
        if len(precisions_at_recall) > 0:
            ap += np.max(precisions_at_recall)
    ap /= 11
    
    # Overall metrics
    total_tp = tp_cumsum[-1] if len(tp_cumsum) > 0 else 0
    total_fp = fp_cumsum[-1] if len(fp_cumsum) > 0 else 0
    
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / num_ground_truths if num_ground_truths > 0 else 0
    
    return {
        'precision': precision,
        'recall': recall,
        'mAP@0.5': ap,
        'true_positives': int(total_tp),
        'false_positives': int(total_fp),
        'ground_truths': num_ground_truths
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="Path to model.tflite file")
    ap.add_argument("--source", required=True, help="Folder or glob of images")
    ap.add_argument("--out", default="face_predictions_tflite", help="Output folder")
    ap.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    ap.add_argument("--iou", type=float, default=0.45, help="IoU threshold for NMS")
    ap.add_argument("--imgsz", type=int, default=640, help="Input image size")
    ap.add_argument("--outline_thickness", type=int, default=4, help="Polygon boundary thickness (px)")
    ap.add_argument("--halo", type=int, default=2, help="Extra halo around outline (px)")
    ap.add_argument("--label_scale", type=float, default=0.3, help="Label text scale")
    ap.add_argument("--calculate_metrics", action="store_true", help="Calculate evaluation metrics on test dataset")
    ap.add_argument("--labels_dir", default="test/labels", help="Directory containing ground truth labels")
    ap.add_argument("--eval_iou", type=float, default=0.5, help="IoU threshold for evaluation metrics (default: 0.5)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    ov_dir = os.path.join(args.out, "overlays")
    os.makedirs(ov_dir, exist_ok=True)
    jsonl_path = os.path.join(args.out, "predictions.jsonl")

    # Load TFLite model
    print(f"Loading TFLite model from {args.weights}")
    interpreter = tf.lite.Interpreter(model_path=args.weights)
    interpreter.allocate_tensors()

    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print(f"\nModel Information:")
    print(f"  Input shape: {input_details[0]['shape']}")
    print(f"  Input dtype: {input_details[0]['dtype']}")
    print(f"  Number of outputs: {len(output_details)}")
    for i, output in enumerate(output_details):
        print(f"  Output {i} shape: {output['shape']}")
    print()

    # Collect images
    if os.path.isdir(args.source):
        patterns = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.JPG", "*.PNG"]
        paths = []
        for p in patterns:
            paths += glob.glob(os.path.join(args.source, "**", p), recursive=True)
    else:
        paths = glob.glob(args.source, recursive=True)
    paths.sort()

    print(f"Found {len(paths)} images to process")

    # Storage for evaluation
    all_predictions = []  # List of (image_id, box, confidence, class)
    all_ground_truths = {}  # Dict mapping image_id to list of (box, class)
    all_mask_predictions = []  # List of (image_id, mask_poly, confidence, class)
    image_shapes = {}  # Store image shapes for mask IoU calculation

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for idx, p in enumerate(paths):
            if idx % 10 == 0:
                print(f"Processing {idx}/{len(paths)}: {os.path.basename(p)}")
            
            # Debug: detailed output for first image
            debug_first = (idx == 0)
            
            img = cv2.imread(p)
            if img is None:
                print(f"Warning: Could not read {p}")
                continue
            
            original_shape = img.shape[:2]
            
            # Load ground truth labels if calculating metrics
            if args.calculate_metrics:
                # Get corresponding label file
                img_basename = os.path.splitext(os.path.basename(p))[0]
                label_path = os.path.join(args.labels_dir, f"{img_basename}.txt")
                gt_boxes, gt_classes = load_yolo_labels(label_path, original_shape[1], original_shape[0])
                if gt_boxes:
                    all_ground_truths[p] = list(zip(gt_boxes, gt_classes))
            
            # Preprocess
            input_data, ratio, padding = preprocess_image(img, args.imgsz)
            
            # Run inference
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            
            # Get outputs
            outputs = [interpreter.get_tensor(output_details[i]['index']) 
                      for i in range(len(output_details))]
            
            if debug_first:
                print(f"\nDebug first image ({os.path.basename(p)}):")
                print(f"  Original shape: {original_shape}")
                print(f"  Ratio: {ratio}, Padding: {padding}")
                print(f"  Output shapes: {[out.shape for out in outputs]}")
                if len(outputs[0].shape) > 2:
                    print(f"  Output[0] sample (first detection): {outputs[0][0][0][:10] if outputs[0].shape[1] > 0 else 'empty'}")
            
            # Postprocess
            boxes, confs, classes, masks_xy = postprocess_tflite(
                outputs, original_shape, ratio, padding, 
                args.conf, args.iou, args.imgsz
            )
            
            if debug_first:
                print(f"  Detections found: {len(boxes)}")
                if len(boxes) > 0:
                    print(f"  First box (pixels): {boxes[0]}")
                    print(f"  First conf: {confs[0]:.3f}")
                    print(f"  First class: {classes[0]}")
                    print(f"  Image dimensions: {original_shape}")
                    print(f"  Number of mask polygons: {len(masks_xy)}")
                    if len(masks_xy) > 0 and len(masks_xy[0]) > 0:
                        print(f"  First mask polygon points: {len(masks_xy[0])}")
                    # Load and print ground truth for comparison
                    img_basename = os.path.splitext(os.path.basename(p))[0]
                    label_path = os.path.join(args.labels_dir, f"{img_basename}.txt")
                    if os.path.exists(label_path):
                        gt_boxes, gt_classes = load_yolo_labels(label_path, original_shape[1], original_shape[0])
                        if gt_boxes:
                            print(f"  Ground truth box: {gt_boxes[0]}")
                            # Calculate IoU for debugging
                            iou = calculate_iou(boxes[0], gt_boxes[0])
                            print(f"  IoU with ground truth: {iou:.3f} (eval threshold: {args.eval_iou})")
                            # Also check mask IoU if available
                            if len(masks_xy) > 0 and len(masks_xy[0]) > 0:
                                # Convert GT box to simple polygon for comparison
                                x1, y1, x2, y2 = gt_boxes[0]
                                gt_poly = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
                                mask_iou = calculate_mask_iou(masks_xy[0], gt_poly, original_shape)
                                print(f"  Mask IoU with GT box region: {mask_iou:.3f}")
                else:
                    print(f"  No detections above confidence threshold {args.conf}")
            
            # Keep only the largest region if multiple detected
            if len(boxes) > 1:
                face_areas = []
                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = box
                    area = (x2 - x1) * (y2 - y1)
                    face_areas.append((area, i))
                face_areas.sort(reverse=True)
                keep_idx = face_areas[0][1]
                boxes = [boxes[keep_idx]]
                confs = [confs[keep_idx]]
                classes = [classes[keep_idx]]
                if masks_xy:
                    masks_xy = [masks_xy[keep_idx]]
            
            # Store predictions for evaluation
            if args.calculate_metrics:
                image_shapes[p] = original_shape
                for box, conf, cls in zip(boxes, confs, classes):
                    all_predictions.append((p, box, conf, cls))
                # Store mask predictions if available
                for i, (conf, cls) in enumerate(zip(confs, classes)):
                    if i < len(masks_xy) and len(masks_xy[i]) > 0:
                        all_mask_predictions.append((p, masks_xy[i], conf, cls))
            
            # Image-level label
            label = image_label_from_classes(classes)
            
            # Build detections
            dets = []
            for i, (box, cls, cf) in enumerate(zip(boxes, classes, confs)):
                class_id = int(cls)
                class_name = NAMES[class_id] if class_id < len(NAMES) else f"Unknown_{class_id}"
                reason = REASON_MAP.get(class_id, class_name)
                
                x1, y1, x2, y2 = box
                det_data = {
                    "class_id": class_id,
                    "class_name": class_name,
                    "reason": reason,
                    "confidence": float(cf),
                    "bounding_box": {
                        "x1": float(x1),
                        "y1": float(y1),
                        "x2": float(x2),
                        "y2": float(y2),
                        "width": float(x2 - x1),
                        "height": float(y2 - y1)
                    }
                }
                
                if i < len(masks_xy) and masks_xy[i]:
                    det_data["polygon_xy"] = [[float(x), float(y)] for x, y in masks_xy[i]]
                
                dets.append(det_data)
            
            rec = {
                "image": p,
                "pred_image_label": label,
                "detections": dets,
                "conf_used": args.conf,
                "iou": args.iou,
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "inference_engine": "TFLite"
            }
            f.write(json.dumps(rec) + "\n")
            
            # Draw visualization
            if len(dets) > 0 and len(masks_xy) > 0:
                vis = draw_polygon_masks(
                    img, masks_xy, classes, confs,
                    thickness=args.outline_thickness,
                    halo=args.halo,
                    label_scale=args.label_scale
                )
            else:
                vis = img
            
            outp = os.path.join(ov_dir, os.path.basename(p))
            cv2.imwrite(outp, vis)

    print(f"\nProcessing complete!")
    print(f"Saved JSONL: {jsonl_path}")
    print(f"Saved overlays to: {ov_dir}")
    
    # Calculate and display metrics
    if args.calculate_metrics:
        print("\n" + "="*60)
        print("EVALUATION METRICS (TFLite Model)")
        print("="*60)
        
        if len(all_predictions) > 0 and len(all_ground_truths) > 0:
            # Calculate metrics at specified IoU threshold
            metrics_eval = calculate_metrics(all_predictions.copy(), all_ground_truths, iou_threshold=args.eval_iou)
            
            # Calculate metrics at IoU 0.5 for comparison
            metrics_50 = calculate_metrics(all_predictions.copy(), all_ground_truths, iou_threshold=0.5)
            
            # Calculate metrics at IoU 0.75
            metrics_75 = calculate_metrics(all_predictions.copy(), all_ground_truths, iou_threshold=0.75)
            
            # Calculate mAP@0.5:0.95 (average over IoU thresholds 0.5 to 0.95)
            map_scores = []
            for iou_thresh in np.linspace(0.5, 0.95, 10):
                metrics = calculate_metrics(all_predictions.copy(), all_ground_truths, iou_threshold=iou_thresh)
                map_scores.append(metrics['mAP@0.5'])
            map_50_95 = np.mean(map_scores)
            
            print("\nBox Detection Metrics:")
            print("-" * 40)
            if args.eval_iou != 0.5:
                print(f"At IoU={args.eval_iou}:")
                print(f"  Precision:      {metrics_eval['precision']:.4f}")
                print(f"  Recall:         {metrics_eval['recall']:.4f}")
                print(f"  mAP:            {metrics_eval['mAP@0.5']:.4f}")
                print(f"\nAt IoU=0.5 (standard):")
            print(f"Precision:        {metrics_50['precision']:.4f}")
            print(f"Recall:           {metrics_50['recall']:.4f}")
            print(f"mAP@0.5:          {metrics_50['mAP@0.5']:.4f}")
            print(f"mAP@0.75:         {metrics_75['mAP@0.5']:.4f}")
            print(f"mAP@0.5:0.95:     {map_50_95:.4f}")
            print(f"\nDetection Summary:")
            print(f"True Positives:   {metrics_50['true_positives']}")
            print(f"False Positives:  {metrics_50['false_positives']}")
            print(f"Ground Truths:    {metrics_50['ground_truths']}")
            print(f"Total Images:     {len(paths)}")
            print(f"Images with GT:   {len(all_ground_truths)}")
                        # Calculate mask-based metrics if masks are available
            if len(all_mask_predictions) > 0:
                print(f"\n{'='*40}")
                print("MASK SEGMENTATION QUALITY")
                print(f"{'='*40}")
                print(f"Total predictions with masks: {len(all_mask_predictions)}")
                
                # Calculate average mask IoU with ground truth boxes
                mask_ious = []
                for img_id, mask_poly, conf, cls in all_mask_predictions:
                    if img_id in all_ground_truths and len(all_ground_truths[img_id]) > 0:
                        gt_box, gt_cls = all_ground_truths[img_id][0]
                        # Convert GT box to polygon
                        x1, y1, x2, y2 = gt_box
                        gt_poly = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
                        mask_iou = calculate_mask_iou(mask_poly, gt_poly, image_shapes[img_id])
                        mask_ious.append(mask_iou)
                
                if mask_ious:
                    avg_mask_iou = np.mean(mask_ious)
                    print(f"Average Mask IoU: {avg_mask_iou:.4f}")
                    print(f"Masks with IoU > 0.5: {sum(1 for iou in mask_ious if iou > 0.5)} / {len(mask_ious)}")
                    print(f"Masks with IoU > 0.3: {sum(1 for iou in mask_ious if iou > 0.3)} / {len(mask_ious)}")
                        # Save detailed results
            results_path = os.path.join(args.out, "evaluation_results.json")
            eval_results = {
                "evaluation_timestamp": datetime.utcnow().isoformat() + "Z",
                "model_weights": args.weights,
                "inference_engine": "TFLite",
                "confidence_threshold": args.conf,
                "iou_threshold": args.iou,
                "image_size": args.imgsz,
                "box_metrics": {
                    "precision": float(metrics_50['precision']),
                    "recall": float(metrics_50['recall']),
                    "mAP_50": float(metrics_50['mAP@0.5']),
                    "mAP_75": float(metrics_75['mAP@0.5']),
                    "mAP_50_95": float(map_50_95)
                },
                "detection_summary": {
                    "true_positives": metrics_50['true_positives'],
                    "false_positives": metrics_50['false_positives'],
                    "ground_truths": metrics_50['ground_truths'],
                    "total_images": len(paths),
                    "images_with_ground_truth": len(all_ground_truths)
                }
            }
            
            with open(results_path, 'w') as f:
                json.dump(eval_results, f, indent=2)
            
            print(f"\nDetailed evaluation results saved to: {results_path}")
            print("="*60)
        else:
            print("\nWarning: No predictions or ground truth labels found.")
            print(f"Predictions: {len(all_predictions)}, Ground truths: {len(all_ground_truths)}")
            if len(all_ground_truths) == 0:
                print(f"Check if labels exist in: {args.labels_dir}")

if __name__ == "__main__":
    main()
