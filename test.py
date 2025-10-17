"""
Neonatal Region Extraction using YOLO11 Segmentation
Processes images to detect and segment neonatal regions with confidence scores and polygon masks.
Calculates loss metrics for test dataset evaluation.
"""

from ultralytics import YOLO
import argparse, os, glob, json, cv2, numpy as np
from datetime import datetime

# === Neonatal Face Detection (matches data.yaml) ===
NAMES = ["Region"]

REASON_MAP = {
    0: "Region",
}

# Bright, high-contrast boundary colors (BGR)
COLORS = {
    0: (0, 255, 0),  # green
}

# Per-class thresholds if you want recall-first behavior (enabled with --use_class_thresh)
CLASS_THRESH = {0: 0.5}

def image_label_from_classes(classes):
    """For Region detection, return 'Region Detected' if any regions found, else 'No Region'"""
    if len(classes) > 0:
        return "Region Detected"
    return "No Region"

def draw_polygon_masks(img, masks_xy, classes, confs, thickness=2, halo=0, label_scale=0.3):
    """Draw filled polygon masks without text labels or boundary lines."""
    out = img.copy()
    for poly, cls, conf in zip(masks_xy, classes, confs):
        if len(poly) == 0:
            continue
            
        pts = np.array(poly, dtype=np.int32).reshape(-1, 1, 2)
        color = COLORS.get(int(cls), (0, 255, 0))  # Default green for face
        
        # Draw only filled polygon mask with some transparency
        overlay = out.copy()
        cv2.fillPoly(overlay, [pts], color)
        cv2.addWeighted(overlay, 0.3, out, 0.7, 0, out)

    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="Path to best.pt or yolo11n-seg.pt")
    ap.add_argument("--source", required=True, help="Folder or glob of images")
    ap.add_argument("--out", default="face_predictions", help="Output folder")
    ap.add_argument("--conf", type=float, default=0.25, help="Global confidence (ignored if --use_class_thresh)")
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--use_class_thresh", action="store_true",
                    help="Use per-class thresholds to favor recall")
    ap.add_argument("--outline_thickness", type=int, default=4, help="Polygon boundary thickness (px)")
    ap.add_argument("--halo", type=int, default=2, help="Extra halo around outline for visibility (px)")
    ap.add_argument("--label_scale", type=float, default=0.3, help="Label text scale")
    ap.add_argument("--calculate_loss", action="store_true", help="Calculate loss metrics on test dataset")
    ap.add_argument("--data_yaml", default="data.yaml", help="Path to data.yaml file for loss calculation")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    ov_dir = os.path.join(args.out, "overlays"); os.makedirs(ov_dir, exist_ok=True)
    jsonl_path = os.path.join(args.out, "predictions.jsonl")

    model = YOLO(args.weights)

    # collect images
    if os.path.isdir(args.source):
        patterns = ["*.jpg","*.jpeg","*.png","*.bmp","*.JPG","*.PNG"]
        paths = []
        for p in patterns:
            paths += glob.glob(os.path.join(args.source, "**", p), recursive=True)
    else:
        paths = glob.glob(args.source, recursive=True)
    paths.sort()

    infer_conf = (min(CLASS_THRESH.values()) if args.use_class_thresh else args.conf)

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for p in paths:
            r = model.predict(p, conf=infer_conf, iou=args.iou, imgsz=args.imgsz, verbose=False)[0]
            classes = r.boxes.cls.cpu().numpy().astype(int).tolist() if r.boxes is not None else []
            confs   = r.boxes.conf.cpu().numpy().tolist() if r.boxes is not None else []
            boxes   = r.boxes.xyxy.cpu().numpy().tolist() if r.boxes is not None else []
            masks_xy= r.masks.xy if (r.masks is not None) else []

            # Filter detections
            if args.use_class_thresh:
                keep = [i for i,(c,cf) in enumerate(zip(classes, confs)) if cf >= CLASS_THRESH.get(int(c), args.conf)]
            else:
                keep = [i for i,cf in enumerate(confs) if cf >= args.conf]

            # Additional filter: only keep region detections (class 0) for neonatal region extraction
            keep = [i for i in keep if classes[i] == 0]

            # If multiple regions detected, keep only the largest one (by bounding box area)
            if len(keep) > 1:
                face_areas = []
                for i in keep:
                    x1, y1, x2, y2 = boxes[i]
                    area = (x2 - x1) * (y2 - y1)
                    face_areas.append((area, i))
                # Sort by area (descending) and keep only the largest
                face_areas.sort(reverse=True)
                keep = [face_areas[0][1]]  # Keep only the index of the largest region
            
            classes = [classes[i] for i in keep]
            confs   = [confs[i]   for i in keep]
            boxes   = [boxes[i]   for i in keep]
            masks_xy= [masks_xy[i] for i in keep] if masks_xy else []

            # Image-level label
            label = image_label_from_classes(classes)

            dets = []
            for i, (box, cls, cf) in enumerate(zip(boxes, classes, confs)):
                class_id = int(cls)
                # Handle unknown classes gracefully
                if class_id < len(NAMES):
                    class_name = NAMES[class_id]
                    reason = REASON_MAP.get(class_id, class_name)
                else:
                    class_name = f"Unknown_Class_{class_id}"
                    reason = f"Unknown class {class_id} detected"
                
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
                
                # Include polygon if available
                if i < len(masks_xy):
                    det_data["polygon_xy"] = [[float(x), float(y)] for x,y in masks_xy[i]]
                
                dets.append(det_data)

            rec = {
                "image": p,
                "pred_image_label": label,
                "detections": dets,
                "conf_used": args.conf if not args.use_class_thresh else "per-class",
                "iou": args.iou,
                "timestamp": datetime.utcnow().isoformat()+"Z"
            }
            f.write(json.dumps(rec) + "\n")

            img = cv2.imread(p)
            if img is not None:
                vis = draw_polygon_masks(
                    img, masks_xy, classes, confs,
                    thickness=int(args.outline_thickness),
                    halo=int(args.halo),
                    label_scale=float(args.label_scale)
                ) if len(dets)>0 else img
                outp = os.path.join(ov_dir, os.path.basename(p))
                cv2.imwrite(outp, vis)

    print("Saved JSONL:", jsonl_path)
    print("Saved overlays to:", ov_dir)
    
    # Calculate loss metrics if requested
    if args.calculate_loss:
        print("\n" + "="*50)
        print("CALCULATING LOSS METRICS ON TEST DATASET")
        print("="*50)
        
        try:
            # Use YOLO's built-in validation function for comprehensive metrics
            if os.path.exists(args.data_yaml):
                print(f"Using data config: {args.data_yaml}")
                results = model.val(data=args.data_yaml, split='test', conf=infer_conf, iou=args.iou, imgsz=args.imgsz)
                
                # Extract and display key metrics
                print("\nValidation Results:")
                print("-" * 30)
                
                # Box metrics
                if hasattr(results, 'box'):
                    box_metrics = results.box
                    print(f"Box mAP@0.5: {box_metrics.map50:.4f}")
                    print(f"Box mAP@0.5:0.95: {box_metrics.map:.4f}")
                    if hasattr(box_metrics, 'mp'):
                        print(f"Box Precision: {box_metrics.mp:.4f}")
                    if hasattr(box_metrics, 'mr'):
                        print(f"Box Recall: {box_metrics.mr:.4f}")
                
                # Mask metrics (for segmentation)
                if hasattr(results, 'seg'):
                    seg_metrics = results.seg
                    print(f"Mask mAP@0.5: {seg_metrics.map50:.4f}")
                    print(f"Mask mAP@0.5:0.95: {seg_metrics.map:.4f}")
                    if hasattr(seg_metrics, 'mp'):
                        print(f"Mask Precision: {seg_metrics.mp:.4f}")
                    if hasattr(seg_metrics, 'mr'):
                        print(f"Mask Recall: {seg_metrics.mr:.4f}")
                
                # Save detailed results
                results_path = os.path.join(args.out, "validation_results.json")
                val_results = {
                    "validation_timestamp": datetime.utcnow().isoformat()+"Z",
                    "model_weights": args.weights,
                    "data_config": args.data_yaml,
                    "confidence_threshold": infer_conf,
                    "iou_threshold": args.iou,
                    "image_size": args.imgsz
                }
                
                if hasattr(results, 'box'):
                    val_results["box_metrics"] = {
                        "mAP_50": float(results.box.map50),
                        "mAP_50_95": float(results.box.map),
                        "precision": float(results.box.mp) if hasattr(results.box, 'mp') else None,
                        "recall": float(results.box.mr) if hasattr(results.box, 'mr') else None
                    }
                
                if hasattr(results, 'seg'):
                    val_results["segmentation_metrics"] = {
                        "mAP_50": float(results.seg.map50),
                        "mAP_50_95": float(results.seg.map),
                        "precision": float(results.seg.mp) if hasattr(results.seg, 'mp') else None,
                        "recall": float(results.seg.mr) if hasattr(results.seg, 'mr') else None
                    }
                
                with open(results_path, 'w') as f:
                    json.dump(val_results, f, indent=2)
                
                print(f"\nDetailed validation results saved to: {results_path}")
                
            else:
                print(f"Warning: data.yaml file not found at {args.data_yaml}")
                print("Skipping loss calculation. Please provide correct path to data.yaml")
                
        except Exception as e:
            print(f"Error calculating loss metrics: {str(e)}")
            print("This might happen if the test dataset format doesn't match YOLO expectations.")
if __name__ == "__main__":
    main()