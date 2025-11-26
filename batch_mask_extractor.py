"""
Batch Mask Extractor
Processes all videos in a folder and outputs binary masks using the specified tracking method.
"""

import os
import glob
from pathlib import Path

# Import tracking methods
import continuous_detection
import yolo_tracking
import motion_detection
import interval_tracking


def process_video_folder(input_folder, output_folder, model_path, method="continuous", **method_kwargs):
    """
    Process all videos in a folder and output binary masks.
    
    Args:
        input_folder: Path to folder containing input videos
        output_folder: Path to folder where output videos will be saved
        model_path: Path to the trained YOLO model
        method: Tracking method to use - "continuous", "yolo", "motion", or "interval"
        **method_kwargs: Additional arguments for the specific method
            - For motion: motion_threshold=5.0, fallback_interval=60
            - For interval: detection_interval=30
            - For yolo: imgsz=320, conf=0.3, vid_stride=2
    
    Returns:
        Dictionary with processing results
    """
    
    print("="*70)
    print("BATCH MASK EXTRACTION")
    print("="*70)
    print(f"Input Folder: {input_folder}")
    print(f"Output Folder: {output_folder}")
    print(f"Model: {model_path}")
    print(f"Method: {method}")
    print("="*70)
    
    # Create output folder
    os.makedirs(output_folder, exist_ok=True)
    
    # Supported video extensions
    video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.MP4', '*.AVI', '*.MOV', '*.MKV']
    
    # Find all video files
    video_files = []
    for ext in video_extensions:
        video_files.extend(glob.glob(os.path.join(input_folder, ext)))
    
    if not video_files:
        print(f"[ERROR] No video files found in {input_folder}")
        return None
    
    print(f"\nFound {len(video_files)} video(s) to process\n")
    
    # Select method
    method_map = {
        "continuous": continuous_detection.process_continuous_detection,
        "yolo": yolo_tracking.process_video_with_yolo_tracking,
        "motion": motion_detection.process_motion_detection,
        "interval": interval_tracking.process_interval_tracking
    }
    
    if method not in method_map:
        print(f"[ERROR] Invalid method: {method}")
        print(f"Available methods: {', '.join(method_map.keys())}")
        return None
    
    process_func = method_map[method]
    
    # Process each video
    results = {
        'total_videos': len(video_files),
        'processed': [],
        'failed': []
    }
    
    for idx, video_path in enumerate(video_files, 1):
        video_name = Path(video_path).stem
        video_ext = Path(video_path).suffix
        
        print(f"\n[{idx}/{len(video_files)}] Processing: {Path(video_path).name}")
        print("-"*70)
        
        try:
            # Prepare output path
            if method == "yolo":
                # YOLO creates a directory
                output_dir = os.path.join(output_folder, f"{video_name}_output")
                process_func(
                    video_path=video_path,
                    model_path=model_path,
                    output_dir=output_dir,
                    binary_mask_only=True,
                    **method_kwargs
                )
                output_location = output_dir
            else:
                # Other methods create a single video file
                output_path = os.path.join(output_folder, f"{video_name}_mask{video_ext}")
                process_func(
                    video_path=video_path,
                    model_path=model_path,
                    output_path=output_path,
                    binary_mask_only=True,
                    **method_kwargs
                )
                output_location = output_path
            
            results['processed'].append({
                'input': video_path,
                'output': output_location,
                'status': 'success'
            })
            
            print(f"[SUCCESS] Successfully processed: {Path(video_path).name}")
            
        except Exception as e:
            print(f"[FAILED] Failed to process {Path(video_path).name}: {e}")
            results['failed'].append({
                'input': video_path,
                'error': str(e)
            })
    
    # Print summary
    print("\n" + "="*70)
    print("BATCH PROCESSING SUMMARY")
    print("="*70)
    print(f"Total Videos: {results['total_videos']}")
    print(f"Successfully Processed: {len(results['processed'])}")
    print(f"Failed: {len(results['failed'])}")
    
    if results['processed']:
        print("\nProcessed Videos:")
        for item in results['processed']:
            print(f"  - {Path(item['input']).name} -> {item['output']}")
    
    if results['failed']:
        print("\nFailed Videos:")
        for item in results['failed']:
            print(f"  - {Path(item['input']).name}: {item['error']}")
    
    print("\n" + "="*70)
    print(f"Output saved to: {output_folder}")
    print("="*70)
    
    return results


def main():
    """Main entry point with default configuration."""
    
    # Default configuration
    input_folder = "./videos"
    output_folder = "./masked_output"
    model_path = "runs/segment/train/weights/best.pt"
    method = "interval"  # Options: "continuous", "yolo", "motion", "interval"
    
    # Method-specific parameters (adjust as needed)
    # method_kwargs = {
    #     "conf_threshold": 0.25
    # }
    
    # For motion method, use:
    # method_kwargs = {
    #     "motion_threshold": 5.0,
    #     "fallback_interval": 60,
    #     "conf_threshold": 0.25
    # }
    
    #For interval method, use:
    method_kwargs = {
        "detection_interval": 30,
        "conf_threshold": 0.25
    }
    
    # For yolo method, use:
    # method_kwargs = {
    #     "imgsz": 320,
    #     "conf": 0.3,
    #     "vid_stride": 2
    # }
    
    # Process videos
    results = process_video_folder(
        input_folder=input_folder,
        output_folder=output_folder,
        model_path=model_path,
        method=method,
        **method_kwargs
    )
    
    if results and results['processed']:
        print("\n[COMPLETE] Batch processing completed successfully!")
    else:
        print("\n[WARNING] Batch processing completed with issues.")


if __name__ == "__main__":
    main()
