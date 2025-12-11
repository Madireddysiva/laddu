
import cv2
import numpy as np
from ultralytics import YOLO
import torch
from pathlib import Path
import time

class ObjectDetector:
    def __init__(self, model_path=None):
        """
        Initialize the object detector with YOLO model
        Args:
            model_path: Path to custom trained model (optional)
        """
        # Load YOLO model
        if model_path and Path(model_path).exists():
            self.model = YOLO(model_path)
        else:
            # Use default YOLOv8n model
            self.model = YOLO('yolov8n.pt')
        
        # Set device (GPU if available)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")

    def process_video(self, video_path, output_path=None, conf_threshold=0.25):
        """
        Process video file and perform object detection
        Args:
            video_path: Path to input video
            output_path: Path to save output video (optional)
            conf_threshold: Confidence threshold for detections
        """
        # Open video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Initialize video writer if output path is provided
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_count = 0
        start_time = time.time()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Perform detection
            results = self.model(frame, conf=conf_threshold)[0]
            
            # Process detections
            for result in results.boxes.data.tolist():
                x1, y1, x2, y2, confidence, class_id = result
                
                # Convert coordinates to integers
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                
                # Get class name
                class_name = results.names[int(class_id)]
                
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Add label
                label = f"{class_name}: {confidence:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Write frame if output path is provided
            if writer:
                writer.write(frame)

            # Update progress
            frame_count += 1
            if frame_count % 30 == 0:  # Update every 30 frames
                elapsed_time = time.time() - start_time
                fps_processing = frame_count / elapsed_time
                print(f"Processed {frame_count}/{total_frames} frames "
                      f"({(frame_count/total_frames)*100:.1f}%) "
                      f"at {fps_processing:.1f} FPS")

        # Release resources
        cap.release()
        if writer:
            writer.release()

        print(f"Processing completed. Total frames: {frame_count}")

def main():
    # Initialize detector
    detector = ObjectDetector()
    
    # Process video
    input_video = "vidf.mp4"
    output_video = "output_video.mp4"
    
    try:
        detector.process_video(input_video, output_video)
        print(f"Output video saved to: {output_video}")
    except Exception as e:
        print(f"Error processing video: {str(e)}")

if __name__ == "__main__":
    main() 