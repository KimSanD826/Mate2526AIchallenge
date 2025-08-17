import cv2
from ultralytics import YOLO
import time

def process_video(model_path, video_path, output_path=None, conf_threshold=0.5):
    # Load model
    model = YOLO(model_path)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), "Error opening video file"
    
    # Get video properties for output if needed
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Video writer if output path is specified
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # For FPS calculation
    prev_time = 0
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Video ended or frame read failed")
            break
        
        # Run inference
        results = model(frame, conf=conf_threshold)
        
        # Calculate FPS
        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time
        
        # Visualize results
        annotated_frame = results[0].plot()
        
        # Display FPS
        cv2.putText(annotated_frame, f"FPS: {int(fps)}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Write to output file if specified
        if output_path:
            out.write(annotated_frame)
        
        # Display
        cv2.imshow("Fish Classification", annotated_frame)
        
        # Break the loop if 'q' is pressed or window is closed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        if cv2.getWindowProperty("Fish Classification", cv2.WND_PROP_VISIBLE) < 1:
            break
    
    # Clean up
    cap.release()
    if output_path:
        out.release()
    cv2.destroyAllWindows()
    print("Video processing completed")

# convert info here
process_video(
    model_path="yolov8n.pt",  
    video_path="computer_coding_challenge_2025.mp4",
    output_path="output.mp4",  
    conf_threshold=0.6
)