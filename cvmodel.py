import cv2
import logging
from ultralytics import YOLO
import time
import matplotlib.pyplot as plt

def process_video(model_path, video_path, output_path=None, conf_threshold=0.5, plot_output_path=None):
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
   
    #  Setup for plotting fish counts every 5 seconds
    plt.ion()
    fig, ax = plt.subplots()
    sampled_times_seconds = []
    sampled_fish_counts = []
    sample_interval_seconds = 5.0
    next_sample_time_seconds = sample_interval_seconds
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Video ended or frame read failed")
            break
        
        # Run inference
        results = model(frame, conf=conf_threshold)
        
        # Calculate FPS
        current_time = time.time()
        fps = 1 / (current_time - prev_time) if prev_time != 0 else 0
        prev_time = current_time
        
         # Count detections (fish) in this frame
        fish_count = 0
        if results and hasattr(results[0], "boxes") and results[0].boxes is not None:
            try:
                fish_count = len(results[0].boxes)
            except TypeError:
                # Fallback for certain ultralytics versions
                if hasattr(results[0].boxes, "xyxy") and getattr(results[0].boxes.xyxy, "shape", None):
                    fish_count = int(results[0].boxes.xyxy.shape[0])

        # Determine current video timestamp in seconds
        current_video_time_seconds = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

        # Sample and update the plot every 5 seconds of video time
        while current_video_time_seconds >= next_sample_time_seconds:
            sampled_times_seconds.append(next_sample_time_seconds)
            sampled_fish_counts.append(fish_count)

            ax.clear()
            ax.plot(sampled_times_seconds, sampled_fish_counts, marker='o')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Fish count')
            ax.set_title('Fish count every 5 seconds')
            ax.grid(True)
            plt.tight_layout()
            plt.pause(0.001)

            next_sample_time_seconds += sample_interval_seconds
        
        # Visualize results
        annotated_frame = results[0].plot()
        

        # Display runtime FPS and current fish count
        cv2.putText(annotated_frame, f"FPS: {int(fps)}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(annotated_frame, f"Fish: {fish_count}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        
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
    try:
        plt.ioff()
        # Save the plot to a file for later viewing
        if plot_output_path is None:
            plot_output_path = "fish_counts.png"
        try:
            fig.savefig(plot_output_path, dpi=200)
            print(f"Saved fish count plot to {plot_output_path}")
        except Exception as e:
            print(f"Failed to save plot: {e}")
        
        if len(sampled_times_seconds) > 0:
            plt.show()
    except Exception:
        logging.error("Failed to save plot")
    print("Video processing completed")

# convert info here
process_video(
    model_path="best.pt",  
    video_path="computer_coding_challenge_2025.mp4",
    output_path="output.mp4",  
    conf_threshold=0.6,
    plot_output_path="fish_counts.png"
)