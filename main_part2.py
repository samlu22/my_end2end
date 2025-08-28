# main_part2_final.py

import cv2
import math
from ultralytics import YOLO
import numpy as np

# --- 1. CONFIGURATION ---
VIDEO_PATH = 'extract_45s-55s.mp4'
FRAME_RATE = 25
FRAME_START_INDEX = 154
NUM_FRAMES_TO_PROCESS = 100
BALL_CONF_THRESHOLD = 0.5

# --- 2. LOAD CUSTOM YOLOv8 MODEL FOR SOCCER ---
try:
    yolo_model = YOLO('best.pt')
except Exception as e:
    print(f"Error: Could not load the YOLO model. Make sure 'best.pt' is in your directory.")
    print(f"Details: {e}")
    exit()

def find_goal_line(frame):
    """
    Finds a horizontal goal line using the Hough Line Transform.
    Returns the average y-coordinate of detected horizontal lines.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=100, maxLineGap=10)
    
    horizontal_lines_y = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # Calculate the angle of the line
            if x2 - x1 != 0:
                angle_rad = np.arctan2(y2 - y1, x2 - x1)
                angle_deg = np.degrees(angle_rad)
                
                if abs(angle_deg) < 10 or abs(angle_deg) > 170:
                    horizontal_lines_y.append((y1 + y2) / 2)
                    cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    if horizontal_lines_y:
        return np.mean(horizontal_lines_y), frame
    
    return None, frame

def calculate_reaction_and_speed(video_path, start_frame_index, num_frames_to_process):
    """
    Analyzes a video segment to track the ball, estimate speed, and reaction time.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_index)
    
    strike_frame_index = -1
    goal_reached_frame_index = -1
    ball_start_pos = None
    goal_line_y = None
    
    print("Beginning video analysis...")

    # First, find the goal line in the first few frames
    ret, frame = cap.read()
    if ret:
        goal_line_y, visualized_frame = find_goal_line(frame.copy())
        if goal_line_y is None:
            print("Error: Could not detect a horizontal goal line. Cannot proceed with calculations.")
            return
        print(f"Debug: Goal line found at y-coordinate: {goal_line_y:.2f}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_index) # Reset to start

    for i in range(num_frames_to_process):
        ret, frame = cap.read()
        if not ret:
            break
        
        current_frame_index = start_frame_index + i
        
        results = yolo_model(frame, verbose=False, conf=BALL_CONF_THRESHOLD)
        
        for r in results:
            for box in r.boxes:
                label = yolo_model.names[int(box.cls[0])]
                
                # --- Ball Tracking Logic ---
                if label == 'ball':
                    ball_x, ball_y = int((box.xyxy[0][0] + box.xyxy[0][2]) / 2), int((box.xyxy[0][1] + box.xyxy[0][3]) / 2)

                    # Only record the strike point the first time a ball is detected
                    if strike_frame_index == -1:
                        strike_frame_index = current_frame_index
                        ball_start_pos = (ball_x, ball_y)
                        print(f"Info: Ball strike detected at frame {strike_frame_index} at position (x={ball_start_pos[0]}, y={ball_start_pos[1]}).")
                    
                    # Track goal-reaching only after the strike point is set
                    if strike_frame_index != -1 and ball_y > goal_line_y:
                        if goal_reached_frame_index == -1:
                            goal_reached_frame_index = current_frame_index
                            print(f"Info: Ball crossed the goal line at frame {goal_reached_frame_index}.")
                    
                    # Optional: visualize ball and lines
                    cv2.circle(frame, (ball_x, ball_y), 5, (0, 0, 255), -1)

    cap.release()

    # --- 3. FINAL CALCULATIONS ---
    if strike_frame_index != -1 and goal_reached_frame_index != -1 and strike_frame_index < goal_reached_frame_index:
        time_s = (goal_reached_frame_index - strike_frame_index) / FRAME_RATE
        
        # This is a conceptual approximation without a homography matrix.
        # Let's use a more realistic pixel distance.
        if ball_start_pos:
            pixel_distance = abs(goal_line_y - ball_start_pos[1])
        else:
            pixel_distance = 0
            
        # Conversion factor (conceptual): 10 meters on the pitch might be 200 pixels
        PIXELS_PER_METER = 20  
        distance_m = pixel_distance / PIXELS_PER_METER
        
        if time_s > 0 and distance_m > 0:
            speed_mps = distance_m / time_s
            speed_mph = speed_mps * 2.23694 # m/s to mph
            
            print("\n--- Final Analysis ---")
            print(f"Time from strike to goal line: {time_s:.2f} seconds")
            print(f"Estimated Ball Speed: {speed_mph:.2f} MPH")
        else:
            print("Error: Calculations resulted in zero or negative values.")
    else:
        print("\nAnalysis failed: Could not reliably detect strike point or goal crossing.")
        print(f"Strike Frame: {strike_frame_index}, Goal Frame: {goal_reached_frame_index}")

if __name__ == "__main__":
    calculate_reaction_and_speed(VIDEO_PATH, FRAME_START_INDEX, NUM_FRAMES_TO_PROCESS)