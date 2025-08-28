# main.py

import cv2
from ultralytics import YOLO
import numpy as np
import os
from collections import deque

# ========== COLOR CONFIGURATION SECTION ==========
# Modify these values to match your specific video/match

TEAM_COLORS = {
    'team1': {
        'name': 'Newcastle (B/W)',
        'detection_method': 'stripe',  # 'stripe' for black/white stripes, 'solid' for solid colors
        'primary_color': [200, 200, 200],  # White (BGR format)
        'secondary_color': [50, 50, 50],   # Black (BGR format)
        'min_brightness': 80,  # Minimum average brightness for detection
        'stripe_variance': 800,  # High variance indicates stripes
        'box_color': (255, 0, 0)  # Blue box color for display
    },
    'team2': {
        'name': 'Opposition',
        'detection_method': 'solid',
        'primary_color': [0, 0, 255],     # Red (BGR format)
        'secondary_color': None,
        'min_brightness': 100,
        'stripe_variance': None,
        'box_color': (0, 0, 255)  # Red box color for display
    }
}

REFEREE_CONFIG = {
    'shirt_color': [0, 255, 255],  # Yellow shirt (BGR format)
    'pants_color': [0, 0, 0],      # Black pants (BGR format)
    'tolerance': 50,               # Color matching tolerance
    'max_count': 3,                # Maximum referees on field
    'box_color': (0, 255, 255)     # Yellow box color for display
}

# ================================================

class RefereeValidator:
    def __init__(self, config):
        self.config = config
        self.referee_history = deque(maxlen=10)
        
    def color_distance(self, color1, color2):
        """Calculate Euclidean distance between two colors"""
        return np.sqrt(sum((c1 - c2) ** 2 for c1, c2 in zip(color1, color2)))
        
    def is_referee_colors(self, person_crop):
        """Check if person has referee colors (yellow shirt, black pants)"""
        h, w, _ = person_crop.shape
        if h < 20 or w < 10:
            return False
            
        # Split into upper (shirt) and lower (pants) regions
        upper_region = person_crop[0:h//2, :]  # Top half - shirt
        lower_region = person_crop[h//2:h, :]  # Bottom half - pants
        
        if upper_region.size == 0 or lower_region.size == 0:
            return False
            
        # Get average colors
        upper_avg = np.mean(upper_region, axis=(0, 1))
        lower_avg = np.mean(lower_region, axis=(0, 1))
        
        # Check if upper matches yellow shirt
        shirt_match = self.color_distance(upper_avg, self.config['shirt_color']) < self.config['tolerance']
        
        # Check if lower matches black pants
        pants_match = self.color_distance(lower_avg, self.config['pants_color']) < self.config['tolerance']
        
        print(f"    Referee color check: shirt_avg={upper_avg}, pants_avg={lower_avg}")
        print(f"    Shirt match: {shirt_match}, Pants match: {pants_match}")
        
        return shirt_match and pants_match
        
    def is_valid_referee(self, bbox, frame, existing_referees):
        """Validate referee with configurable colors"""
        x1, y1, x2, y2 = bbox
        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
        
        # 1. Position check - not in sky
        frame_height = frame.shape[0]
        if y2 < frame_height * 0.3:
            print(f"    Rejected referee: too high in frame")
            return False
        
        # 2. Count limit
        if len(existing_referees) >= self.config['max_count']:
            print(f"    Rejected referee: too many referees already")
            return False
        
        # 3. Color check
        person_crop = frame[y1:y2, x1:x2]
        if not self.is_referee_colors(person_crop):
            print(f"    Rejected referee: wrong colors")
            return False
        
        # 4. Distance check
        for ref_pos in existing_referees:
            ref_x, ref_y = ref_pos
            distance = np.sqrt((center_x - ref_x)**2 + (center_y - ref_y)**2)
            if distance < 50:
                print(f"    Rejected referee: too close to existing referee")
                return False
        
        return True

class TeamClassifier:
    def __init__(self, team_colors):
        self.team_colors = team_colors
        
    def color_distance(self, color1, color2):
        """Calculate color distance"""
        return np.sqrt(sum((c1 - c2) ** 2 for c1, c2 in zip(color1, color2)))
    
    def detect_stripes(self, crop):
        """Detect if jersey has stripes (high variance in brightness)"""
        if crop.size == 0:
            return False, 0
            
        # Convert to grayscale and calculate variance across rows
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        
        # Calculate variance across horizontal strips
        h = gray.shape[0]
        if h < 10:
            return False, 0
            
        # Sample horizontal strips and calculate brightness variance
        strip_means = []
        num_strips = min(10, h // 2)
        for i in range(num_strips):
            start_row = i * h // num_strips
            end_row = (i + 1) * h // num_strips
            strip_mean = np.mean(gray[start_row:end_row, :])
            strip_means.append(strip_mean)
        
        variance = np.var(strip_means)
        has_stripes = variance > 800  # High variance indicates stripes
        
        return has_stripes, variance
    
    def classify_player_team(self, player_crop):
        """Classify player into team based on configurable colors"""
        if player_crop.size == 0:
            return 'player', (0, 255, 0)
            
        h, w, _ = player_crop.shape
        if h < 10 or w < 5:
            return 'player', (0, 255, 0)
            
        # Focus on torso area (jersey)
        torso_crop = player_crop[h//4:3*h//4, w//4:3*w//4]
        if torso_crop.size == 0:
            return 'player', (0, 255, 0)
            
        avg_color = np.mean(torso_crop, axis=(0, 1))
        avg_brightness = np.mean(avg_color)
        
        print(f"    Player analysis: avg_color={avg_color}, brightness={avg_brightness:.1f}")
        
        # Check each team
        for team_id, config in self.team_colors.items():
            if config['detection_method'] == 'stripe':
                # Check for stripe pattern
                has_stripes, variance = self.detect_stripes(torso_crop)
                brightness_ok = avg_brightness > config['min_brightness']
                
                print(f"    {config['name']}: stripes={has_stripes}, variance={variance:.1f}, brightness_ok={brightness_ok}")
                
                if has_stripes and brightness_ok and variance > config['stripe_variance']:
                    return config['name'], config['box_color']
                    
            elif config['detection_method'] == 'solid':
                # Check for solid color match
                color_dist = self.color_distance(avg_color, config['primary_color'])
                brightness_ok = avg_brightness > config['min_brightness']
                
                print(f"    {config['name']}: color_dist={color_dist:.1f}, brightness_ok={brightness_ok}")
                
                if color_dist < 80 and brightness_ok:  # Adjust threshold as needed
                    return config['name'], config['box_color']
        
        # Default to generic player
        return 'player', (0, 255, 0)

class BallTrajectoryDetector:
    def __init__(self, max_trajectory_length=20):
        self.ball_positions = deque(maxlen=max_trajectory_length)
        self.previous_frame_gray = None
        self.ball_velocity = None
        self.last_yolo_position = None
        self.search_radius = 100
        self.debug_mode = False
        
    def frame_to_center_coords(self, x, y, frame_shape):
        """Convert frame coordinates to center-based coordinates"""
        h, w = frame_shape[:2]
        center_x = w // 2
        center_y = h // 2
        
        rel_x = x - center_x
        rel_y = y - center_y
        
        return rel_x, rel_y
        
    def get_search_area(self, frame_shape):
        """Define search area - exclude 25% top border, 15% other borders"""
        h, w = frame_shape[:2]
        
        border_top = int(h * 0.25)     # 25% exclusion from top
        border_bottom = int(h * 0.15)  # 15% exclusion from bottom  
        border_left = int(w * 0.15)    # 15% exclusion from left
        border_right = int(w * 0.15)   # 15% exclusion from right
        
        default_area = (border_left, border_top, w - border_right, h - border_bottom)
        
        if self.last_yolo_position:
            x, y = self.last_yolo_position
            
            search_x1 = max(border_left, x - self.search_radius)
            search_y1 = max(border_top, y - self.search_radius)
            search_x2 = min(w - border_right, x + self.search_radius)
            search_y2 = min(h - border_bottom, y + self.search_radius)
            
            return (search_x1, search_y1, search_x2, search_y2)
        
        return default_area
    
    def is_likely_field_marking_or_artifact(self, contour, x, y, w, h, area):
        """Filter out field markings, artifacts, and non-ball objects"""
        if area == 0:
            return True
        
        perimeter = cv2.arcLength(contour, True)
        aspect_ratio = w / h if h > 0 else 10
        compactness = (perimeter * perimeter) / (4 * np.pi * area) if area > 0 else 1000
        
        if aspect_ratio > 6 or aspect_ratio < 0.16:
            print(f"    Rejected: aspect ratio {aspect_ratio:.2f}")
            return True
            
        if compactness > 3.0:
            print(f"    Rejected: compactness {compactness:.2f}")
            return True
            
        if area > 1200 or area < 20:
            print(f"    Rejected: area {area}")
            return True
        
        if w > 60 or h > 60 or w < 4 or h < 4:
            print(f"    Rejected: size {w}x{h}")
            return True
            
        return False
    
    def detect_ball_motion_filtered(self, frame, frame_number=0):
        """Ball detection with comprehensive filtering"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if self.previous_frame_gray is not None:
            search_x1, search_y1, search_x2, search_y2 = self.get_search_area(frame.shape)
            
            search_gray = gray[search_y1:search_y2, search_x1:search_x2]
            search_prev = self.previous_frame_gray[search_y1:search_y2, search_x1:search_x2]
            
            if search_gray.size == 0 or search_prev.size == 0:
                self.previous_frame_gray = gray
                return []
            
            diff = cv2.absdiff(search_prev, search_gray)
            _, motion_mask = cv2.threshold(diff, 40, 255, cv2.THRESH_BINARY)
            _, white_mask = cv2.threshold(search_gray, 200, 255, cv2.THRESH_BINARY)
            ball_mask = cv2.bitwise_and(motion_mask, white_mask)
            
            if self.debug_mode and frame_number % 25 == 0:
                os.makedirs('debug', exist_ok=True)
                cv2.imwrite(f'debug/motion_mask_{frame_number}.jpg', motion_mask)
                cv2.imwrite(f'debug/white_mask_{frame_number}.jpg', white_mask)
                cv2.imwrite(f'debug/combined_mask_{frame_number}.jpg', ball_mask)
            
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            ball_mask = cv2.morphologyEx(ball_mask, cv2.MORPH_OPEN, kernel)
            ball_mask = cv2.morphologyEx(ball_mask, cv2.MORPH_CLOSE, kernel)
            
            contours, _ = cv2.findContours(ball_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            ball_candidates = []
            print(f"    Found {len(contours)} contours in motion detection")
            
            for i, contour in enumerate(contours):
                area = cv2.contourArea(contour)
                
                if area > 15:
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    if self.is_likely_field_marking_or_artifact(contour, x, y, w, h, area):
                        continue
                    
                    center_x = search_x1 + x + w//2
                    center_y = search_y1 + y + h//2
                    rel_x, rel_y = self.frame_to_center_coords(center_x, center_y, frame.shape)
                    
                    contour_area = cv2.contourArea(contour)
                    bounding_area = w * h
                    fill_ratio = contour_area / bounding_area if bounding_area > 0 else 0
                    
                    if fill_ratio > 0.3:
                        ball_candidates.append((center_x, center_y, area, fill_ratio, rel_x, rel_y))
                        print(f"    Accepted: frame=({center_x}, {center_y}), center=({rel_x:+d},{rel_y:+d})")
            
            self.previous_frame_gray = gray
            return ball_candidates
        
        self.previous_frame_gray = gray
        return []
    
    def update_trajectory(self, ball_position, is_yolo=False):
        """Updates ball trajectory with validation"""
        if ball_position:
            if len(self.ball_positions) > 0:
                last_pos = self.ball_positions[-1]
                distance = np.sqrt((ball_position[0] - last_pos[0])**2 + (ball_position[1] - last_pos[1])**2)
                
                max_distance = 300 if is_yolo else 150
                
                if distance > max_distance:
                    print(f"  Warning: Rejecting trajectory jump of {distance:.1f} pixels")
                    return
            
            self.ball_positions.append(ball_position)
            
            if is_yolo:
                self.last_yolo_position = ball_position
            
            if len(self.ball_positions) >= 2:
                prev_pos = self.ball_positions[-2]
                curr_pos = self.ball_positions[-1]
                self.ball_velocity = (curr_pos[0] - prev_pos[0], curr_pos[1] - prev_pos[1])
    
    def draw_trajectory(self, frame):
        """Draw trajectory with coordinate system"""
        # Draw exclusion zones
        h, w = frame.shape[:2]
        
        cv2.rectangle(frame, (0, 0), (w, int(h * 0.25)), (0, 0, 200), 2)
        cv2.putText(frame, "EXCLUDED TOP (25%)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 200), 2)
        
        cv2.rectangle(frame, (0, 0), (int(w * 0.15), h), (0, 100, 200), 1)
        cv2.rectangle(frame, (w - int(w * 0.15), 0), (w, h), (0, 100, 200), 1)
        cv2.rectangle(frame, (0, h - int(h * 0.15)), (w, h), (0, 100, 200), 1)
        
        # Coordinate system
        center_x, center_y = w // 2, h // 2
        cv2.circle(frame, (center_x, center_y), 5, (255, 255, 255), -1)
        cv2.putText(frame, "(0,0)", (center_x + 10, center_y - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.line(frame, (center_x - 50, center_y), (center_x + 50, center_y), (255, 255, 255), 1)
        cv2.line(frame, (center_x, center_y - 50), (center_x, center_y + 50), (255, 255, 255), 1)
        
        if len(self.ball_positions) < 2:
            return frame
        
        # Search area
        if self.last_yolo_position:
            search_x1, search_y1, search_x2, search_y2 = self.get_search_area(frame.shape)
            cv2.rectangle(frame, (search_x1, search_y1), (search_x2, search_y2), (100, 200, 100), 2)
        
        # Trajectory
        points = list(self.ball_positions)
        for i in range(1, len(points)):
            cv2.line(frame, points[i-1][:2], points[i][:2], (255, 255, 0), 4)
        
        for i, point in enumerate(points):
            x, y = point[:2]
            rel_x, rel_y = self.frame_to_center_coords(x, y, frame.shape)
            
            radius = max(3, int(5 * (i + 1) / len(points)))
            cv2.circle(frame, (x, y), radius, (255, 255, 0), -1)
            
            if i == len(points) - 1:
                cv2.putText(frame, f"({rel_x:+d},{rel_y:+d})", (x + 15, y - 15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Prediction arrow
        if len(points) >= 2 and self.ball_velocity:
            last_pos = points[-1][:2]
            velocity_magnitude = np.sqrt(self.ball_velocity[0]**2 + self.ball_velocity[1]**2)
            
            if velocity_magnitude > 5:
                arrow_length = min(60, velocity_magnitude * 2)
                norm_vel_x = self.ball_velocity[0] / velocity_magnitude * arrow_length
                norm_vel_y = self.ball_velocity[1] / velocity_magnitude * arrow_length
                
                arrow_end = (
                    int(last_pos[0] + norm_vel_x),
                    int(last_pos[1] + norm_vel_y)
                )
                
                cv2.arrowedLine(frame, last_pos, arrow_end, (255, 0, 255), 4, tipLength=0.3)
        
        return frame

def post_process_detections(detections, frame, referee_validator, team_classifier):
    """Post-process YOLO detections with configurable team and referee classification"""
    processed_detections = []
    players = []
    
    # Separate person detections from others
    for detection in detections:
        label, bbox, conf = detection
        if label in ['player', 'goalkeeper', 'referee']:
            players.append(detection)
        else:
            processed_detections.append(detection)
    
    # Process person detections
    validated_referees = []
    
    for detection in players:
        label, bbox, conf = detection
        x1, y1, x2, y2 = bbox
        
        # Extract person crop for analysis
        person_crop = frame[y1:y2, x1:x2]
        
        # Check if it's a valid referee first
        if referee_validator.is_valid_referee(bbox, frame, validated_referees):
            validated_referees.append(((x1 + x2) // 2, (y1 + y2) // 2))
            processed_detections.append(('referee', bbox, conf))
            print(f"  Validated referee at ({(x1+x2)//2}, {(y1+y2)//2})")
        else:
            # Classify as player team
            team_name, box_color = team_classifier.classify_player_team(person_crop)
            processed_detections.append((team_name, bbox, conf))
            print(f"  Classified as: {team_name}")
    
    return processed_detections

def detect_and_draw_boxes(frame, model, ball_detector, frame_number):
    """Enhanced detection with configurable team and referee colors"""
    results = model(frame, verbose=False, conf=0.25)
    annotated_frame = frame.copy()
    
    # Initialize classifiers with configurations
    referee_validator = RefereeValidator(REFEREE_CONFIG)
    team_classifier = TeamClassifier(TEAM_COLORS)

    yolo_ball_detected = False
    yolo_ball_position = None

    print(f"Frame {frame_number}: Processing with configured colors...")
    print(f"  Teams: {[config['name'] for config in TEAM_COLORS.values()]}")
    print(f"  Referee: Yellow shirt + Black pants")

    # Collect raw detections
    raw_detections = []
    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls = int(box.cls[0])
            label = model.names[cls]

            if label not in ['player', 'goalkeeper', 'referee', 'ball']:
                continue

            x1, y1, x2, y2 = box.xyxy[0].int().tolist()
            conf = float(box.conf[0])
            
            raw_detections.append((label, (x1, y1, x2, y2), conf))

    # Post-process detections
    processed_detections = post_process_detections(raw_detections, frame, referee_validator, team_classifier)

    # Draw processed detections
    for label, bbox, conf in processed_detections:
        x1, y1, x2, y2 = bbox
        
        # Get box color based on team/referee configuration
        if 'Newcastle' in label:
            box_color = TEAM_COLORS['team1']['box_color']
        elif 'Opposition' in label:
            box_color = TEAM_COLORS['team2']['box_color']
        elif label == 'referee':
            box_color = REFEREE_CONFIG['box_color']
        elif label == 'ball':
            box_color = (0, 255, 255)  # Cyan
            yolo_ball_detected = True
            yolo_ball_position = ((x1 + x2) // 2, (y1 + y2) // 2)
            
            rel_x, rel_y = ball_detector.frame_to_center_coords(
                yolo_ball_position[0], yolo_ball_position[1], frame.shape
            )
            print(f"  YOLO ball: frame=({yolo_ball_position[0]}, {yolo_ball_position[1]}) center=({rel_x:+d},{rel_y:+d})")
        else:
            box_color = (0, 255, 0)  # Default green

        # Draw detection
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), box_color, 2)
        cv2.putText(annotated_frame, f'{label} {conf:.2f}', (x1, y1 - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

    # Ball trajectory handling (unchanged)
    if not yolo_ball_detected:
        print(f"  YOLO missed ball, trying motion detection...")
        motion_candidates = ball_detector.detect_ball_motion_filtered(frame, frame_number)
        
        if motion_candidates:
            best_candidate = max(motion_candidates, key=lambda x: x[2] * x[3])
            center_x, center_y, area, fill_ratio, rel_x, rel_y = best_candidate
            
            print(f"  Motion ball: frame=({center_x}, {center_y}) center=({rel_x:+d},{rel_y:+d})")
            
            cv2.circle(annotated_frame, (center_x, center_y), 12, (255, 0, 255), 3)
            cv2.putText(annotated_frame, f'ball (motion)', 
                       (center_x - 60, center_y - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
            cv2.putText(annotated_frame, f'({rel_x:+d},{rel_y:+d})', 
                       (center_x - 40, center_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            ball_detector.update_trajectory((center_x, center_y), is_yolo=False)
        else:
            ball_detector.detect_ball_motion_filtered(frame, frame_number)
    else:
        rel_x, rel_y = ball_detector.frame_to_center_coords(
            yolo_ball_position[0], yolo_ball_position[1], frame.shape
        )
        cv2.putText(annotated_frame, f'({rel_x:+d},{rel_y:+d})', 
                   (yolo_ball_position[0] + 15, yolo_ball_position[1] + 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        ball_detector.update_trajectory(yolo_ball_position, is_yolo=True)
        ball_detector.detect_ball_motion_filtered(frame, frame_number)

    annotated_frame = ball_detector.draw_trajectory(annotated_frame)
    return annotated_frame

def process_video(video_path, model, num_frames_to_process, output_dir="output_frames"):
    """Process video with configurable team and referee colors"""
    os.makedirs(output_dir, exist_ok=True)
    
    ball_detector = BallTrajectoryDetector(max_trajectory_length=15)
    ball_detector.debug_mode = True

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print("=== COLOR CONFIGURATION ===")
    for team_id, config in TEAM_COLORS.items():
        print(f"{team_id}: {config['name']} - {config['detection_method']}")
    print(f"Referee: {REFEREE_CONFIG['shirt_color']} shirt + {REFEREE_CONFIG['pants_color']} pants")
    print(f"Video: {frame_width}x{frame_height}")
    print("===========================\n")

    frame_count = 0
    processed_count = 0
    
    while frame_count < num_frames_to_process:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % 5 == 0:
            print(f"\n--- Processing Frame {frame_count} ---")
            processed_frame = detect_and_draw_boxes(frame, model, ball_detector, frame_count)

            output_path = os.path.join(output_dir, f"frame_{frame_count:04d}.jpg")
            cv2.imwrite(output_path, processed_frame)
            print(f"  Info: Saved {output_path}")
            processed_count += 1

        frame_count += 1

    cap.release()
    print(f"\nFinished processing {processed_count} frames.")

if __name__ == "__main__":
    model = YOLO('best.pt')
    video_file = 'extract_45s-55s.mp4'
    num_frames = 500
    
    process_video(video_file, model, num_frames)