import cv2
import numpy as np

# Function to detect traffic lights using HSV color space and contour analysis
def detect_traffic_lights(frame, hue_ranges, sat_min=100, val_min=100, min_area=50, max_area=5000):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lights = []
    
    # Process each hue range (e.g., red or green)
    for hue_range in hue_ranges:
        # Create mask for the hue range
        mask = cv2.inRange(hsv, (hue_range[0], sat_min, val_min), (hue_range[1], 255, 255))
        # Apply morphological operations to reduce noise
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if min_area < area < max_area:
                # Get bounding box
                x, y, w, h = cv2.boundingRect(contour)
                # Ensure aspect ratio is reasonable for traffic lights (roughly circular/square)
                aspect_ratio = w / h if h > 0 else 1
                if 0.5 < aspect_ratio < 2.0:
                    # Calculate average intensity in the region
                    region = hsv[y:y+h, x:x+w]
                    mask_region = mask[y:y+h, x:x+w]
                    intensity = np.sum(mask_region) / (255 * w * h) if w * h > 0 else 0
                    if intensity > 0.2:  # Threshold for strong detection
                        lights.append({
                            'bbox': (x, y, w, h),
                            'intensity': intensity,
                            'color': hue_range[2]  # Label as 'red' or 'green'
                        })
    
    return lights

# Function to track and refine detected lights
def track_lights(prev_lights, new_lights, max_distance=50):
    if not prev_lights:
        return new_lights, new_lights
    
    tracked_lights = []
    used_new = set()
    
    for prev in prev_lights:
        prev_x, prev_y, prev_w, prev_h = prev['bbox']
        prev_center = (prev_x + prev_w // 2, prev_y + prev_h // 2)
        min_dist = float('inf')
        best_match = None
        
        for i, new_light in enumerate(new_lights):
            if i in used_new:
                continue
            new_x, new_y, new_w, new_h = new_light['bbox']
            new_center = (new_x + new_w // 2, new_y + new_h // 2)
            dist = np.sqrt((prev_center[0] - new_center[0])**2 + (prev_center[1] - new_center[1])**2)
            
            if dist < min_dist and dist < max_distance and prev['color'] == new_light['color']:
                min_dist = dist
                best_match = i
        
        if best_match is not None:
            tracked_lights.append(new_lights[best_match])
            used_new.add(best_match)
    
    # Add new lights that weren't matched
    for i, new_light in enumerate(new_lights):
        if i not in used_new:
            tracked_lights.append(new_light)
    
    return tracked_lights, new_lights

# Initialize USB camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open USB camera.")
    exit()

# Set camera resolution (optimized for Raspberry Pi)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Define hue ranges for red and green in HSV
red_hue_ranges = [(0, 10, 'red'), (170, 180, 'red')]  # Red split due to hue wrap-around
green_hue_range = [(35, 85, 'green')]  # Green
all_hue_ranges = red_hue_ranges + green_hue_range

# Initialize variables for tracking
prev_lights = []
frame_count = 0
skip_frames = 2  # Process every 2nd frame for performance

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break
        
        frame_count += 1
        if frame_count % skip_frames == 0:
            # Detect traffic lights
            detected_lights = detect_traffic_lights(frame, all_hue_ranges)
            # Track lights across frames
            prev_lights, current_lights = track_lights(prev_lights, detected_lights)
        else:
            current_lights = prev_lights  # Use previous detections for skipped frames

        # Draw detected lights and determine dominant light
        max_intensity = 0
        dominant_light = None
        for light in current_lights:
            x, y, w, h = light['bbox']
            color = (0, 0, 255) if light['color'] == 'red' else (0, 255, 0)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            if light['intensity'] > max_intensity:
                max_intensity = light['intensity']
                dominant_light = light

        # Display status
        if dominant_light and max_intensity > 0.3:
            text = f"{dominant_light['color'].capitalize()} Light"
            color = (0, 0, 255) if dominant_light['color'] == 'red' else (0, 255, 0)
        else:
            text = "No Light Detected"
            color = (255, 255, 255)
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        # Show frame
        cv2.imshow('Advanced Traffic Light Detection', frame)

        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Program terminated by user.")

finally:
    cap.release()
    cv2.destroyAllWindows()