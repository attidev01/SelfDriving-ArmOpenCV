#!/usr/bin/env python3
"""
Complete Advanced Vision System
Integrates lane detection, advanced traffic light detection, speed limit sign detection,
and partial stop sign detection with improved tracking and detection algorithms.
"""

import cv2
import numpy as np
import time
import os
from collections import deque

class CompleteVisionSystem:
    def __init__(self, camera_id=0, resolution=(640, 480)):
        """Initialize the complete vision system"""
        self.camera = cv2.VideoCapture(camera_id)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
        
        # Performance tracking
        self.fps = 0
        self.frame_count = 0
        self.start_time = time.time()
        
        # Lane detection parameters
        self.lane_lower_white = np.array([0, 0, 180])
        self.lane_upper_white = np.array([180, 40, 255])
        
        # Traffic light detection parameters
        self.red_hue_ranges = [(0, 10, 'red'), (170, 180, 'red')]  # Red split due to hue wrap-around
        self.green_hue_range = [(35, 85, 'green')]  # Green
        self.all_hue_ranges = self.red_hue_ranges + self.green_hue_range
        self.prev_lights = []
        self.skip_frames = 2  # Process every 2nd frame for performance
        
        # Speed limit sign detection parameters
        self.speed_red_lower = np.array([0, 100, 100])
        self.speed_red_upper = np.array([10, 255, 255])
        self.speed_red_lower2 = np.array([170, 100, 100])
        self.speed_red_upper2 = np.array([180, 255, 255])
        self.speed_white_lower = np.array([0, 0, 200])
        self.speed_white_upper = np.array([180, 30, 255])
        self.speed_black_lower = np.array([0, 0, 0])
        self.speed_black_upper = np.array([180, 255, 70])
        
        # Stop sign detection parameters
        self.stop_red_lower1 = np.array([0, 70, 50])
        self.stop_red_upper1 = np.array([10, 255, 255])
        self.stop_red_lower2 = np.array([170, 70, 50])
        self.stop_red_upper2 = np.array([180, 255, 255])
        
        # Detection history for temporal filtering
        self.history_length = 5
        self.traffic_light_history = []
        self.traffic_light_status = "UNKNOWN"
        self.traffic_light_confidence = 0
        
        self.speed_limit_history = []
        self.speed_limit_detected = False
        self.speed_limit_confidence = 0
        
        self.stop_detection_history = []
        self.stop_sign_detected = False
        self.stop_detection_confidence = 0
        
        # Lane detection history for temporal filtering
        self.lane_direction_history = []
        self.current_direction = "UNKNOWN"
        
        # Display options
        self.show_masks = True
        
        print("Complete Vision System Initialized")
        print(f"Camera resolution: {resolution}")
    
    def get_binary_mask(self, img):
        """Returns a binary mask of white-ish road area with improved stability"""
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lane_lower_white, self.lane_upper_white)
        
        # Apply morphological operations to reduce noise and improve stability
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # Close gaps
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)   # Remove noise
        
        return mask
    
    def find_lane_contours(self, mask):
        """Finds contours in binary mask and returns the largest contour"""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        return max(contours, key=cv2.contourArea)
    
    def detect_traffic_lights(self, frame, hue_ranges, sat_min=100, val_min=100, min_area=50, max_area=5000):
        """Detect traffic lights using HSV color space and contour analysis"""
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
    
    def track_lights(self, prev_lights, new_lights, max_distance=50):
        """Track and refine detected lights across frames"""
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
    
    def detect_speed_limit(self, frame):
        """Detect speed limit signs in the frame"""
        output = frame.copy()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Red mask (for the outer circle)
        mask1 = cv2.inRange(hsv, self.speed_red_lower, self.speed_red_upper)
        mask2 = cv2.inRange(hsv, self.speed_red_lower2, self.speed_red_upper2)
        red_mask = cv2.bitwise_or(mask1, mask2)
        
        # White mask (for the inner circle)
        white_mask = cv2.inRange(hsv, self.speed_white_lower, self.speed_white_upper)
        
        # Black mask (for the text/numbers)
        black_mask = cv2.inRange(hsv, self.speed_black_lower, self.speed_black_upper)
        
        # Apply morphological operations to clean up masks
        kernel = np.ones((5, 5), np.uint8)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)
        black_mask = cv2.morphologyEx(black_mask, cv2.MORPH_OPEN, kernel)
        black_mask = cv2.morphologyEx(black_mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours in red mask
        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Current frame detection status
        current_detection = False
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 1000:  # Filter small contours
                continue
                
            # Check circularity
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * (area / (perimeter * perimeter))
            if circularity < 0.5:  # Relaxed circularity check
                continue
                
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = float(w) / h
            if aspect_ratio < 0.7 or aspect_ratio > 1.3:  # Relaxed aspect ratio check
                continue
                
            # Check if there's white inside the red contour
            mask_roi = white_mask[y:y+h, x:x+w]
            white_area = cv2.countNonZero(mask_roi)
            if white_area < 0.3 * area:
                continue
                
            # Check if there's black text inside the white area
            black_roi = black_mask[y:y+h, x:x+w]
            black_area = cv2.countNonZero(black_roi)
            if black_area < 0.05 * area:
                continue
                
            # Draw rectangle around the sign
            cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(output, "Speed Limit: 30", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        
            # Add confidence percentage
            confidence_text = f"{int(self.speed_limit_confidence * 100)}%"
            cv2.putText(output, confidence_text, (x, y + h + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Mark as detected in this frame
            current_detection = True
        
        # Update detection history
        self.speed_limit_history.append(current_detection)
        if len(self.speed_limit_history) > self.history_length:
            self.speed_limit_history.pop(0)
            
        # Apply temporal filtering (majority vote)
        if self.speed_limit_history:
            # Count occurrences of detection
            detection_count = sum(self.speed_limit_history)
            
            # Calculate confidence
            self.speed_limit_confidence = detection_count / len(self.speed_limit_history)
            
            # Update detection state if confidence is high enough
            if self.speed_limit_confidence > 0.6:  # More than 60% of recent frames show detection
                self.speed_limit_detected = True
            else:
                self.speed_limit_detected = False
        
        return output, red_mask, white_mask, black_mask
    
    def detect_stop_sign(self, frame):
        """Detect partial stop signs in the frame with relaxed parameters"""
        output = frame.copy()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Red color mask
        mask1 = cv2.inRange(hsv, self.stop_red_lower1, self.stop_red_upper1)
        mask2 = cv2.inRange(hsv, self.stop_red_lower2, self.stop_red_upper2)
        red_mask = cv2.bitwise_or(mask1, mask2)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Current frame detection status
        current_detection = False

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 1000:  # Reduced minimum area for better detection
                continue

            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

            # Accept polygons with 5 to 8 sides (partial octagons)
            if 5 <= len(approx) <= 8:
                x, y, w, h = cv2.boundingRect(approx)
                aspect_ratio = float(w) / h
                
                # Relaxed aspect ratio check (stop sign roughly square)
                if 0.6 <= aspect_ratio <= 1.4:
                    hull = cv2.convexHull(cnt)
                    hull_area = cv2.contourArea(hull)
                    solidity = float(area) / hull_area if hull_area != 0 else 0
                    
                    # Relaxed solidity check
                    if solidity >= 0.75:
                        cv2.drawContours(output, [approx], -1, (0, 255, 0), 4)
                        cv2.putText(output, "STOP SIGN", (x, y - 15),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 3)
                        
                        # Add confidence percentage
                        confidence_text = f"{int(self.stop_detection_confidence * 100)}%"
                        cv2.putText(output, confidence_text, (x, y + h + 20), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        
                        # Mark as detected in this frame
                        current_detection = True

        # Update detection history
        self.stop_detection_history.append(current_detection)
        if len(self.stop_detection_history) > self.history_length:
            self.stop_detection_history.pop(0)
            
        # Apply temporal filtering (majority vote)
        if self.stop_detection_history:
            # Count occurrences of detection
            detection_count = sum(self.stop_detection_history)
            
            # Calculate confidence
            self.stop_detection_confidence = detection_count / len(self.stop_detection_history)
            
            # Update detection state if confidence is high enough
            if self.stop_detection_confidence > 0.6:  # More than 60% of recent frames show detection
                self.stop_sign_detected = True
            else:
                self.stop_sign_detected = False
        
        return output, red_mask
    
    def get_direction(self, center_x, image_center_x, threshold=30):
        """Determine direction based on lane center position"""
        offset = center_x - image_center_x
        if abs(offset) < threshold:
            return "STRAIGHT"
        elif offset < 0:
            return "LEFT"
        else:
            return "RIGHT"
    
    def process_lane_detection(self, frame):
        """Process lane detection and return visualization with temporal filtering"""
        # Create a copy for visualization
        output = frame.copy()
        overlay = np.zeros_like(frame)
        
        # Get binary mask and find contours
        lane_mask = self.get_binary_mask(frame)
        contour = self.find_lane_contours(lane_mask)
        
        # Default direction if no contour found
        detected_direction = "UNKNOWN"
        
        if contour is not None and cv2.contourArea(contour) > 300:
            # Paint lane region
            cv2.drawContours(overlay, [contour], -1, (0, 255, 0), thickness=cv2.FILLED)
            
            pts = contour[:, 0, :]
            left = pts[np.argmin(pts[:, 0])]
            right = pts[np.argmax(pts[:, 0])]
            
            # Visual points
            cv2.circle(output, tuple(left), 5, (0, 0, 255), -1)
            cv2.circle(output, tuple(right), 5, (0, 0, 255), -1)
            
            center = ((left[0] + right[0]) // 2, (left[1] + right[1]) // 2)
            cv2.circle(output, center, 6, (255, 0, 255), -1)
            
            # Image center line
            img_center_x = output.shape[1] // 2
            cv2.line(output, (img_center_x, output.shape[0]), center, (255, 0, 0), 2)
            
            # Direction determination
            detected_direction = self.get_direction(center[0], img_center_x)
        
        # Apply temporal filtering to direction
        self.lane_direction_history.append(detected_direction)
        if len(self.lane_direction_history) > self.history_length:
            self.lane_direction_history.pop(0)
        
        # Only change direction if we have consistent readings
        if len(self.lane_direction_history) >= 3:
            # Count occurrences of each direction
            direction_counts = {}
            for direction in self.lane_direction_history:
                if direction not in direction_counts:
                    direction_counts[direction] = 0
                direction_counts[direction] += 1
            
            # Find the most common direction
            max_count = 0
            max_direction = "UNKNOWN"
            for direction, count in direction_counts.items():
                if count > max_count:
                    max_count = count
                    max_direction = direction
            
            # Only update if the most common direction appears at least 60% of the time
            if max_count >= len(self.lane_direction_history) * 0.6:
                self.current_direction = max_direction
        
        # Blend overlay
        blended = cv2.addWeighted(output, 1, overlay, 0.4, 0)
        
        return blended, lane_mask
    
    def run(self):
        """Run the complete vision system"""
        try:
            while True:
                # Capture frame
                ret, frame = self.camera.read()
                if not ret:
                    print("Failed to capture frame")
                    break
                
                # Process lane detection
                lane_output, lane_mask = self.process_lane_detection(frame)
                
                # Process traffic light detection (every few frames for performance)
                self.frame_count += 1
                if self.frame_count % self.skip_frames == 0:
                    # Detect traffic lights
                    detected_lights = self.detect_traffic_lights(frame, self.all_hue_ranges)
                    # Track lights across frames
                    self.prev_lights, current_lights = self.track_lights(self.prev_lights, detected_lights)
                else:
                    current_lights = self.prev_lights  # Use previous detections for skipped frames
                
                # Get dominant traffic light
                max_intensity = 0
                dominant_light = None
                for light in current_lights:
                    x, y, w, h = light['bbox']
                    color = (0, 0, 255) if light['color'] == 'red' else (0, 255, 0)
                    cv2.rectangle(lane_output, (x, y), (x + w, y + h), color, 2)
                    if light['intensity'] > max_intensity:
                        max_intensity = light['intensity']
                        dominant_light = light
                
                # Update traffic light status
                if dominant_light and max_intensity > 0.3:
                    current_status = dominant_light['color'].upper()
                else:
                    current_status = "NONE"
                
                # Apply temporal filtering to traffic light status
                self.traffic_light_history.append(current_status)
                if len(self.traffic_light_history) > self.history_length:
                    self.traffic_light_history.pop(0)
                
                # Count occurrences of each status
                if self.traffic_light_history:
                    status_counts = {}
                    for status in self.traffic_light_history:
                        if status not in status_counts:
                            status_counts[status] = 0
                        status_counts[status] += 1
                    
                    # Find most common status
                    max_count = 0
                    for status, count in status_counts.items():
                        if count > max_count:
                            max_count = count
                            self.traffic_light_status = status
                    
                    # Calculate confidence
                    self.traffic_light_confidence = max_count / len(self.traffic_light_history)
                
                # Process speed limit sign detection
                speed_output, speed_red_mask, speed_white_mask, speed_black_mask = self.detect_speed_limit(frame)
                
                # Process stop sign detection
                stop_output, stop_mask = self.detect_stop_sign(frame)
                
                # Combine detections into final output
                output = lane_output.copy()
                
                # Add status information
                # Traffic light status
                if self.traffic_light_status == "RED":
                    status_color = (0, 0, 255)  # Red
                elif self.traffic_light_status == "GREEN":
                    status_color = (0, 255, 0)  # Green
                else:
                    status_color = (255, 255, 255)  # White
                
                cv2.putText(output, f"Traffic Light: {self.traffic_light_status} ({int(self.traffic_light_confidence * 100)}%)", 
                           (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
                
                # Speed limit status
                speed_status = "DETECTED" if self.speed_limit_detected else "NOT DETECTED"
                cv2.putText(output, f"Speed Limit: {speed_status} ({int(self.speed_limit_confidence * 100)}%)", 
                           (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                # Stop sign status
                stop_status = "DETECTED" if self.stop_sign_detected else "NOT DETECTED"
                cv2.putText(output, f"Stop Sign: {stop_status} ({int(self.stop_detection_confidence * 100)}%)", 
                           (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Draw direction arrow (flech)
                if self.current_direction != "UNKNOWN":
                    arrow_origin = (output.shape[1] // 2, output.shape[0] - 30)
                    if self.current_direction == "LEFT":
                        arrow_tip = (arrow_origin[0] - 50, arrow_origin[1] - 20)
                    elif self.current_direction == "RIGHT":
                        arrow_tip = (arrow_origin[0] + 50, arrow_origin[1] - 20)
                    else:  # STRAIGHT
                        arrow_tip = (arrow_origin[0], arrow_origin[1] - 50)
                    cv2.arrowedLine(output, arrow_origin, arrow_tip, (0, 255, 255), 4, tipLength=0.4)
                
                # Calculate FPS
                elapsed_time = time.time() - self.start_time
                if elapsed_time > 1:
                    self.fps = self.frame_count / elapsed_time
                    self.frame_count = 0
                    self.start_time = time.time()
                
                # Add FPS counter
                cv2.putText(output, f"FPS: {self.fps:.1f}", 
                           (output.shape[1] - 120, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Show masks in corners if enabled
                if self.show_masks:
                    h, w = frame.shape[:2]
                    mask_size = (80, 60)
                    
                    # Lane mask (top-left)
                    lane_small = cv2.resize(cv2.cvtColor(lane_mask, cv2.COLOR_GRAY2BGR), mask_size)
                    output[10:10+mask_size[1], 10:10+mask_size[0]] = lane_small
                    
                    # Traffic light masks (top-right)
                    if dominant_light:
                        x, y, w_light, h_light = dominant_light['bbox']
                        if 0 <= y < frame.shape[0] and 0 <= x < frame.shape[1] and y+h_light <= frame.shape[0] and x+w_light <= frame.shape[1]:
                            light_roi = frame[y:y+h_light, x:x+w_light]
                            if light_roi.size > 0:  # Check if ROI is valid
                                light_small = cv2.resize(light_roi, mask_size)
                                output[10:10+mask_size[1], output.shape[1]-10-mask_size[0]:output.shape[1]-10] = light_small
                    
                    # Speed limit masks (bottom-left)
                    speed_red_small = cv2.resize(cv2.cvtColor(speed_red_mask, cv2.COLOR_GRAY2BGR), mask_size)
                    speed_red_small[:,:,2] = np.maximum(speed_red_small[:,:,2], 100)  # Add red tint
                    output[h-70-mask_size[1]:h-70, 10:10+mask_size[0]] = speed_red_small
                    
                    speed_white_small = cv2.resize(cv2.cvtColor(speed_white_mask, cv2.COLOR_GRAY2BGR), mask_size)
                    output[h-70-mask_size[1]:h-70, 10+mask_size[0]+10:10+2*mask_size[0]+10] = speed_white_small
                    
                    speed_black_small = cv2.resize(cv2.cvtColor(speed_black_mask, cv2.COLOR_GRAY2BGR), mask_size)
                    output[h-70-mask_size[1]:h-70, 10+2*mask_size[0]+20:10+3*mask_size[0]+20] = speed_black_small
                    
                    # Stop sign mask (bottom-right)
                    stop_small = cv2.resize(cv2.cvtColor(stop_mask, cv2.COLOR_GRAY2BGR), mask_size)
                    stop_small[:,:,2] = np.maximum(stop_small[:,:,2], 100)  # Add red tint
                    output[h-70:h-10, 10:10+mask_size[0]] = stop_small
                
                # Display the resulting frame
                cv2.imshow('Complete Vision System', output)
                
                # Toggle mask display with 'm' key
                key = cv2.waitKey(1) & 0xFF
                if key == ord('m'):
                    self.show_masks = not self.show_masks
                elif key == 27:  # ESC key
                    break
                    
        except KeyboardInterrupt:
            print("Interrupted by user")
            
        finally:
            # Clean up
            self.camera.release()
            cv2.destroyAllWindows()
            print("Vision System Stopped")

if __name__ == "__main__":
    print("Complete Advanced Vision System")
    print("Press 'm' to toggle mask display")
    print("Press ESC to exit")
    system = CompleteVisionSystem()
    system.run()
