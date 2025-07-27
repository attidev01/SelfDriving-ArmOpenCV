#!/usr/bin/env python3
"""
Partial Stop Sign Detection System
Uses shape detection and color filtering to identify partially visible stop signs
"""

import cv2
import numpy as np
import time
import os

class StopSignDetector:
    def __init__(self, camera_id=0, resolution=(640, 480)):
        """Initialize the stop sign detector"""
        self.camera = cv2.VideoCapture(camera_id)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
        
        # Performance tracking
        self.fps = 0
        self.frame_count = 0
        self.start_time = time.time()
        
        # Stop sign detection parameters
        # Red color mask (cover both ends of hue)
        self.stop_red_lower1 = np.array([0, 70, 50])
        self.stop_red_upper1 = np.array([10, 255, 255])
        self.stop_red_lower2 = np.array([170, 70, 50])
        self.stop_red_upper2 = np.array([180, 255, 255])
        
        # Stop sign state tracking
        self.sign_detected = False
        self.detection_history = []
        self.history_length = 5  # Number of frames to keep in history
        self.detection_confidence = 0
        
        print("Stop Sign Detector Initialized")
        print(f"Camera resolution: {resolution}")
    
    def preprocess_frame(self, frame):
        """Apply preprocessing to improve detection quality"""
        # Apply slight Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(frame, (5, 5), 0)
        
        # Enhance contrast using CLAHE
        lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        enhanced_lab = cv2.merge((cl, a, b))
        enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        
        return enhanced
    
    def detect_stop_sign(self, frame):
        """Detect partial stop signs in the frame with relaxed parameters"""
        # Preprocess the frame
        enhanced = self.preprocess_frame(frame)
        
        output = enhanced.copy()
        hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV)

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
                        confidence_text = f"{int(self.detection_confidence * 100)}%"
                        cv2.putText(output, confidence_text, (x, y + h + 20), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        
                        # Mark as detected in this frame
                        current_detection = True
        
        # Update detection history
        self.detection_history.append(current_detection)
        if len(self.detection_history) > self.history_length:
            self.detection_history.pop(0)
            
        # Apply temporal filtering (majority vote)
        if self.detection_history:
            # Count occurrences of detection
            detection_count = sum(self.detection_history)
            
            # Calculate confidence
            self.detection_confidence = detection_count / len(self.detection_history)
            
            # Update detection state if confidence is high enough
            if self.detection_confidence > 0.6:  # More than 60% of recent frames show detection
                self.sign_detected = True
            else:
                self.sign_detected = False
        
        # Add stop sign detection status
        status_text = "DETECTED" if self.sign_detected else "NOT DETECTED"
        confidence_text = f"{int(self.detection_confidence * 100)}%" if self.detection_history else "0%"
        cv2.putText(output, f"Stop Sign: {status_text} ({confidence_text})", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Show the mask in corner
        h, w = frame.shape[:2]
        mask_size = (80, 60)
        
        red_small = cv2.resize(cv2.cvtColor(red_mask, cv2.COLOR_GRAY2BGR), mask_size)
        red_small[:,:,2] = np.maximum(red_small[:,:,2], 100)  # Add red tint
        output[h-70:h-10, 10:10+mask_size[0]] = red_small
        
        return output, red_mask
    
    def run(self):
        """Run the stop sign detection loop"""
        try:
            while True:
                # Capture frame
                ret, frame = self.camera.read()
                if not ret:
                    print("Failed to capture frame")
                    break
                
                # Process frame
                result, red_mask = self.detect_stop_sign(frame)
                
                # Calculate FPS
                self.frame_count += 1
                elapsed_time = time.time() - self.start_time
                if elapsed_time > 1:
                    self.fps = self.frame_count / elapsed_time
                    self.frame_count = 0
                    self.start_time = time.time()
                
                # Add FPS counter
                cv2.putText(result, f"FPS: {self.fps:.1f}", 
                           (result.shape[1] - 120, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Display the resulting frame
                cv2.imshow('Stop Sign Detection', result)
                
                # Break on ESC key
                if cv2.waitKey(1) == 27:
                    break
                    
        except KeyboardInterrupt:
            print("Interrupted by user")
            
        finally:
            # Clean up
            self.camera.release()
            cv2.destroyAllWindows()
            print("Stop Sign Detection Stopped")

if __name__ == "__main__":
    print("Partial Stop Sign Detection")
    print("Press ESC to exit")
    detector = StopSignDetector()
    detector.run()
