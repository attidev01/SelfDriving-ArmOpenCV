# Smart Car Vision System- IAE UIZ

## Overview
This repository contains a comprehensive real-time vision system for Raspberry Pi that integrates multiple detection capabilities for autonomous driving applications:

- Lane detection with direction guidance
- Advanced adaptive traffic light detection
- Speed limit sign detection
- Partial stop sign detection

The system uses efficient computer vision techniques optimized for Raspberry Pi, including HSV color segmentation, contour analysis, shape detection, and temporal filtering.

## Features

### Lane Detection
- Detects lane markings using HSV color segmentation
- Provides directional guidance (LEFT/RIGHT/STRAIGHT)
- Includes visual arrow indicator for direction
- Uses temporal filtering for stable detection

### Traffic Light Detection
- Detects and classifies traffic lights as RED or GREEN
- Uses adaptive region tracking for improved detection
- Implements temporal filtering to prevent false positives
- Displays confidence percentage for detected state

### Speed Limit Sign Detection
- Detects circular speed limit signs
- Uses multi-color HSV segmentation (red border, white interior)
- Shows confidence percentage for detections
- Placeholder for actual speed value recognition

### Stop Sign Detection
- Detects partial and complete stop signs
- Relaxed polygon detection (5-8 sides)
- Uses HSV red color segmentation
- Implements temporal filtering for stability

## Requirements
- Raspberry Pi 4 (recommended) or 3B+
- Python 3.7+ with OpenCV 4.x
- USB webcam or Raspberry Pi Camera Module
- See requirements.txt for detailed dependencies

## Installation

### On Raspberry Pi
1. Clone this repository
2. Create a virtual environment:
   ```
   python3 -m venv py311env
   source py311env/bin/activate
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage
Run the complete vision system:
```
python complete_advanced_vision_system.py
```

### Interactive Controls
- Press 'm' to toggle mask display
- Press ESC to exit

## Files
- `complete_advanced_vision_system.py`: Main integrated vision system
- `adaptive_traffic_light.py`: Standalone traffic light detection
- `partial_stop_sign_detection.py`: Standalone stop sign detection

## Performance Optimization
The system is optimized for Raspberry Pi with:
- Frame skipping for intensive operations
- Efficient morphological operations
- Lightweight contour processing
- Optimized HSV color segmentation

## License
This project is provided for educational purposes.
