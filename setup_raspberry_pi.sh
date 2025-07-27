#!/bin/bash
# Smart Car Vision System Setup Script for Raspberry Pi

echo "Setting up Smart Car Vision System..."

# Create project directory
mkdir -p ~/smart_car_project
cd ~/smart_car_project

# Create virtual environment if it doesn't exist
if [ ! -d "~/py311env" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv ~/py311env
fi

# Activate virtual environment
source ~/py311env/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

echo "Setup complete!"
echo "To run the vision system:"
echo "1. Activate the environment: source ~/py311env/bin/activate"
echo "2. Run the system: python complete_advanced_vision_system.py"
