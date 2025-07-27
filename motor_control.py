#!/usr/bin/env python3
"""
Motor Control Module for Smart Car
Controls 4 motors via dual L298N motor drivers and handles ultrasonic sensor
"""

import RPi.GPIO as GPIO
import time

class MotorController:
    def __init__(self):
        # Pin setup for first L298N (Motors 1 and 2)
        self.IN1 = 17  # GPIO17 - Pin 11
        self.IN2 = 22  # GPIO22 - Pin 15
        self.IN3 = 23  # GPIO23 - Pin 16
        self.IN4 = 24  # GPIO24 - Pin 18
        self.ENA = 18  # GPIO18 - Pin 12 (PWM for Motor 1)
        self.ENB = 25  # GPIO25 - Pin 22 (PWM for Motor 2)

        # Pin setup for second L298N (Motors 3 and 4)
        self.IN5 = 5   # GPIO5  - Pin 29
        self.IN6 = 6   # GPIO6  - Pin 31
        self.IN7 = 13  # GPIO13 - Pin 33
        self.IN8 = 19  # GPIO19 - Pin 35
        self.ENC = 12  # GPIO12 - Pin 32 (PWM for Motor 3)
        self.END = 16  # GPIO16 - Pin 36 (PWM for Motor 4)

        # Pin setup for HC-SR04 ultrasonic sensor
        self.TRIG = 27  # GPIO27 - Pin 13
        self.ECHO = 26  # GPIO26 - Pin 37
        
        # Initialize GPIO
        self._setup_gpio()
        
        # Default speed
        self.default_speed = 75
        
        # Obstacle detection threshold (cm)
        self.obstacle_threshold = 20
        
    def _setup_gpio(self):
        """Set up GPIO pins for motors and ultrasonic sensor"""
        # GPIO setup
        GPIO.setmode(GPIO.BCM)
        GPIO.setup([self.IN1, self.IN2, self.IN3, self.IN4, self.IN5, self.IN6, self.IN7, self.IN8], GPIO.OUT)
        GPIO.setup([self.ENA, self.ENB, self.ENC, self.END], GPIO.OUT)
        GPIO.setup(self.TRIG, GPIO.OUT)
        GPIO.setup(self.ECHO, GPIO.IN)

        # PWM setup (50Hz for L298N)
        self.pwm1 = GPIO.PWM(self.ENA, 50)  # Motor 1
        self.pwm2 = GPIO.PWM(self.ENB, 50)  # Motor 2
        self.pwm3 = GPIO.PWM(self.ENC, 50)  # Motor 3
        self.pwm4 = GPIO.PWM(self.END, 50)  # Motor 4
        self.pwm1.start(0)
        self.pwm2.start(0)
        self.pwm3.start(0)
        self.pwm4.start(0)
        
        # Initialize ultrasonic sensor
        GPIO.output(self.TRIG, GPIO.LOW)
        time.sleep(0.1)  # Allow sensor to settle

    # Motor control functions
    def motor1_forward(self, speed=None):
        if speed is None:
            speed = self.default_speed
        GPIO.output(self.IN1, GPIO.HIGH)
        GPIO.output(self.IN2, GPIO.LOW)
        self.pwm1.ChangeDutyCycle(speed)

    def motor1_backward(self, speed=None):
        if speed is None:
            speed = self.default_speed
        GPIO.output(self.IN1, GPIO.LOW)
        GPIO.output(self.IN2, GPIO.HIGH)
        self.pwm1.ChangeDutyCycle(speed)

    def motor1_stop(self):
        GPIO.output(self.IN1, GPIO.LOW)
        GPIO.output(self.IN2, GPIO.LOW)
        self.pwm1.ChangeDutyCycle(0)

    def motor2_forward(self, speed=None):
        if speed is None:
            speed = self.default_speed
        GPIO.output(self.IN3, GPIO.HIGH)
        GPIO.output(self.IN4, GPIO.LOW)
        self.pwm2.ChangeDutyCycle(speed)

    def motor2_backward(self, speed=None):
        if speed is None:
            speed = self.default_speed
        GPIO.output(self.IN3, GPIO.LOW)
        GPIO.output(self.IN4, GPIO.HIGH)
        self.pwm2.ChangeDutyCycle(speed)

    def motor2_stop(self):
        GPIO.output(self.IN3, GPIO.LOW)
        GPIO.output(self.IN4, GPIO.LOW)
        self.pwm2.ChangeDutyCycle(0)

    def motor3_forward(self, speed=None):
        if speed is None:
            speed = self.default_speed
        GPIO.output(self.IN5, GPIO.HIGH)
        GPIO.output(self.IN6, GPIO.LOW)
        self.pwm3.ChangeDutyCycle(speed)

    def motor3_backward(self, speed=None):
        if speed is None:
            speed = self.default_speed
        GPIO.output(self.IN5, GPIO.LOW)
        GPIO.output(self.IN6, GPIO.HIGH)
        self.pwm3.ChangeDutyCycle(speed)

    def motor3_stop(self):
        GPIO.output(self.IN5, GPIO.LOW)
        GPIO.output(self.IN6, GPIO.LOW)
        self.pwm3.ChangeDutyCycle(0)

    def motor4_forward(self, speed=None):
        if speed is None:
            speed = self.default_speed
        GPIO.output(self.IN7, GPIO.HIGH)
        GPIO.output(self.IN8, GPIO.LOW)
        self.pwm4.ChangeDutyCycle(speed)

    def motor4_backward(self, speed=None):
        if speed is None:
            speed = self.default_speed
        GPIO.output(self.IN7, GPIO.LOW)
        GPIO.output(self.IN8, GPIO.HIGH)
        self.pwm4.ChangeDutyCycle(speed)

    def motor4_stop(self):
        GPIO.output(self.IN7, GPIO.LOW)
        GPIO.output(self.IN8, GPIO.LOW)
        self.pwm4.ChangeDutyCycle(0)

    # Combined motor control functions
    def all_motors_forward(self, speed=None):
        if speed is None:
            speed = self.default_speed
        self.motor1_forward(speed)
        self.motor2_forward(speed)
        self.motor3_forward(speed)
        self.motor4_forward(speed)

    def all_motors_backward(self, speed=None):
        if speed is None:
            speed = self.default_speed
        self.motor1_backward(speed)
        self.motor2_backward(speed)
        self.motor3_backward(speed)
        self.motor4_backward(speed)

    def all_motors_stop(self):
        self.motor1_stop()
        self.motor2_stop()
        self.motor3_stop()
        self.motor4_stop()
        
    def turn_left(self, speed=None):
        """Turn left by running right motors forward and left motors backward"""
        if speed is None:
            speed = self.default_speed
        # Left side motors backward
        self.motor1_backward(speed)
        self.motor3_backward(speed)
        # Right side motors forward
        self.motor2_forward(speed)
        self.motor4_forward(speed)
        
    def turn_right(self, speed=None):
        """Turn right by running left motors forward and right motors backward"""
        if speed is None:
            speed = self.default_speed
        # Left side motors forward
        self.motor1_forward(speed)
        self.motor3_forward(speed)
        # Right side motors backward
        self.motor2_backward(speed)
        self.motor4_backward(speed)

    # Ultrasonic sensor function
    def get_distance(self):
        """Get distance from ultrasonic sensor in cm"""
        # Send 10us trigger pulse
        GPIO.output(self.TRIG, GPIO.HIGH)
        time.sleep(0.00001)  # 10 microseconds
        GPIO.output(self.TRIG, GPIO.LOW)

        # Wait for echo start
        pulse_start = time.time()
        timeout = pulse_start + 0.1  # 100ms timeout
        while GPIO.input(self.ECHO) == 0:
            pulse_start = time.time()
            if pulse_start > timeout:
                return 400  # Return max distance if timeout

        # Wait for echo end
        pulse_end = time.time()
        timeout = pulse_end + 0.1  # 100ms timeout
        while GPIO.input(self.ECHO) == 1:
            pulse_end = time.time()
            if pulse_end > timeout:
                return 400  # Return max distance if timeout

        # Calculate distance (speed of sound = 343m/s = 0.0343cm/us)
        pulse_duration = pulse_end - pulse_start
        distance = pulse_duration * 17150  # (34300 cm/s) / 2
        return round(distance, 2)
    
    def check_obstacle(self):
        """Check if there's an obstacle in front of the car"""
        distance = self.get_distance()
        return distance < self.obstacle_threshold, distance
    
    def cleanup(self):
        """Clean up GPIO resources"""
        self.all_motors_stop()
        self.pwm1.stop()
        self.pwm2.stop()
        self.pwm3.stop()
        self.pwm4.stop()
        GPIO.cleanup()

# Test code when run directly
if __name__ == "__main__":
    try:
        motor_controller = MotorController()
        print("Motor controller initialized")
        
        while True:
            # Measure distance
            distance = motor_controller.get_distance()
            print(f"Distance: {distance} cm")

            # Obstacle detection (stop if closer than threshold)
            if distance < motor_controller.obstacle_threshold:
                print("Obstacle detected! Stopping motors")
                motor_controller.all_motors_stop()
                time.sleep(1)  # Wait before next measurement
            else:
                print("Path clear, moving forward")
                motor_controller.all_motors_forward()  # Move forward at default speed
                time.sleep(0.1)  # Short delay to avoid overloading sensor

    except KeyboardInterrupt:
        print("Exiting program")

    finally:
        if 'motor_controller' in locals():
            motor_controller.cleanup()
