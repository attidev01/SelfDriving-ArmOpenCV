@echo off
echo Transferring Smart Car Vision System files to Raspberry Pi...

REM Set Raspberry Pi connection details
set PI_IP=192.168.137.173
set PI_USER=pi
set PI_PASSWORD=raspberry
set PI_FOLDER=/home/pi/smart_car_project/

REM Transfer all Python files
pscp -pw %PI_PASSWORD% -batch "%~dp0*.py" %PI_USER%@%PI_IP%:%PI_FOLDER%

REM Transfer requirements.txt
pscp -pw %PI_PASSWORD% -batch "%~dp0requirements.txt" %PI_USER%@%PI_IP%:%PI_FOLDER%

REM Transfer setup script
pscp -pw %PI_PASSWORD% -batch "%~dp0setup_raspberry_pi.sh" %PI_USER%@%PI_IP%:%PI_FOLDER%

echo Transfer complete!
echo.
echo To set up the environment on your Raspberry Pi:
echo 1. SSH into your Pi: ssh %PI_USER%@%PI_IP%
echo 2. Run: cd %PI_FOLDER% && chmod +x setup_raspberry_pi.sh && ./setup_raspberry_pi.sh
echo.
pause
