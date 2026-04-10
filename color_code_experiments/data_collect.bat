@echo off

:: 1. Tell Python where the root of your project is (where the 'src' folder lives)
set PYTHONPATH=C:\Users\monam\OneDrive\Desktop\UNI\II\PartIIProject\CST-Part-II-Project-Code

:: 2. Navigate to where your scripts actually are
cd C:\Users\monam\OneDrive\Desktop\UNI\II\PartIIProject\CST-Part-II-Project-Code\color_code_experiments

echo Starting Z-basis and Z-basis data collection...
python 488_cc_datacollector.py

echo.
echo All data collection is complete!
pause