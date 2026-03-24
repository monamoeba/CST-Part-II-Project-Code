@echo off
echo Starting Z-basis data collection...
python 666_cc_datacollector.py

echo.
echo Z-basis collection finished!
echo Starting X-basis data collection...
python 666_cc_datacollector_XL.py

echo.
echo All data collection is complete!
pause