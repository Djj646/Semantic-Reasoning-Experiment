@echo off

echo Running command 1 ...
python img2pm_fakenav.py --model UNet

echo Running command 2 ...
python img2pm_fakenav.py --model CNN

echo All commands finished.