@echo off

set models=MA2 UNet CNN SD

for %%d in (%models%) do (
    call :run_command %%d
)

echo All commands finished.
goto :eof

:run_command
set model=%1

echo Running command for model %model% ...
python hm.py --model %model% --data 1

goto :eof
