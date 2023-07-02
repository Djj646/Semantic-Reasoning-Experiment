@echo off

set models=MA2 UNet CNN SD
set distance=006

for %%d in (%models%) do (
    call :run_command %%d
)

echo All commands finished.
goto :eof

:run_command
set model=%1

echo Running command for model %model% %distance% ...
python eval_turning.py --model %model% --rate %distance%
python eval_turning.py --model %model% --rate %distance% --turning

goto :eof
