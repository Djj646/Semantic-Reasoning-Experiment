@echo off

set model=MA2
set distances=016 044

for %%d in (%distances%) do (
    call :run_command %%d
)

echo All commands finished.
goto :eof

:run_command
set distance=%1

echo Running command for model %model% %distance% ...
python eval_turning.py --model %model% --rate %distance% --turning

goto :eof
