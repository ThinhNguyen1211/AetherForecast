@echo off
setlocal EnableExtensions

set SCRIPT_DIR=%~dp0
for %%I in ("%SCRIPT_DIR%..") do set ROOT=%%~fI

set VENV_DIR=%ROOT%\train-gpu
set PY=%VENV_DIR%\Scripts\python.exe

if not exist "%PY%" (
  echo [run-train-gpu] Missing GPU venv: %VENV_DIR%
  echo [run-train-gpu] Create it first or rerun environment setup.
  exit /b 1
)

call "%VENV_DIR%\Scripts\activate.bat"
set PYTHONNOUSERSITE=1

if not defined AWS_REGION set AWS_REGION=ap-southeast-1
if not defined DATA_BUCKET set DATA_BUCKET=aetherforecast-data-800762439372-ap-southeast-1
if not defined DATA_S3_BUCKET set DATA_S3_BUCKET=%DATA_BUCKET%
if not defined MODEL_BUCKET set MODEL_BUCKET=aetherforecast-models-800762439372-ap-southeast-1
if not defined SYMBOLS set SYMBOLS=BTCUSDT,ETHUSDT,SOLUSDT,XAUUSD,BNBUSDT,PAXGUSDT
if not defined TIMEFRAME set TIMEFRAME=1h
if not defined EPOCHS set EPOCHS=2
if not defined BATCH_SIZE set BATCH_SIZE=2
if not defined GRAD_ACCUM_STEPS set GRAD_ACCUM_STEPS=8
if not defined LEARNING_RATE set LEARNING_RATE=0.0002
if not defined BASE_MODEL_ID set BASE_MODEL_ID=amazon/chronos-2
if not defined BASE_MODEL_FALLBACK_ID set BASE_MODEL_FALLBACK_ID=amazon/chronos-t5-large

echo [run-train-gpu] Python: %PY%
echo [run-train-gpu] AWS_REGION=%AWS_REGION%
echo [run-train-gpu] DATA_BUCKET=%DATA_BUCKET%
echo [run-train-gpu] MODEL_BUCKET=%MODEL_BUCKET%
echo [run-train-gpu] SYMBOLS=%SYMBOLS%
echo [run-train-gpu] TIMEFRAME=%TIMEFRAME%
echo [run-train-gpu] Checking CUDA availability...
"%PY%" -c "import sys, torch; print('python=', sys.executable); print('torch=', torch.__version__); print('torch.cuda=', torch.version.cuda); print('cuda_available=', torch.cuda.is_available()); print('gpu=', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
if errorlevel 1 exit /b 1

if "%FORCE_CPU%"=="1" (
  set CUDA_VISIBLE_DEVICES=
  echo [run-train-gpu] FORCE_CPU=1 -> running on CPU
) else (
  set CUDA_VISIBLE_DEVICES=0
)

set HF_HOME=%ROOT%\artifacts\hf-cache
set HF_CACHE_DIR=%ROOT%\artifacts\hf-cache

cd /d "%ROOT%\packages\backend"
"%PY%" "%ROOT%\packages\backend\train-local.py" ^
  --aws-region %AWS_REGION% ^
  --data-bucket %DATA_BUCKET% ^
  --model-bucket %MODEL_BUCKET% ^
  --symbols %SYMBOLS% ^
  --timeframe %TIMEFRAME% ^
  --epochs %EPOCHS% ^
  --batch-size %BATCH_SIZE% ^
  --grad-accum-steps %GRAD_ACCUM_STEPS% ^
  --learning-rate %LEARNING_RATE% ^
  --base-model-id %BASE_MODEL_ID% ^
  --base-model-fallback-id %BASE_MODEL_FALLBACK_ID%

set EXIT_CODE=%ERRORLEVEL%
if not "%EXIT_CODE%"=="0" (
  echo [run-train-gpu] Training failed with exit code %EXIT_CODE%.
) else (
  echo [run-train-gpu] Training completed successfully.
)

exit /b %EXIT_CODE%
