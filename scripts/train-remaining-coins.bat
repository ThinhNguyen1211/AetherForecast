@echo off
setlocal EnableExtensions EnableDelayedExpansion

rem ---------------------------------------------------------------------------
rem AetherForecast - Train all remaining symbols after the first 40-coin set.
rem - Auto-discovers symbols from S3 parquet partitions: market/klines/symbol=*
rem - Excludes the 40 symbols already trained
rem - Trains in chunks to avoid Windows command length limits
rem - Appends full output to artifacts\train-remaining.log
rem ---------------------------------------------------------------------------

set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%..") do set "ROOT=%%~fI"

set "LOG_FILE=%ROOT%\artifacts\train-remaining.log"
set "CHUNK_FILE=%ROOT%\artifacts\remaining-symbol-chunks.txt"
set "PY=%ROOT%\train-gpu\Scripts\python.exe"
set "SYMBOL_DISCOVERY_SCRIPT=%SCRIPT_DIR%generate_remaining_symbol_chunks.py"
set /a EXIT_CODE=0

if not exist "%ROOT%\artifacts" mkdir "%ROOT%\artifacts"

if not exist "%PY%" (
  echo [train-remaining] Missing Python: %PY%
  exit /b 1
)

if not exist "%SYMBOL_DISCOVERY_SCRIPT%" (
  echo [train-remaining] Missing symbol discovery script: %SYMBOL_DISCOVERY_SCRIPT%
  exit /b 1
)

if /I "%DATA_BUCKET%"=="dummy-bucket" set "DATA_BUCKET="
if /I "%DATA_S3_BUCKET%"=="dummy-bucket" set "DATA_S3_BUCKET="
if /I "%MODEL_BUCKET%"=="dummy-model" set "MODEL_BUCKET="

if not defined AWS_REGION set "AWS_REGION=ap-southeast-1"
if not defined DATA_BUCKET set "DATA_BUCKET=aetherforecast-data-800762439372-ap-southeast-1"
if not defined DATA_S3_BUCKET set "DATA_S3_BUCKET=%DATA_BUCKET%"
if not defined MODEL_BUCKET set "MODEL_BUCKET=aetherforecast-models-800762439372-ap-southeast-1"
if not defined PARQUET_PREFIX set "PARQUET_PREFIX=market/klines"
if not defined REMAINING_SYMBOL_CHUNK_SIZE set "REMAINING_SYMBOL_CHUNK_SIZE=20"

set "TIMEFRAME=all"
set "EPOCHS=2"
set "BATCH_SIZE=2"
set "GRAD_ACCUM_STEPS=8"
set "LEARNING_RATE=0.0002"
set "MAX_SEQ_LENGTH=512"
set "BASE_MODEL_ID=amazon/chronos-2"
set "BASE_MODEL_FALLBACK_ID=amazon/chronos-t5-large"
set "HF_FORCE_DOWNLOAD=1"
set "HF_LOCAL_FILES_ONLY=0"
set "FORCE_CPU=0"
set "CHRONOS2_TRAIN_STEPS="

set "REMAINING_CHUNK_FILE=%CHUNK_FILE%"

echo [train-remaining] Start: %DATE% %TIME%> "%LOG_FILE%"
echo [train-remaining] ROOT=%ROOT%>> "%LOG_FILE%"
echo [train-remaining] AWS_REGION=%AWS_REGION%>> "%LOG_FILE%"
echo [train-remaining] DATA_BUCKET=%DATA_BUCKET%>> "%LOG_FILE%"
echo [train-remaining] MODEL_BUCKET=%MODEL_BUCKET%>> "%LOG_FILE%"
echo [train-remaining] PARQUET_PREFIX=%PARQUET_PREFIX%>> "%LOG_FILE%"
echo [train-remaining] CHUNK_SIZE=%REMAINING_SYMBOL_CHUNK_SIZE%>> "%LOG_FILE%"

echo [train-remaining] Discovering remaining symbols...>> "%LOG_FILE%"
"%PY%" "%SYMBOL_DISCOVERY_SCRIPT%" >> "%LOG_FILE%" 2>&1
if errorlevel 1 (
  echo [train-remaining] Failed while generating remaining symbol chunks.>> "%LOG_FILE%"
  set /a EXIT_CODE=1
  goto :cleanup
)

if /I "%TRAIN_REMAINING_DRY_RUN%"=="1" (
  echo [train-remaining] DRY RUN only. Skipping training chunks.>> "%LOG_FILE%"
  goto :cleanup
)

set /a CHUNK_INDEX=0

for /f "usebackq delims=" %%S in ("%CHUNK_FILE%") do (
  set /a CHUNK_INDEX+=1
  set "SYMBOLS=%%S"
  echo [train-remaining] ==== Chunk !CHUNK_INDEX! START ====>> "%LOG_FILE%"
  echo [train-remaining] SYMBOLS=!SYMBOLS!>> "%LOG_FILE%"
  call "%SCRIPT_DIR%run-train-gpu.bat" >> "%LOG_FILE%" 2>&1
  if errorlevel 1 (
    echo [train-remaining] Chunk !CHUNK_INDEX! FAILED. Stopping.>> "%LOG_FILE%"
    set /a EXIT_CODE=1
    goto :cleanup
  )
  echo [train-remaining] ==== Chunk !CHUNK_INDEX! DONE ====>> "%LOG_FILE%"
)

if !CHUNK_INDEX! EQU 0 (
  echo [train-remaining] No chunks found in %CHUNK_FILE%.>> "%LOG_FILE%"
  set /a EXIT_CODE=1
  goto :cleanup
)

echo [train-remaining] All chunks completed successfully.>> "%LOG_FILE%"

:cleanup
if %EXIT_CODE% EQU 0 (
  echo [train-remaining] Finished: %DATE% %TIME%>> "%LOG_FILE%"
) else (
  echo [train-remaining] Failed: %DATE% %TIME%>> "%LOG_FILE%"
)

endlocal & exit /b %EXIT_CODE%
