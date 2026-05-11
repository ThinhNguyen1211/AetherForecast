@echo off
setlocal EnableExtensions EnableDelayedExpansion

:: ============================================================
:: AetherForecast - Production Training Batch Runner
:: Double-click to launch. This script:
::   1. Detects GPU + Python venv
::   2. Resolves all symbols (S3 or fallback list)
::   3. Runs train-local.py via train-stream.ps1 (live terminal output)
::   4. Promotes model via manifest/latest.json
::   5. Keeps terminal open until you press a key
:: ============================================================

:: ── Auto-open a persistent window if launched by double-click ───────────────
:: (Explorer runs bat files without /k, so the window closes on exit)
if "%AETHER_TRAINING_CONSOLE%"=="" (
  set "AETHER_TRAINING_CONSOLE=1"
  start "AetherForecast Training" cmd /k ""%~f0""
  exit /b 0
)

title AetherForecast - Training Pipeline

set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%..\..") do set "ROOT=%%~fI"
set "BACKEND_DIR=%SCRIPT_DIR%"
cd /d "%BACKEND_DIR%"

:: ── Python selection ─────────────────────────────────────────────────────────
set "PY="
set "GPU_VISIBLE=0"
set "PY_TORCH_CUDA=0"

call :detect_gpu

:: Prefer train-gpu venv (has GPU torch) then .venv fallback
for %%P in (
  "%ROOT%\train-gpu\Scripts\python.exe"
  "%ROOT%\train-gpu\backend-code\.venv\Scripts\python.exe"
  "%ROOT%\.venv\Scripts\python.exe"
) do (
  if not defined PY (
    if exist %%P (
      set "PY=%%~fP"
    )
  )
)

if not defined PY (
  echo.
  echo [train-batch] ERROR: No Python venv found.
  echo [train-batch] Expected: %ROOT%\train-gpu\Scripts\python.exe
  echo [train-batch]       or: %ROOT%\.venv\Scripts\python.exe
  goto :fatal
)

:: Probe CUDA
call :probe_cuda

echo.
echo [train-batch] ============================================================
echo [train-batch]   AetherForecast - Production Training Pipeline
echo [train-batch] ============================================================
echo [train-batch] Python   : %PY%
if "%PY_TORCH_CUDA%"=="1" (
  echo [train-batch] GPU      : ENABLED ^(CUDA^)
) else (
  echo [train-batch] GPU      : NOT AVAILABLE - will train on CPU
)

:: ── Hyperparameters ──────────────────────────────────────────────────────────
if not defined AWS_REGION                   set "AWS_REGION=ap-southeast-1"
if not defined DATA_BUCKET                  set "DATA_BUCKET=aetherforecast-data-800762439372-ap-southeast-1"
if not defined DATA_S3_BUCKET               set "DATA_S3_BUCKET=%DATA_BUCKET%"
if not defined MODEL_BUCKET                 set "MODEL_BUCKET=aetherforecast-models-800762439372-ap-southeast-1"
if not defined PARQUET_PREFIX               set "PARQUET_PREFIX=market/klines"
if not defined TIMEFRAME                    set "TIMEFRAME=1h,4h,1d"
if not defined EPOCHS                       set "EPOCHS=5"
if not defined BATCH_SIZE                   set "BATCH_SIZE=2"
if not defined GRAD_ACCUM_STEPS             set "GRAD_ACCUM_STEPS=8"
if not defined LEARNING_RATE                set "LEARNING_RATE=0.0002"
if not defined MAX_ROWS_PER_SYMBOL          set "MAX_ROWS_PER_SYMBOL=12000"
if not defined CONTEXT_LENGTH               set "CONTEXT_LENGTH=1024"
if not defined MAX_SEQ_LENGTH               set "MAX_SEQ_LENGTH=1024"
if not defined TRAINING_HORIZON             set "TRAINING_HORIZON=7"
if not defined LORA_R                       set "LORA_R=16"
if not defined LORA_ALPHA                   set "LORA_ALPHA=32"
if not defined LORA_DROPOUT                 set "LORA_DROPOUT=0.05"
if not defined SAVE_STEPS                   set "SAVE_STEPS=50"
if not defined EVAL_STEPS                   set "EVAL_STEPS=50"
if not defined LOGGING_STEPS                set "LOGGING_STEPS=5"
if not defined TRAIN_SPLIT_RATIO            set "TRAIN_SPLIT_RATIO=0.95"
if not defined WALK_FORWARD_WINDOWS         set "WALK_FORWARD_WINDOWS=5"
if not defined WALK_FORWARD_EVAL_SIZE       set "WALK_FORWARD_EVAL_SIZE=128"
if not defined EXTERNAL_COVARIATE_SCALE     set "EXTERNAL_COVARIATE_SCALE=0.0018"
if not defined PREDICT_VARIANCE_SCALE       set "PREDICT_VARIANCE_SCALE=1.18"
if not defined PREDICT_DIFFUSION_STEPS      set "PREDICT_DIFFUSION_STEPS=3"
if not defined ENABLE_EXTERNAL_FETCH        set "ENABLE_EXTERNAL_FETCH=1"
if not defined STRICT_EXTERNAL_DATA         set "STRICT_EXTERNAL_DATA=0"
if not defined BASE_MODEL_ID                set "BASE_MODEL_ID=amazon/chronos-2"
if not defined BASE_MODEL_FALLBACK_ID       set "BASE_MODEL_FALLBACK_ID=amazon/chronos-t5-large"
if not defined MAX_PARQUET_FILES_PER_SYMBOL set "MAX_PARQUET_FILES_PER_SYMBOL=320"
if not defined COIN_GROUP_SIZE              set "COIN_GROUP_SIZE=90"

set "LIVE_FETCH_FLAG=--enable-live-external-fetch"
if /I "%ENABLE_EXTERNAL_FETCH%"=="0" set "LIVE_FETCH_FLAG=--disable-live-external-fetch"
set "STRICT_FLAG=--allow-external-fallback"
if /I "%STRICT_EXTERNAL_DATA%"=="1" set "STRICT_FLAG=--strict-external-data"

:: Python env flags
set "PYTHONUNBUFFERED=1"
set "PYTHONDONTWRITEBYTECODE=1"
set "PYTHONIOENCODING=utf-8"

:: ── Log dir + lock ───────────────────────────────────────────────────────────
set "LOG_DIR=%ROOT%\artifacts\train-batch-logs"
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

set "LOCK_FILE=%LOG_DIR%\train-batch.lock"
set "EXIT_CODE=0"

if exist "%LOCK_FILE%" (
  echo [train-batch] Stale lock found - removing it...
  del /q "%LOCK_FILE%" >nul 2>&1
)
> "%LOCK_FILE%" echo started=%DATE% %TIME%

for /f %%I in ('powershell -NoProfile -Command "(Get-Date).ToString(\"yyyyMMdd-HHmmss\")"') do set "RUN_ID=%%I"
if not defined RUN_ID set "RUN_ID=manual"

set "SUMMARY_LOG=%LOG_DIR%\train-batch-%RUN_ID%.log"
set "SYMBOL_FILE=%LOG_DIR%\symbols-%RUN_ID%.txt"
set "GROUP_FILE=%LOG_DIR%\groups-%RUN_ID%.txt"

echo [train-batch] AWS_REGION  : %AWS_REGION%
echo [train-batch] DATA_BUCKET : %DATA_BUCKET%
echo [train-batch] MODEL_BUCKET: %MODEL_BUCKET%
echo [train-batch] TIMEFRAME   : %TIMEFRAME%
echo [train-batch] EPOCHS=%EPOCHS%  LR=%LEARNING_RATE%  BATCH=%BATCH_SIZE%  GRAD_ACCUM=%GRAD_ACCUM_STEPS%
echo [train-batch] LORA_R=%LORA_R% ALPHA=%LORA_ALPHA% DROP=%LORA_DROPOUT%
echo [train-batch] CONTEXT=%CONTEXT_LENGTH%  ROWS=%MAX_ROWS_PER_SYMBOL%  WALK_FWD_WIN=%WALK_FORWARD_WINDOWS%
echo [train-batch] Run started: %DATE% %TIME%
echo [train-batch] Summary log: %SUMMARY_LOG%
echo.

>> "%SUMMARY_LOG%" echo [train-batch] Run started: %DATE% %TIME%
>> "%SUMMARY_LOG%" echo [train-batch] Python: %PY%

:: ── Resolve symbols ───────────────────────────────────────────────────────────
echo [train-batch] Resolving symbols from S3...
"%PY%" -c "import os,sys,boto3; bucket=os.environ.get('DATA_BUCKET',''); prefix=os.environ.get('PARQUET_PREFIX','market/klines').strip('/'); region=os.environ.get('AWS_REGION','ap-southeast-1'); s3=boto3.client('s3',region_name=region); pag=s3.get_paginator('list_objects_v2'); syms=set(); [syms.add(p['Prefix'].split('symbol=')[1].split('/')[0].upper()) for page in pag.paginate(Bucket=bucket,Prefix=prefix+'/',Delimiter='/') for p in page.get('CommonPrefixes',[]) if 'symbol=' in p.get('Prefix','')]; sys.exit(1) if not syms else sys.stdout.write('\n'.join(sorted(syms)))" > "%SYMBOL_FILE%" 2>> "%SUMMARY_LOG%"

if errorlevel 1 (
  echo [train-batch] S3 symbol fetch failed ^(see log^). Using bundled fallback list...
  >> "%SUMMARY_LOG%" echo [train-batch] WARNING: S3 symbol fetch failed - using fallback list
)

:: Check if symbol file got anything
set "SYM_COUNT=0"
if exist "%SYMBOL_FILE%" (
  for /f %%C in ('type "%SYMBOL_FILE%" ^| find /c /v ""') do set "SYM_COUNT=%%C"
)

if "%SYM_COUNT%"=="0" (
  echo [train-batch] Writing fallback symbol list...
  >> "%SUMMARY_LOG%" echo [train-batch] Using bundled fallback symbol list
  (
    echo BTCUSDT
    echo ETHUSDT
    echo BNBUSDT
    echo SOLUSDT
    echo XRPUSDT
    echo ADAUSDT
    echo DOGEUSDT
    echo TRXUSDT
    echo AVAXUSDT
    echo LINKUSDT
    echo TONUSDT
    echo SHIBUSDT
    echo DOTUSDT
    echo LTCUSDT
    echo BCHUSDT
    echo NEARUSDT
    echo APTUSDT
    echo ARBUSDT
    echo OPUSDT
    echo ATOMUSDT
    echo INJUSDT
    echo RNDRUSDT
    echo ETCUSDT
    echo XLMUSDT
    echo FILUSDT
    echo SEIUSDT
    echo SUIUSDT
    echo ICPUSDT
    echo GRTUSDT
    echo AAVEUSDT
    echo MKRUSDT
    echo UNIUSDT
    echo PEPEUSDT
    echo FETUSDT
    echo RUNEUSDT
    echo ALGOUSDT
    echo MATICUSDT
    echo HBARUSDT
    echo IMXUSDT
    echo TAOUSDT
    echo STXUSDT
    echo TIAUSDT
    echo ENAUSDT
    echo PENDLEUSDT
    echo THETAUSDT
    echo EGLDUSDT
    echo KASUSDT
    echo JASMYUSDT
    echo CFXUSDT
    echo ARUSDT
    echo WIFUSDT
    echo BONKUSDT
    echo FLOKIUSDT
    echo ORDIUSDT
    echo PYTHUSDT
    echo AXSUSDT
    echo SANDUSDT
    echo MANAUSDT
    echo GALAUSDT
    echo CHZUSDT
    echo CRVUSDT
    echo SNXUSDT
    echo LDOUSDT
    echo DYDXUSDT
    echo YFIUSDT
    echo 1INCHUSDT
    echo KAVAUSDT
    echo COMPUSDT
    echo ZECUSDT
    echo ENSUSDT
    echo KSMUSDT
    echo MINAUSDT
    echo ROSEUSDT
    echo GMTUSDT
    echo APEUSDT
    echo BLURUSDT
    echo AKTUSDT
    echo JTOUSDT
    echo WLDUSDT
    echo XAIUSDT
    echo ONDOUSDT
    echo BEAMUSDT
    echo NOTUSDT
    echo OMUSDT
    echo ZROUSDT
    echo AEVOUSDT
    echo STRKUSDT
  ) > "%SYMBOL_FILE%"
  for /f %%C in ('type "%SYMBOL_FILE%" ^| find /c /v ""') do set "SYM_COUNT=%%C"
)

echo [train-batch] Total symbols to train: %SYM_COUNT%
>> "%SUMMARY_LOG%" echo [train-batch] Total symbols: %SYM_COUNT%

:: ── Build groups (all in 1 group by default since COIN_GROUP_SIZE=90) ─────────
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$syms=Get-Content '%SYMBOL_FILE%' | Where-Object{$_} | Select-Object -Unique;" ^
  "$chunk=[Math]::Max(10,[int]$env:COIN_GROUP_SIZE);" ^
  "Write-Host ('[train-batch] Grouping ' + $syms.Count + ' symbols into chunks of ' + $chunk);" ^
  "$groups=@(); for($i=0;$i -lt $syms.Count;$i+=$chunk){$groups+=(($syms[$i..([Math]::Min($i+$chunk-1,$syms.Count-1))])-join ',')};" ^
  "$groups | Set-Content -Encoding ASCII '%GROUP_FILE%'"
if errorlevel 1 (
  echo [train-batch] ERROR: Failed to build symbol groups.
  set "EXIT_CODE=1"
  goto :finish
)

:: ── Training loop ─────────────────────────────────────────────────────────────
set /a GROUP_INDEX=0
for /f "usebackq delims=" %%G in ("%GROUP_FILE%") do (
  set /a GROUP_INDEX+=1
  set "GROUP_SYMBOLS=%%G"
  set "GROUP_LOG=%LOG_DIR%\group-!GROUP_INDEX!-%RUN_ID%.log"

  echo.
  echo [train-batch] ==================================================
  echo [train-batch]  GROUP !GROUP_INDEX! - Starting training
  echo [train-batch] ==================================================
  echo [train-batch]  Symbols: !GROUP_SYMBOLS!
  echo [train-batch]  Log    : !GROUP_LOG!
  echo [train-batch] ==================================================
  echo.

  >> "%SUMMARY_LOG%" echo [train-batch] GROUP !GROUP_INDEX! START: !GROUP_SYMBOLS!

  :: Pass all args via env vars to train-stream.ps1 (avoids all quoting issues)
  set "TRAIN_PY=!PY!"
  set "TRAIN_SCRIPT=!BACKEND_DIR!train-local.py"
  set "TRAIN_LOG=!GROUP_LOG!"
  set "TRAIN_AWS_REGION=%AWS_REGION%"
  set "TRAIN_DATA_BUCKET=%DATA_BUCKET%"
  set "TRAIN_MODEL_BUCKET=%MODEL_BUCKET%"
  set "TRAIN_SYMBOLS=!GROUP_SYMBOLS!"
  set "TRAIN_TIMEFRAME=%TIMEFRAME%"
  set "TRAIN_EPOCHS=%EPOCHS%"
  set "TRAIN_BATCH_SIZE=%BATCH_SIZE%"
  set "TRAIN_GRAD_ACCUM_STEPS=%GRAD_ACCUM_STEPS%"
  set "TRAIN_LEARNING_RATE=%LEARNING_RATE%"
  set "TRAIN_MAX_ROWS_PER_SYMBOL=%MAX_ROWS_PER_SYMBOL%"
  set "TRAIN_CONTEXT_LENGTH=%CONTEXT_LENGTH%"
  set "TRAIN_HORIZON=%TRAINING_HORIZON%"
  set "TRAIN_MAX_SEQ_LENGTH=%MAX_SEQ_LENGTH%"
  set "TRAIN_SAVE_STEPS=%SAVE_STEPS%"
  set "TRAIN_EVAL_STEPS=%EVAL_STEPS%"
  set "TRAIN_LOGGING_STEPS=%LOGGING_STEPS%"
  set "TRAIN_LORA_R=%LORA_R%"
  set "TRAIN_LORA_ALPHA=%LORA_ALPHA%"
  set "TRAIN_LORA_DROPOUT=%LORA_DROPOUT%"
  set "TRAIN_SPLIT_RATIO=%TRAIN_SPLIT_RATIO%"
  set "TRAIN_WALK_FORWARD_WINDOWS=%WALK_FORWARD_WINDOWS%"
  set "TRAIN_WALK_FORWARD_EVAL_SIZE=%WALK_FORWARD_EVAL_SIZE%"
  set "TRAIN_EXTERNAL_COVARIATE_SCALE=%EXTERNAL_COVARIATE_SCALE%"
  set "TRAIN_PREDICT_VARIANCE_SCALE=%PREDICT_VARIANCE_SCALE%"
  set "TRAIN_PREDICT_DIFFUSION_STEPS=%PREDICT_DIFFUSION_STEPS%"
  set "TRAIN_BASE_MODEL_ID=%BASE_MODEL_ID%"
  set "TRAIN_BASE_MODEL_FALLBACK_ID=%BASE_MODEL_FALLBACK_ID%"
  set "TRAIN_PARQUET_PREFIX=%PARQUET_PREFIX%"
  set "TRAIN_MAX_PARQUET_FILES_PER_SYMBOL=%MAX_PARQUET_FILES_PER_SYMBOL%"
  set "TRAIN_LIVE_FETCH_FLAG=%LIVE_FETCH_FLAG%"
  set "TRAIN_STRICT_FLAG=%STRICT_FLAG%"

  powershell -NoProfile -ExecutionPolicy Bypass -File "%BACKEND_DIR%train-stream.ps1"

  if errorlevel 1 (
    echo.
    echo [train-batch] GROUP !GROUP_INDEX! FAILED with errors. See: !GROUP_LOG!
    >> "%SUMMARY_LOG%" echo [train-batch] GROUP !GROUP_INDEX! FAILED
    set "EXIT_CODE=1"
    goto :finish
  )

  echo.
  echo [train-batch] GROUP !GROUP_INDEX! COMPLETE.
  >> "%SUMMARY_LOG%" echo [train-batch] GROUP !GROUP_INDEX! COMPLETE
)

if !GROUP_INDEX! EQU 0 (
  echo [train-batch] ERROR: No groups were generated. Something went wrong.
  >> "%SUMMARY_LOG%" echo [train-batch] ERROR: No groups generated
  set "EXIT_CODE=1"
)

:finish
if exist "%GROUP_FILE%"  del /q "%GROUP_FILE%"  >nul 2>&1
if exist "%SYMBOL_FILE%" del /q "%SYMBOL_FILE%" >nul 2>&1
if exist "%LOCK_FILE%"   del /q "%LOCK_FILE%"   >nul 2>&1

echo.
echo ============================================================
if "%EXIT_CODE%"=="0" (
  echo   TRAINING COMPLETE - All groups finished successfully!
  >> "%SUMMARY_LOG%" echo [train-batch] TRAINING COMPLETE: %DATE% %TIME%
) else (
  echo   TRAINING FAILED - Check logs above and in:
  echo   %LOG_DIR%
  >> "%SUMMARY_LOG%" echo [train-batch] TRAINING FAILED: %DATE% %TIME%
)
echo   Summary log: %SUMMARY_LOG%
echo ============================================================
echo.
echo Press any key to close this window...
pause >nul
exit /b %EXIT_CODE%

:: ── Fatal: show error and wait ────────────────────────────────────────────────
:fatal
if exist "%LOCK_FILE%" del /q "%LOCK_FILE%" >nul 2>&1
echo.
echo [train-batch] *** FATAL ERROR - see messages above ***
echo.
echo Press any key to close...
pause >nul
exit /b 1

:: ── Subroutines ───────────────────────────────────────────────────────────────
:detect_gpu
set "GPU_VISIBLE=0"
where nvidia-smi >nul 2>&1
if errorlevel 1 exit /b 0
nvidia-smi -L >nul 2>&1
if not errorlevel 1 set "GPU_VISIBLE=1"
exit /b 0

:probe_cuda
set "PY_TORCH_CUDA=0"
set "_TMP=%TEMP%\aether_cuda_%RANDOM%.txt"
"%PY%" -c "import torch,sys; sys.stdout.write('1' if torch.cuda.is_available() else '0')" > "%_TMP%" 2>nul
if exist "%_TMP%" (
  set /p PY_TORCH_CUDA=<"%_TMP%"
  del /q "%_TMP%" >nul 2>&1
)
if not defined PY_TORCH_CUDA set "PY_TORCH_CUDA=0"
exit /b 0
