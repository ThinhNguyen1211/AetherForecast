@echo off
title AetherForecast - Train Batch (Production)
setlocal EnableExtensions EnableDelayedExpansion

set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%..\..") do set "ROOT=%%~fI"
set "BACKEND_DIR=%SCRIPT_DIR%"
cd /d "%ROOT%"

set "PY_CANDIDATE_VENV=%ROOT%\.venv\Scripts\python.exe"
set "PY_CANDIDATE_TRAIN_GPU=%ROOT%\train-gpu\Scripts\python.exe"
set "PY_CANDIDATE_TRAIN_GPU_BACKEND=%ROOT%\train-gpu\backend-code\.venv\Scripts\python.exe"
set "PY="
set "PY_TORCH_CUDA=0"
set "GPU_VISIBLE=0"

call :detect_gpu

call :select_python
if not defined PY (
  echo [train-batch] No project Python found. Bootstrapping .venv automatically...
  call :bootstrap_python
  if errorlevel 1 exit /b 1
  call :select_python
)

if not defined PY (
  echo [train-batch] Python interpreter not found after bootstrap.
  echo [train-batch] Expected one of:
  echo   %ROOT%\.venv\Scripts\python.exe
  echo   %ROOT%\train-gpu\Scripts\python.exe
  exit /b 1
)

call :ensure_runtime_deps
if errorlevel 1 exit /b 1

call :probe_torch_cuda "%PY%"
if "%GPU_VISIBLE%"=="1" if not "%PY_TORCH_CUDA%"=="1" (
  call :ensure_cuda_torch
  if errorlevel 1 exit /b 1
  call :probe_torch_cuda "%PY%"
)

if /I "%DATA_BUCKET%"=="dummy-bucket" set "DATA_BUCKET="
if /I "%DATA_S3_BUCKET%"=="dummy-bucket" set "DATA_S3_BUCKET="
if /I "%MODEL_BUCKET%"=="dummy-model" set "MODEL_BUCKET="

if not defined AWS_REGION set "AWS_REGION=ap-southeast-1"
if not defined DATA_BUCKET set "DATA_BUCKET=aetherforecast-data-800762439372-ap-southeast-1"
if not defined DATA_S3_BUCKET set "DATA_S3_BUCKET=%DATA_BUCKET%"
if not defined MODEL_BUCKET set "MODEL_BUCKET=aetherforecast-models-800762439372-ap-southeast-1"
if not defined PARQUET_PREFIX set "PARQUET_PREFIX=market/klines"

if not defined TIMEFRAME set "TIMEFRAME=1h,4h,1d"
if not defined CONTEXT_LENGTH set "CONTEXT_LENGTH=1024"
if not defined MAX_SEQ_LENGTH set "MAX_SEQ_LENGTH=1024"
if not defined BATCH_SIZE set "BATCH_SIZE=2"
if not defined LORA_R set "LORA_R=16"
if not defined LORA_ALPHA set "LORA_ALPHA=32"
if not defined LORA_DROPOUT set "LORA_DROPOUT=0.05"
if not defined SAVE_STEPS set "SAVE_STEPS=50"
if not defined EVAL_STEPS set "EVAL_STEPS=50"
if not defined LOGGING_STEPS set "LOGGING_STEPS=5"
if not defined TRAIN_SPLIT_RATIO set "TRAIN_SPLIT_RATIO=0.95"

if not defined EPOCHS set "EPOCHS=5"
if not defined GRAD_ACCUM_STEPS set "GRAD_ACCUM_STEPS=8"
if not defined LEARNING_RATE set "LEARNING_RATE=0.0002"
if not defined MAX_ROWS_PER_SYMBOL set "MAX_ROWS_PER_SYMBOL=12000"
if not defined TRAINING_HORIZON set "TRAINING_HORIZON=7"
if not defined MAX_PARQUET_FILES_PER_SYMBOL set "MAX_PARQUET_FILES_PER_SYMBOL=320"
if not defined PARQUET_KEY_DEDUP_BY_DAY set "PARQUET_KEY_DEDUP_BY_DAY=0"

if not defined WALK_FORWARD_WINDOWS set "WALK_FORWARD_WINDOWS=5"
if not defined WALK_FORWARD_EVAL_SIZE set "WALK_FORWARD_EVAL_SIZE=128"
if not defined EXTERNAL_COVARIATE_SCALE set "EXTERNAL_COVARIATE_SCALE=0.0018"
if not defined ENABLE_EXTERNAL_FETCH set "ENABLE_EXTERNAL_FETCH=1"
if not defined STRICT_EXTERNAL_DATA set "STRICT_EXTERNAL_DATA=0"
if not defined PREDICT_VARIANCE_SCALE set "PREDICT_VARIANCE_SCALE=1.18"
if not defined PREDICT_DIFFUSION_STEPS set "PREDICT_DIFFUSION_STEPS=3"
if not defined REQUIRE_CUDA set "REQUIRE_CUDA=0"

if not defined BASE_MODEL_ID set "BASE_MODEL_ID=amazon/chronos-2"
if not defined BASE_MODEL_FALLBACK_ID set "BASE_MODEL_FALLBACK_ID=amazon/chronos-t5-large"

if not defined COIN_GROUP_SIZE set "COIN_GROUP_SIZE=90"
if not defined ALL_SYMBOLS set "ALL_SYMBOLS=AUTO"
set "ALL_SYMBOLS_FALLBACK=BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT,XRPUSDT,ADAUSDT,DOGEUSDT,TRXUSDT,AVAXUSDT,LINKUSDT,TONUSDT,SHIBUSDT,DOTUSDT,LTCUSDT,BCHUSDT,NEARUSDT,APTUSDT,ARBUSDT,OPUSDT,ATOMUSDT,INJUSDT,RNDRUSDT,ETCUSDT,XLMUSDT,FILUSDT,SEIUSDT,SUIUSDT,ICPUSDT,GRTUSDT,AAVEUSDT,MKRUSDT,UNIUSDT,PEPEUSDT,FETUSDT,RUNEUSDT,ALGOUSDT,MATICUSDT,HBARUSDT,IMXUSDT,TAOUSDT,STXUSDT,TIAUSDT,ENAUSDT,PENDLEUSDT,THETAUSDT,EGLDUSDT,KASUSDT,JASMYUSDT,CFXUSDT,ARUSDT,WIFUSDT,BONKUSDT,FLOKIUSDT,ORDIUSDT,PYTHUSDT,AXSUSDT,SANDUSDT,MANAUSDT,GALAUSDT,CHZUSDT,CRVUSDT,SNXUSDT,LDOUSDT,DYDXUSDT,YFIUSDT,1INCHUSDT,KAVAUSDT,COMPUSDT,ZECUSDT,ENSUSDT,KSMUSDT,MINAUSDT,ROSEUSDT,GMTUSDT,APEUSDT,BLURUSDT,AKTUSDT,JTOUSDT,WLDUSDT,XAIUSDT,ONDOUSDT,BEAMUSDT,NOTUSDT,OMUSDT,ZROUSDT,AEVOUSDT,STRKUSDT"

set "LOG_DIR=%ROOT%\artifacts\train-batch-logs"
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

for /f %%I in ('powershell -NoProfile -Command "(Get-Date).ToString('yyyyMMdd-HHmmss')"') do set "RUN_ID=%%I"
if not defined RUN_ID set "RUN_ID=manual"

set "SUMMARY_LOG=%LOG_DIR%\train-batch-%RUN_ID%.log"
set "SYMBOL_FILE=%LOG_DIR%\train-batch-symbols-%RUN_ID%.txt"
set "GROUP_FILE=%LOG_DIR%\train-batch-groups-%RUN_ID%.txt"
set "LOCK_FILE=%LOG_DIR%\train-batch.lock"
set "EXIT_CODE=0"

if exist "%LOCK_FILE%" (
  call :has_active_train_local
  if "!HAS_ACTIVE_TRAIN_LOCAL!"=="1" (
    echo [train-batch] Another train-batch run is still active.
    echo [train-batch] Lock file: %LOCK_FILE%
    exit /b 1
  )
  echo [train-batch] Stale lock detected. Removing %LOCK_FILE% ...
  del /q "%LOCK_FILE%" >nul 2>&1
)

> "%LOCK_FILE%" echo started=%DATE% %TIME%

if "%REQUIRE_CUDA%"=="1" if not "%PY_TORCH_CUDA%"=="1" (
  call :log [train-batch] REQUIRE_CUDA=1 but selected Python has no CUDA-enabled torch.
  call :log [train-batch] Python=%PY%
  set "EXIT_CODE=1"
  goto :finish
)

call :log [train-batch] ============================================================
call :log [train-batch]   AetherForecast - Production Training Pipeline
call :log [train-batch] ============================================================
call :log [train-batch] Python=%PY%
call :log [train-batch] GPU_VISIBLE=%GPU_VISIBLE% TORCH_CUDA=%PY_TORCH_CUDA%
if "%PY_TORCH_CUDA%"=="1" (
  call :log [train-batch] Runtime mode: GPU ^(CUDA^)
) else (
  call :log [train-batch] Runtime mode: CPU (CUDA not available in selected Python)
)
call :log [train-batch] AWS_REGION=%AWS_REGION%
call :log [train-batch] DATA_BUCKET=%DATA_BUCKET%
call :log [train-batch] MODEL_BUCKET=%MODEL_BUCKET%
call :log [train-batch] TIMEFRAME=%TIMEFRAME%
call :log [train-batch] EPOCHS=%EPOCHS%  LEARNING_RATE=%LEARNING_RATE%  GRAD_ACCUM=%GRAD_ACCUM_STEPS%
call :log [train-batch] CONTEXT_LENGTH=%CONTEXT_LENGTH%  MAX_SEQ_LENGTH=%MAX_SEQ_LENGTH%
call :log [train-batch] BATCH_SIZE=%BATCH_SIZE%  LORA_R=%LORA_R%  LORA_ALPHA=%LORA_ALPHA%  LORA_DROPOUT=%LORA_DROPOUT%
call :log [train-batch] SAVE_STEPS=%SAVE_STEPS% EVAL_STEPS=%EVAL_STEPS% LOGGING_STEPS=%LOGGING_STEPS%
call :log [train-batch] TRAIN_SPLIT_RATIO=%TRAIN_SPLIT_RATIO%
call :log [train-batch] WALK_FORWARD_WINDOWS=%WALK_FORWARD_WINDOWS%  WALK_FORWARD_EVAL_SIZE=%WALK_FORWARD_EVAL_SIZE%
call :log [train-batch] MAX_ROWS_PER_SYMBOL=%MAX_ROWS_PER_SYMBOL%  MAX_PARQUET_FILES=%MAX_PARQUET_FILES_PER_SYMBOL%
call :log [train-batch] COIN_GROUP_SIZE=%COIN_GROUP_SIZE%
call :log [train-batch] BASE_MODEL=%BASE_MODEL_ID%  FALLBACK=%BASE_MODEL_FALLBACK_ID%
call :log [train-batch] EXTERNAL_FETCH=%ENABLE_EXTERNAL_FETCH%  STRICT=%STRICT_EXTERNAL_DATA%
call :log [train-batch] Run started at %DATE% %TIME%

call :resolve_symbols
if errorlevel 1 (
  call :log [train-batch] Failed to resolve symbol list.
  set "EXIT_CODE=1"
  goto :finish
)

powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$ErrorActionPreference='Stop';" ^
  "$chunk=[Math]::Max(15,[Math]::Min(200,[int]$env:COIN_GROUP_SIZE));" ^
  "$symbols=@(Get-Content '%SYMBOL_FILE%' | ForEach-Object { $_.Trim().ToUpper() } | Where-Object { $_ -ne '' } | Select-Object -Unique);" ^
  "if($symbols.Count -eq 0){ throw 'Symbol file is empty'; }" ^
  "Write-Host ('[train-batch] Total symbols: ' + $symbols.Count + ', group size: ' + $chunk);" ^
  "$groups=New-Object System.Collections.Generic.List[string];" ^
  "for($i=0; $i -lt $symbols.Count; $i += $chunk){$end=[Math]::Min($i + $chunk - 1, $symbols.Count - 1); $groups.Add((@($symbols[$i..$end]) -join ',')); }" ^
  "$groups | Set-Content -Encoding ASCII '%GROUP_FILE%'"
if errorlevel 1 (
  call :log [train-batch] Failed to build symbol groups from ALL_SYMBOLS.
  set "EXIT_CODE=1"
  goto :finish
)

set "LIVE_FETCH_FLAG=--enable-live-external-fetch"
if /I "%ENABLE_EXTERNAL_FETCH%"=="0" set "LIVE_FETCH_FLAG=--disable-live-external-fetch"

set "STRICT_FLAG=--allow-external-fallback"
if /I "%STRICT_EXTERNAL_DATA%"=="1" set "STRICT_FLAG=--strict-external-data"

set /a GROUP_INDEX=0
for /f "usebackq delims=" %%G in ("%GROUP_FILE%") do (
  set /a GROUP_INDEX+=1
  set "GROUP_SYMBOLS=%%G"
  set "GROUP_LOG=%LOG_DIR%\group-!GROUP_INDEX!-%RUN_ID%.log"

  call :log [train-batch] ---- GROUP !GROUP_INDEX! START ----
  call :log [train-batch] Symbols=!GROUP_SYMBOLS!
  call :log [train-batch] Log=!GROUP_LOG!

  echo [train-batch] GROUP !GROUP_INDEX! START %DATE% %TIME%> "!GROUP_LOG!"
  echo [train-batch] Symbols=!GROUP_SYMBOLS!>> "!GROUP_LOG!"

  "%PY%" "%BACKEND_DIR%train-local.py" ^
    --aws-region %AWS_REGION% ^
    --data-bucket %DATA_BUCKET% ^
    --model-bucket %MODEL_BUCKET% ^
    --symbols !GROUP_SYMBOLS! ^
    --timeframe %TIMEFRAME% ^
    --epochs %EPOCHS% ^
    --batch-size %BATCH_SIZE% ^
    --grad-accum-steps %GRAD_ACCUM_STEPS% ^
    --learning-rate %LEARNING_RATE% ^
    --max-rows-per-symbol %MAX_ROWS_PER_SYMBOL% ^
    --context-length %CONTEXT_LENGTH% ^
    --horizon %TRAINING_HORIZON% ^
    --max-seq-length %MAX_SEQ_LENGTH% ^
    --save-steps %SAVE_STEPS% ^
    --eval-steps %EVAL_STEPS% ^
    --logging-steps %LOGGING_STEPS% ^
    --lora-r %LORA_R% ^
    --lora-alpha %LORA_ALPHA% ^
    --lora-dropout %LORA_DROPOUT% ^
    --train-split-ratio %TRAIN_SPLIT_RATIO% ^
    --walk-forward-windows %WALK_FORWARD_WINDOWS% ^
    --walk-forward-eval-size %WALK_FORWARD_EVAL_SIZE% ^
    --external-covariate-scale %EXTERNAL_COVARIATE_SCALE% ^
    --predict-variance-scale %PREDICT_VARIANCE_SCALE% ^
    --predict-diffusion-steps %PREDICT_DIFFUSION_STEPS% ^
    --base-model-id %BASE_MODEL_ID% ^
    --base-model-fallback-id %BASE_MODEL_FALLBACK_ID% ^
    --parquet-prefix %PARQUET_PREFIX% ^
    --max-parquet-files-per-symbol %MAX_PARQUET_FILES_PER_SYMBOL% ^
    !LIVE_FETCH_FLAG! ^
    !STRICT_FLAG! >> "!GROUP_LOG!" 2>&1

  if errorlevel 1 (
    call :log [train-batch] GROUP !GROUP_INDEX! FAILED. See !GROUP_LOG!
    set "EXIT_CODE=1"
    goto :finish
  )

  call :log [train-batch] GROUP !GROUP_INDEX! DONE.
)

if !GROUP_INDEX! EQU 0 (
  call :log [train-batch] No symbol group generated. Check ALL_SYMBOLS.
  set "EXIT_CODE=1"
  goto :finish
)

call :log [train-batch] All groups completed successfully.

:finish
if exist "%GROUP_FILE%" del /q "%GROUP_FILE%" >nul 2>&1
if exist "%SYMBOL_FILE%" del /q "%SYMBOL_FILE%" >nul 2>&1
if exist "%LOCK_FILE%" del /q "%LOCK_FILE%" >nul 2>&1

if "%EXIT_CODE%"=="0" (
  call :log [train-batch] ============================================================
  call :log [train-batch]   TRAINING COMPLETE - All groups finished successfully
  call :log [train-batch] ============================================================
  call :log [train-batch] Run finished at %DATE% %TIME%
) else (
  call :log [train-batch] ============================================================
  call :log [train-batch]   TRAINING FAILED - Check logs for details
  call :log [train-batch] ============================================================
  call :log [train-batch] Run failed at %DATE% %TIME%
)
call :log [train-batch] Summary log: %SUMMARY_LOG%
call :log [train-batch] Tip: press Ctrl+C during a running group to stop.
echo.
echo ============================================================
echo   Training session ended. Review logs above.
echo   Summary log: %SUMMARY_LOG%
echo ============================================================
pause
exit /b %EXIT_CODE%

:log
echo %*
>> "%SUMMARY_LOG%" echo %*
exit /b 0

:select_python
set "PY="
set "PY_FALLBACK="
for %%P in ("%PY_CANDIDATE_TRAIN_GPU%" "%PY_CANDIDATE_TRAIN_GPU_BACKEND%" "%PY_CANDIDATE_VENV%") do (
  if exist "%%~fP" (
    if not defined PY_FALLBACK set "PY_FALLBACK=%%~fP"
    call :probe_torch_cuda "%%~fP"
    if "!PY_TORCH_CUDA!"=="1" (
      set "PY=%%~fP"
      goto :select_python_done
    )
  )
)

if not defined PY set "PY=%PY_FALLBACK%"
:select_python_done
exit /b 0

:detect_gpu
set "GPU_VISIBLE=0"
where nvidia-smi >nul 2>&1
if errorlevel 1 exit /b 0
nvidia-smi -L >nul 2>&1
if not errorlevel 1 set "GPU_VISIBLE=1"
exit /b 0

:probe_torch_cuda
set "PY_TORCH_CUDA=0"
set "CUDA_PROBE_FILE=%TEMP%\aetherforecast-cuda-%RANDOM%-%RANDOM%.txt"
"%~1" -c "import torch,sys;sys.stdout.write(str(int(torch.cuda.is_available())))" > "%CUDA_PROBE_FILE%" 2>nul
if exist "%CUDA_PROBE_FILE%" (
  set /p PY_TORCH_CUDA=<"%CUDA_PROBE_FILE%"
  del /q "%CUDA_PROBE_FILE%" >nul 2>&1
)
if not defined PY_TORCH_CUDA set "PY_TORCH_CUDA=0"
exit /b 0

:bootstrap_python
set "SYS_PY="
where py >nul 2>&1
if not errorlevel 1 set "SYS_PY=py -3"
if not defined SYS_PY (
  where python >nul 2>&1
  if not errorlevel 1 set "SYS_PY=python"
)

if not defined SYS_PY (
  echo [train-batch] Could not find system Python (py or python) to create .venv.
  exit /b 1
)

if not exist "%PY_CANDIDATE_VENV%" (
  %SYS_PY% -m venv "%ROOT%\.venv"
  if errorlevel 1 (
    echo [train-batch] Failed to create virtual environment at %ROOT%\.venv
    exit /b 1
  )
)

"%PY_CANDIDATE_VENV%" -m pip install --disable-pip-version-check --upgrade pip >nul 2>&1
exit /b 0

:ensure_runtime_deps
"%PY%" -c "import boto3,pandas,numpy,torch,transformers,pydantic,pydantic_settings,chronos" >nul 2>&1
if errorlevel 1 (
  echo [train-batch] Installing core training dependencies ...
  "%PY%" -m pip install --disable-pip-version-check ^
    boto3==1.40.1 ^
    botocore==1.40.1 ^
    pandas==2.3.2 ^
    numpy==2.2.6 ^
    polars==1.33.0 ^
    pyarrow==20.0.0 ^
    datasets==4.0.0 ^
    peft==0.17.1 ^
    transformers==4.55.4 ^
    accelerate==1.10.0 ^
    safetensors==0.6.2 ^
    sentencepiece==0.2.1 ^
    scipy==1.16.1 ^
    huggingface-hub==0.34.4 ^
    pydantic==2.11.7 ^
    pydantic-settings==2.11.0 ^
    python-dotenv==1.2.1 ^
    structlog==24.4.0 ^
    requests==2.32.4 ^
    chronos-forecasting==2.2.2
  if errorlevel 1 (
    echo [train-batch] Failed to install Python dependencies.
    exit /b 1
  )
)
exit /b 0

:ensure_cuda_torch
echo [train-batch] GPU detected but CUDA-enabled torch is unavailable in selected Python.
echo [train-batch] Attempting to install CUDA torch wheels (cu128) ...
"%PY%" -m pip install --disable-pip-version-check --index-url https://download.pytorch.org/whl/cu128 torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0
if errorlevel 1 (
  echo [train-batch] Failed to install CUDA torch wheels.
  if "%REQUIRE_CUDA%"=="1" exit /b 1
)
exit /b 0

:has_active_train_local
set "HAS_ACTIVE_TRAIN_LOCAL=0"
for /f %%I in ('powershell -NoProfile -ExecutionPolicy Bypass -Command "$p=Get-CimInstance Win32_Process ^| Where-Object { $_.Name -eq 'python.exe' -and $_.CommandLine -like '*train-local.py*' }; if($p){'1'}else{'0'}"') do set "HAS_ACTIVE_TRAIN_LOCAL=%%I"
if not defined HAS_ACTIVE_TRAIN_LOCAL set "HAS_ACTIVE_TRAIN_LOCAL=0"
exit /b 0

:resolve_symbols
set "SYMBOL_SOURCE=%ALL_SYMBOLS%"
if /I "%SYMBOL_SOURCE%"=="AUTO" set "SYMBOL_SOURCE="
if /I "%SYMBOL_SOURCE%"=="ALL" set "SYMBOL_SOURCE="
if /I "%SYMBOL_SOURCE%"=="*" set "SYMBOL_SOURCE="

if exist "%SYMBOL_FILE%" del /q "%SYMBOL_FILE%" >nul 2>&1

if defined SYMBOL_SOURCE (
  powershell -NoProfile -ExecutionPolicy Bypass -Command ^
    "$ErrorActionPreference='Stop';" ^
    "$symbols=@($env:SYMBOL_SOURCE -split ',' | ForEach-Object { $_.Trim().ToUpper() } | Where-Object { $_ -ne '' } | Select-Object -Unique);" ^
    "if($symbols.Count -eq 0){ throw 'ALL_SYMBOLS is empty'; }" ^
    "$symbols | Set-Content -Encoding ASCII '%SYMBOL_FILE%'"
  exit /b 0
)

call :log [train-batch] ALL_SYMBOLS=ALL/AUTO. Resolving symbols from S3 parquet prefixes...
"%PY%" -c "import os,sys,boto3; bucket=os.getenv('DATA_BUCKET') or os.getenv('DATA_S3_BUCKET'); prefix=os.getenv('PARQUET_PREFIX','market/klines').strip('/'); region=os.getenv('AWS_REGION','ap-southeast-1');
if not bucket: sys.exit(2)
s3=boto3.client('s3', region_name=region); paginator=s3.get_paginator('list_objects_v2'); symbols=set();
for page in paginator.paginate(Bucket=bucket, Prefix=prefix + '/', Delimiter='/'):
    for item in page.get('CommonPrefixes', []):
        path=str(item.get('Prefix',''))
        if 'symbol=' in path:
            symbol=path.split('symbol=',1)[1].split('/',1)[0].strip().upper()
            if symbol: symbols.add(symbol)
if not symbols: sys.exit(3)
sys.stdout.write('\n'.join(sorted(symbols)))" > "%SYMBOL_FILE%" 2>nul
if errorlevel 1 goto :resolve_symbols_fallback

powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$count=(Get-Content '%SYMBOL_FILE%' | Where-Object { $_ -ne '' }).Count; if($count -lt 1){ exit 1 }" >nul 2>&1
if errorlevel 1 goto :resolve_symbols_fallback

call :log [train-batch] Loaded symbols from S3 into %SYMBOL_FILE%
exit /b 0

:resolve_symbols_fallback
call :log [train-batch] Failed to load symbols from S3. Falling back to bundled list.
set "SYMBOL_SOURCE=%ALL_SYMBOLS_FALLBACK%"
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$ErrorActionPreference='Stop';" ^
  "$symbols=@($env:SYMBOL_SOURCE -split ',' | ForEach-Object { $_.Trim().ToUpper() } | Where-Object { $_ -ne '' } | Select-Object -Unique);" ^
  "if($symbols.Count -eq 0){ throw 'Fallback symbol list is empty'; }" ^
  "$symbols | Set-Content -Encoding ASCII '%SYMBOL_FILE%'"
exit /b 0
