@echo off
setlocal EnableExtensions EnableDelayedExpansion

set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%..\..") do set "ROOT=%%~fI"
set "BACKEND_DIR=%SCRIPT_DIR%"
cd /d "%ROOT%"

set "PY_CANDIDATE_VENV=%ROOT%\.venv\Scripts\python.exe"
set "PY_CANDIDATE_TRAIN_GPU=%ROOT%\train-gpu\Scripts\python.exe"
set "PY="

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

if not defined EPOCHS set "EPOCHS=2"
if not defined GRAD_ACCUM_STEPS set "GRAD_ACCUM_STEPS=8"
if not defined LEARNING_RATE set "LEARNING_RATE=0.0002"
if not defined MAX_ROWS_PER_SYMBOL set "MAX_ROWS_PER_SYMBOL=8000"
if not defined TRAINING_HORIZON set "TRAINING_HORIZON=7"
if not defined MAX_PARQUET_FILES_PER_SYMBOL set "MAX_PARQUET_FILES_PER_SYMBOL=320"
if not defined PARQUET_KEY_DEDUP_BY_DAY set "PARQUET_KEY_DEDUP_BY_DAY=0"

if not defined WALK_FORWARD_WINDOWS set "WALK_FORWARD_WINDOWS=4"
if not defined WALK_FORWARD_EVAL_SIZE set "WALK_FORWARD_EVAL_SIZE=128"
if not defined EXTERNAL_COVARIATE_SCALE set "EXTERNAL_COVARIATE_SCALE=0.0018"
if not defined ENABLE_EXTERNAL_FETCH set "ENABLE_EXTERNAL_FETCH=1"
if not defined STRICT_EXTERNAL_DATA set "STRICT_EXTERNAL_DATA=0"
if not defined PREDICT_VARIANCE_SCALE set "PREDICT_VARIANCE_SCALE=1.18"
if not defined PREDICT_DIFFUSION_STEPS set "PREDICT_DIFFUSION_STEPS=3"

if not defined BASE_MODEL_ID set "BASE_MODEL_ID=amazon/chronos-2"
if not defined BASE_MODEL_FALLBACK_ID set "BASE_MODEL_FALLBACK_ID=amazon/chronos-t5-large"

if not defined COIN_GROUP_SIZE set "COIN_GROUP_SIZE=18"
if not defined ALL_SYMBOLS set "ALL_SYMBOLS=BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT,XRPUSDT,ADAUSDT,DOGEUSDT,TRXUSDT,AVAXUSDT,LINKUSDT,TONUSDT,SHIBUSDT,DOTUSDT,LTCUSDT,BCHUSDT,NEARUSDT,APTUSDT,ARBUSDT,OPUSDT,ATOMUSDT,INJUSDT,RNDRUSDT,ETCUSDT,XLMUSDT,FILUSDT,SEIUSDT,SUIUSDT,ICPUSDT,GRTUSDT,AAVEUSDT,MKRUSDT,UNIUSDT,PEPEUSDT,FETUSDT,RUNEUSDT,ALGOUSDT,MATICUSDT,HBARUSDT,IMXUSDT,TAOUSDT,STXUSDT,TIAUSDT,ENAUSDT,PENDLEUSDT,THETAUSDT,EGLDUSDT,KASUSDT,JASMYUSDT,CFXUSDT,ARUSDT,WIFUSDT,BONKUSDT,FLOKIUSDT,ORDIUSDT,PYTHUSDT,AXSUSDT,SANDUSDT,MANAUSDT,GALAUSDT,CHZUSDT,CRVUSDT,SNXUSDT,LDOUSDT,DYDXUSDT,YFIUSDT,1INCHUSDT,KAVAUSDT,COMPUSDT,ZECUSDT,ENSUSDT,KSMUSDT,MINAUSDT,ROSEUSDT,GMTUSDT,APEUSDT,BLURUSDT,AKTUSDT,JTOUSDT,WLDUSDT,XAIUSDT,ONDOUSDT,BEAMUSDT,NOTUSDT,OMUSDT,ZROUSDT,AEVOUSDT,STRKUSDT"

set "LOG_DIR=%ROOT%\artifacts\train-batch-logs"
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

for /f %%I in ('powershell -NoProfile -Command "(Get-Date).ToString('yyyyMMdd-HHmmss')"') do set "RUN_ID=%%I"
if not defined RUN_ID set "RUN_ID=manual"

set "SUMMARY_LOG=%LOG_DIR%\train-batch-%RUN_ID%.log"
set "GROUP_FILE=%LOG_DIR%\train-batch-groups-%RUN_ID%.txt"
set "LOCK_FILE=%LOG_DIR%\train-batch.lock"
set "EXIT_CODE=0"

if exist "%LOCK_FILE%" (
  echo [train-batch] Another train-batch run appears to be active.
  echo [train-batch] Lock file: %LOCK_FILE%
  echo [train-batch] If no train-local.py process is running, delete this lock file and run again.
  exit /b 1
)

> "%LOCK_FILE%" echo started=%DATE% %TIME%

call :log [train-batch] ============================================================
call :log [train-batch] Python=%PY%
call :log [train-batch] AWS_REGION=%AWS_REGION%
call :log [train-batch] DATA_BUCKET=%DATA_BUCKET%
call :log [train-batch] MODEL_BUCKET=%MODEL_BUCKET%
call :log [train-batch] TIMEFRAME=%TIMEFRAME%
call :log [train-batch] CONTEXT_LENGTH=%CONTEXT_LENGTH%
call :log [train-batch] BATCH_SIZE=%BATCH_SIZE% LORA_R=%LORA_R%
call :log [train-batch] COIN_GROUP_SIZE=%COIN_GROUP_SIZE% (auto-clamped to 15..20)
call :log [train-batch] Run started at %DATE% %TIME%

powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$ErrorActionPreference='Stop';" ^
  "$chunk=[Math]::Max(15,[Math]::Min(20,[int]$env:COIN_GROUP_SIZE));" ^
  "$symbols=@($env:ALL_SYMBOLS -split ',' | ForEach-Object { $_.Trim().ToUpper() } | Where-Object { $_ -ne '' } | Select-Object -Unique);" ^
  "if($symbols.Count -eq 0){ throw 'ALL_SYMBOLS is empty'; }" ^
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
    --lora-r %LORA_R% ^
    --lora-alpha %LORA_ALPHA% ^
    --lora-dropout %LORA_DROPOUT% ^
    --walk-forward-windows %WALK_FORWARD_WINDOWS% ^
    --walk-forward-eval-size %WALK_FORWARD_EVAL_SIZE% ^
    --external-covariate-scale %EXTERNAL_COVARIATE_SCALE% ^
    --predict-variance-scale %PREDICT_VARIANCE_SCALE% ^
    --predict-diffusion-steps %PREDICT_DIFFUSION_STEPS% ^
    --base-model-id %BASE_MODEL_ID% ^
    --base-model-fallback-id %BASE_MODEL_FALLBACK_ID% ^
    --parquet-prefix %PARQUET_PREFIX% ^
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
if exist "%LOCK_FILE%" del /q "%LOCK_FILE%" >nul 2>&1

if "%EXIT_CODE%"=="0" (
  call :log [train-batch] Run finished at %DATE% %TIME%
) else (
  call :log [train-batch] Run failed at %DATE% %TIME%
)
call :log [train-batch] Summary log: %SUMMARY_LOG%
call :log [train-batch] Tip: press Ctrl+C during a running group to stop.
exit /b %EXIT_CODE%

:log
echo %*
>> "%SUMMARY_LOG%" echo %*
exit /b 0

:select_python
set "PY="
if exist "%PY_CANDIDATE_VENV%" set "PY=%PY_CANDIDATE_VENV%"
if not defined PY if exist "%PY_CANDIDATE_TRAIN_GPU%" set "PY=%PY_CANDIDATE_TRAIN_GPU%"
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
"%PY%" -c "import boto3,pandas,numpy,torch,transformers" >nul 2>&1
if errorlevel 1 (
  echo [train-batch] Installing Python dependencies from requirements.txt ...
  "%PY%" -m pip install --disable-pip-version-check -r "%BACKEND_DIR%requirements.txt"
  if errorlevel 1 (
    echo [train-batch] Failed to install Python dependencies.
    exit /b 1
  )
)
exit /b 0
