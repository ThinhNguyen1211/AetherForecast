# train-stream.ps1
# Launches train-local.py with real-time output streaming to terminal + log file.
# Uses cmd /c pipe with Tee-Object. $LASTEXITCODE is properly set by PowerShell.
# Called by train-batch.bat via TRAIN_* environment variables.
param()
$ErrorActionPreference = "Stop"

$py      = $env:TRAIN_PY
$script  = $env:TRAIN_SCRIPT
$logFile = $env:TRAIN_LOG

if (-not $py -or -not (Test-Path $py)) {
    Write-Error "TRAIN_PY not set or not found: '$py'"
    exit 1
}
if (-not $script -or -not (Test-Path $script)) {
    Write-Error "TRAIN_SCRIPT not set or not found: '$script'"
    exit 1
}
if (-not $logFile) {
    Write-Error "TRAIN_LOG not set"
    exit 1
}

# Ensure log directory exists
$logDir = Split-Path $logFile -Parent
if ($logDir -and -not (Test-Path $logDir)) {
    New-Item -ItemType Directory -Path $logDir -Force | Out-Null
}

# Build argument list
$argList = [System.Collections.Generic.List[string]]::new()
$argList.Add("`"$script`"")

$pairs = @(
    "--aws-region",                   $env:TRAIN_AWS_REGION,
    "--data-bucket",                  $env:TRAIN_DATA_BUCKET,
    "--model-bucket",                 $env:TRAIN_MODEL_BUCKET,
    "--symbols",                      $env:TRAIN_SYMBOLS,
    "--timeframe",                    $env:TRAIN_TIMEFRAME,
    "--epochs",                       $env:TRAIN_EPOCHS,
    "--batch-size",                   $env:TRAIN_BATCH_SIZE,
    "--grad-accum-steps",             $env:TRAIN_GRAD_ACCUM_STEPS,
    "--learning-rate",                $env:TRAIN_LEARNING_RATE,
    "--max-rows-per-symbol",          $env:TRAIN_MAX_ROWS_PER_SYMBOL,
    "--context-length",               $env:TRAIN_CONTEXT_LENGTH,
    "--horizon",                      $env:TRAIN_HORIZON,
    "--max-seq-length",               $env:TRAIN_MAX_SEQ_LENGTH,
    "--save-steps",                   $env:TRAIN_SAVE_STEPS,
    "--eval-steps",                   $env:TRAIN_EVAL_STEPS,
    "--logging-steps",                $env:TRAIN_LOGGING_STEPS,
    "--lora-r",                       $env:TRAIN_LORA_R,
    "--lora-alpha",                   $env:TRAIN_LORA_ALPHA,
    "--lora-dropout",                 $env:TRAIN_LORA_DROPOUT,
    "--train-split-ratio",            $env:TRAIN_SPLIT_RATIO,
    "--walk-forward-windows",         $env:TRAIN_WALK_FORWARD_WINDOWS,
    "--walk-forward-eval-size",       $env:TRAIN_WALK_FORWARD_EVAL_SIZE,
    "--external-covariate-scale",     $env:TRAIN_EXTERNAL_COVARIATE_SCALE,
    "--predict-variance-scale",       $env:TRAIN_PREDICT_VARIANCE_SCALE,
    "--predict-diffusion-steps",      $env:TRAIN_PREDICT_DIFFUSION_STEPS,
    "--base-model-id",                $env:TRAIN_BASE_MODEL_ID,
    "--base-model-fallback-id",       $env:TRAIN_BASE_MODEL_FALLBACK_ID,
    "--parquet-prefix",               $env:TRAIN_PARQUET_PREFIX,
    "--max-parquet-files-per-symbol", $env:TRAIN_MAX_PARQUET_FILES_PER_SYMBOL
)
for ($i = 0; $i -lt $pairs.Count; $i += 2) {
    $flag  = $pairs[$i]
    $value = $pairs[$i + 1]
    if ($null -ne $value -and $value -ne '') {
        $argList.Add($flag)
        $argList.Add("`"$value`"")
    }
}
if ($env:TRAIN_LIVE_FETCH_FLAG) { $argList.Add($env:TRAIN_LIVE_FETCH_FLAG) }
if ($env:TRAIN_STRICT_FLAG)     { $argList.Add($env:TRAIN_STRICT_FLAG) }

$argString = $argList -join ' '

Write-Host ""
Write-Host "[train-stream] ============================================================"
Write-Host "[train-stream] Python  : $py"
Write-Host "[train-stream] Script  : $script"
Write-Host "[train-stream] Log     : $logFile"
Write-Host "[train-stream] Symbols : $($env:TRAIN_SYMBOLS)"
Write-Host "[train-stream] ============================================================"
Write-Host ""

# Set PYTHONUNBUFFERED so Python flushes each line immediately
$env:PYTHONUNBUFFERED = "1"
$env:PYTHONDONTWRITEBYTECODE = "1"

# Open log file writer (UTF-8, append)
$sw = [System.IO.StreamWriter]::new($logFile, $true, [System.Text.Encoding]::UTF8)
$sw.AutoFlush = $true

# Filter that tees: write to console (passthrough) AND to log file
filter Tee-Log {
    Write-Host $_
    $sw.WriteLine($_)
}

# Run Python via cmd /c, merge stderr into stdout (2>&1), pipe to our tee filter
# $LASTEXITCODE after the pipeline reflects cmd.exe exit code = Python exit code
$cmdLine = "`"$py`" $argString"
cmd /c "$cmdLine 2>&1" | Tee-Log

$exitCode = $LASTEXITCODE
$sw.Close()

Write-Host ""
Write-Host "[train-stream] Process finished with exit code: $exitCode"
exit $exitCode
