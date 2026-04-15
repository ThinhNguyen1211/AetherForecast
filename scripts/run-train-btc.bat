@echo off
setlocal EnableExtensions

set SYMBOLS=BTCUSDT
if not defined TIMEFRAME set TIMEFRAME=all

if /I "%DATA_BUCKET%"=="dummy-bucket" (
	echo [run-train-btc] Ignoring test DATA_BUCKET=dummy-bucket
	set DATA_BUCKET=
)

if /I "%DATA_S3_BUCKET%"=="dummy-bucket" (
	echo [run-train-btc] Ignoring test DATA_S3_BUCKET=dummy-bucket
	set DATA_S3_BUCKET=
)

if /I "%MODEL_BUCKET%"=="dummy-model" (
	echo [run-train-btc] Ignoring test MODEL_BUCKET=dummy-model
	set MODEL_BUCKET=
)

call "%~dp0run-train-gpu.bat" %*
exit /b %ERRORLEVEL%
