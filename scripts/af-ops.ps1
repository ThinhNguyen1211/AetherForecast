[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("health", "tail-logs", "trigger-cron", "check-s3", "submit-train", "check-train", "alarms", "dashboard", "rollout-check", "ssm")]
    [string]$Action,

    [string]$AwsRegion = "ap-southeast-1",
    [string]$StackName = "AetherForecastStack",
    [string]$OutputsFile = "",
    [switch]$AutoLoadOutputs,

    [string]$ApiBaseUrl = "",
    [string]$BackendEc2InstanceId = "",
    [string]$BackendEc2LogGroupName = "",
    [string]$ParquetDataBucketName = "",
    [string]$BatchJobQueueArn = "",
    [string]$BatchJobDefinitionArn = "",
    [string]$OperationsDashboardUrl = "",

    [string]$Symbol = "BTCUSDT",
    [int]$TailMinutes = 30,
    [string]$JobId = ""
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
if ([string]::IsNullOrWhiteSpace($OutputsFile)) {
    $OutputsFile = Join-Path $ProjectRoot "artifacts/deploy-outputs.env"
}

function Write-Section {
    param([string]$Message)
    Write-Host ""
    Write-Host ("=== {0} ===" -f $Message) -ForegroundColor Cyan
}

function Invoke-Tool {
    param(
        [string]$Exe,
        [string[]]$Arguments,
        [switch]$CaptureOutput
    )

    $previousErrorPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"

    if ($CaptureOutput) {
        try {
            $raw = & $Exe @Arguments 2>&1
            $exitCode = $LASTEXITCODE
        }
        finally {
            $ErrorActionPreference = $previousErrorPreference
        }
        if ($exitCode -ne 0) {
            throw ("Lenh that bai: {0} {1}`n{2}" -f $Exe, ($Arguments -join " "), (($raw | ForEach-Object { "$_" }) -join "`n"))
        }
        return (($raw | ForEach-Object { "$_" }) -join "`n")
    }

    try {
        & $Exe @Arguments
    }
    finally {
        $ErrorActionPreference = $previousErrorPreference
    }
    if ($LASTEXITCODE -ne 0) {
        throw ("Lenh that bai: {0} {1}" -f $Exe, ($Arguments -join " "))
    }
}

function Import-EnvFileMap {
    param([string]$Path)

    $map = @{}
    if (-not (Test-Path $Path)) {
        return $map
    }

    foreach ($line in Get-Content -Path $Path) {
        if ([string]::IsNullOrWhiteSpace($line)) {
            continue
        }
        if ($line.TrimStart().StartsWith("#")) {
            continue
        }

        $pair = $line.Split("=", 2)
        if ($pair.Count -eq 2) {
            $map[$pair[0].Trim()] = $pair[1].Trim()
        }
    }

    return $map
}

function Get-StackOutputsMap {
    param(
        [string]$Name,
        [string]$Region
    )

    $map = @{}

    try {
        $json = Invoke-Tool -Exe "aws" -Arguments @(
            "cloudformation", "describe-stacks",
            "--stack-name", $Name,
            "--region", $Region,
            "--output", "json"
        ) -CaptureOutput

        $obj = $json | ConvertFrom-Json
        if (-not $obj.Stacks -or $obj.Stacks.Count -eq 0) {
            return $map
        }

        foreach ($item in $obj.Stacks[0].Outputs) {
            $map[$item.OutputKey] = $item.OutputValue
        }
    }
    catch {
        Write-Host "Khong lay duoc outputs tu CloudFormation stack. Van tiep tuc neu ban da truyen du tham so." -ForegroundColor Yellow
    }

    return $map
}

function Require-Value {
    param(
        [string]$Name,
        [string]$Value
    )

    if ([string]::IsNullOrWhiteSpace($Value)) {
        throw ("Thieu gia tri bat buoc: {0}. Hay truyen param hoac bat -AutoLoadOutputs." -f $Name)
    }
}

function Invoke-SsmCommands {
    param(
        [string]$InstanceId,
        [string]$Region,
        [string]$Comment,
        [string[]]$Commands
    )

    Require-Value -Name "BackendEc2InstanceId" -Value $InstanceId

    $tmp = Join-Path $env:TEMP ("af-ssm-{0}.json" -f ([Guid]::NewGuid().ToString("N")))
    $payload = @{ commands = $Commands } | ConvertTo-Json -Depth 5
    Set-Content -Path $tmp -Value $payload -Encoding ASCII

    try {
        $paramsArg = "file://{0}" -f ($tmp.Replace("\", "/"))

        $commandId = (Invoke-Tool -Exe "aws" -Arguments @(
            "ssm", "send-command",
            "--instance-ids", $InstanceId,
            "--document-name", "AWS-RunShellScript",
            "--comment", $Comment,
            "--parameters", $paramsArg,
            "--region", $Region,
            "--query", "Command.CommandId",
            "--output", "text"
        ) -CaptureOutput).Trim()

        Write-Host ("CommandId: {0}" -f $commandId) -ForegroundColor Green

        $maxPolls = 180
        $pollIntervalSeconds = 5
        $invocation = $null

        for ($poll = 1; $poll -le $maxPolls; $poll += 1) {
            $invocationRaw = Invoke-Tool -Exe "aws" -Arguments @(
                "ssm", "get-command-invocation",
                "--command-id", $commandId,
                "--instance-id", $InstanceId,
                "--region", $Region,
                "--output", "json"
            ) -CaptureOutput

            $invocation = $invocationRaw | ConvertFrom-Json
            $status = [string]$invocation.Status

            if ($status -in @("Success", "Failed", "Cancelled", "TimedOut", "Cancelling")) {
                break
            }

            Write-Host ("Dang cho SSM command... attempt {0}/{1}, status={2}" -f $poll, $maxPolls, $status)
            Start-Sleep -Seconds $pollIntervalSeconds
        }

        if ($null -eq $invocation) {
            throw "Khong lay duoc ket qua SSM invocation."
        }

        Write-Host ("Status: {0}" -f $invocation.Status)

        if ($invocation.StandardOutputContent) {
            Write-Host ""
            Write-Host "----- STDOUT -----" -ForegroundColor DarkGray
            Write-Host $invocation.StandardOutputContent
        }

        if ($invocation.StandardErrorContent) {
            Write-Host ""
            Write-Host "----- STDERR -----" -ForegroundColor Yellow
            Write-Host $invocation.StandardErrorContent
        }

        if ($invocation.Status -ne "Success") {
            throw "SSM command khong thanh cong."
        }
    }
    finally {
        Remove-Item -Path $tmp -Force -ErrorAction SilentlyContinue
    }
}

Write-Section "Nap context"
$envMap = @{}
$stackMap = @{}

if ($AutoLoadOutputs) {
    $envMap = Import-EnvFileMap -Path $OutputsFile
    if ($envMap.Count -gt 0) {
        Write-Host ("Da nap outputs tu file: {0}" -f $OutputsFile) -ForegroundColor Green
    }
    else {
        Write-Host "Khong tim thay file outputs. Thu nap truc tiep tu CloudFormation stack..." -ForegroundColor Yellow
    }
}

$stackMap = Get-StackOutputsMap -Name $StackName -Region $AwsRegion

if ([string]::IsNullOrWhiteSpace($ApiBaseUrl)) {
    if ($envMap.ContainsKey("ApiBaseUrl")) { $ApiBaseUrl = $envMap["ApiBaseUrl"] }
}
if ([string]::IsNullOrWhiteSpace($BackendEc2InstanceId)) {
    if ($envMap.ContainsKey("BackendEc2InstanceId")) { $BackendEc2InstanceId = $envMap["BackendEc2InstanceId"] }
    elseif ($stackMap.ContainsKey("BackendEc2InstanceId")) { $BackendEc2InstanceId = $stackMap["BackendEc2InstanceId"] }
}
if ([string]::IsNullOrWhiteSpace($BackendEc2LogGroupName)) {
    if ($envMap.ContainsKey("BackendEc2LogGroupName")) { $BackendEc2LogGroupName = $envMap["BackendEc2LogGroupName"] }
    elseif ($stackMap.ContainsKey("BackendEc2LogGroupName")) { $BackendEc2LogGroupName = $stackMap["BackendEc2LogGroupName"] }
}
if ([string]::IsNullOrWhiteSpace($ParquetDataBucketName)) {
    if ($envMap.ContainsKey("ParquetDataBucketName")) { $ParquetDataBucketName = $envMap["ParquetDataBucketName"] }
    elseif ($stackMap.ContainsKey("ParquetDataBucketName")) { $ParquetDataBucketName = $stackMap["ParquetDataBucketName"] }
}
if ([string]::IsNullOrWhiteSpace($BatchJobQueueArn)) {
    if ($stackMap.ContainsKey("BatchJobQueueArn")) { $BatchJobQueueArn = $stackMap["BatchJobQueueArn"] }
    elseif ($envMap.ContainsKey("BatchJobQueueArn")) { $BatchJobQueueArn = $envMap["BatchJobQueueArn"] }
}
if ([string]::IsNullOrWhiteSpace($BatchJobDefinitionArn)) {
    if ($stackMap.ContainsKey("BatchJobDefinitionArn")) { $BatchJobDefinitionArn = $stackMap["BatchJobDefinitionArn"] }
    elseif ($envMap.ContainsKey("BatchJobDefinitionArn")) { $BatchJobDefinitionArn = $envMap["BatchJobDefinitionArn"] }
}
if ([string]::IsNullOrWhiteSpace($OperationsDashboardUrl)) {
    if ($envMap.ContainsKey("OperationsDashboardUrl")) { $OperationsDashboardUrl = $envMap["OperationsDashboardUrl"] }
    elseif ($stackMap.ContainsKey("OperationsDashboardUrl")) { $OperationsDashboardUrl = $stackMap["OperationsDashboardUrl"] }
}

switch ($Action) {
    "health" {
        Write-Section "Kiem tra API health + docs"
        Require-Value -Name "ApiBaseUrl" -Value $ApiBaseUrl

        $health = Invoke-RestMethod -Method Get -Uri ("{0}/health" -f $ApiBaseUrl)
        Write-Host "Health:" -ForegroundColor Green
        $health | ConvertTo-Json -Depth 5 | Write-Host

        $null = Invoke-WebRequest -Method Head -UseBasicParsing -Uri ("{0}/docs" -f $ApiBaseUrl)
        Write-Host "Docs reachable: OK" -ForegroundColor Green
    }

    "tail-logs" {
        Write-Section "Tail CloudWatch logs backend"
        Require-Value -Name "BackendEc2LogGroupName" -Value $BackendEc2LogGroupName
        Invoke-Tool -Exe "aws" -Arguments @(
            "logs", "tail", $BackendEc2LogGroupName,
            "--since", ("{0}m" -f $TailMinutes),
            "--follow",
            "--region", $AwsRegion
        )
    }

    "trigger-cron" {
        Write-Section "Trigger cron fetch thu cong tren EC2"
        Invoke-SsmCommands -InstanceId $BackendEc2InstanceId -Region $AwsRegion -Comment "Manual cron fetch trigger" -Commands @(
            "set -euo pipefail",
            "sudo /usr/local/bin/aetherforecast-fetch-cron.sh",
            'echo "cron script exit code: $?"',
            "sudo tail -n 120 /var/log/aetherforecast-cron.log"
        )
    }

    "check-s3" {
        Write-Section "Kiem tra parquet trong S3"
        Require-Value -Name "ParquetDataBucketName" -Value $ParquetDataBucketName
        $upperSymbol = $Symbol.ToUpperInvariant()

        $paths = @(
            ("s3://{0}/market/klines/symbol={1}/" -f $ParquetDataBucketName, $upperSymbol),
            ("s3://{0}/symbol={1}/" -f $ParquetDataBucketName, $upperSymbol)
        )

        foreach ($path in $paths) {
            Write-Host ("Path: {0}" -f $path) -ForegroundColor DarkGray
            $previousErrorPreference = $ErrorActionPreference
            $ErrorActionPreference = "Continue"
            try {
                $output = & aws s3 ls $path --recursive --region $AwsRegion 2>&1
                $exitCode = $LASTEXITCODE
            }
            finally {
                $ErrorActionPreference = $previousErrorPreference
            }

            if ($exitCode -ne 0 -or -not $output) {
                Write-Host "(khong co object hoac path khong ton tai)" -ForegroundColor Yellow
                continue
            }

            $output | Select-Object -First 120 | ForEach-Object { Write-Host $_ }
        }
    }

    "submit-train" {
        Write-Section "Submit manual training job"
        Require-Value -Name "BatchJobQueueArn" -Value $BatchJobQueueArn
        Require-Value -Name "BatchJobDefinitionArn" -Value $BatchJobDefinitionArn

        $jobName = "af-train-manual-{0}" -f (Get-Date -Format "yyyyMMddHHmmss")
        $submittedJobId = (Invoke-Tool -Exe "aws" -Arguments @(
            "batch", "submit-job",
            "--job-name", $jobName,
            "--job-queue", $BatchJobQueueArn,
            "--job-definition", $BatchJobDefinitionArn,
            "--region", $AwsRegion,
            "--query", "jobId",
            "--output", "text"
        ) -CaptureOutput).Trim()

        Write-Host ("Training job da submit: {0}" -f $submittedJobId) -ForegroundColor Green
        Write-Host ("Theo doi bang lenh: .\scripts\af-ops.ps1 -Action check-train -JobId {0} -AutoLoadOutputs" -f $submittedJobId)
    }

    "check-train" {
        Write-Section "Kiem tra training job"
        Require-Value -Name "JobId" -Value $JobId

        Invoke-Tool -Exe "aws" -Arguments @(
            "batch", "describe-jobs",
            "--jobs", $JobId,
            "--region", $AwsRegion,
            "--query", "jobs[0].{JobId:jobId,Status:status,StatusReason:statusReason,LogStream:container.logStreamName}",
            "--output", "table"
        )
    }

    "alarms" {
        Write-Section "Kiem tra CloudWatch alarms"
        Invoke-Tool -Exe "aws" -Arguments @(
            "cloudwatch", "describe-alarms",
            "--alarm-name-prefix", "aetherforecast-",
            "--region", $AwsRegion,
            "--query", "MetricAlarms[].[AlarmName,StateValue,MetricName,Namespace]",
            "--output", "table"
        )
    }

    "dashboard" {
        Write-Section "Mo dashboard"
        Require-Value -Name "OperationsDashboardUrl" -Value $OperationsDashboardUrl
        Start-Process $OperationsDashboardUrl
        Write-Host ("Da mo: {0}" -f $OperationsDashboardUrl) -ForegroundColor Green
    }

    "rollout-check" {
        Write-Section "Kiem tra backend rollout tren EC2"
        Invoke-SsmCommands -InstanceId $BackendEc2InstanceId -Region $AwsRegion -Comment "Rollout verification" -Commands @(
            "set -euo pipefail",
            'if [ -f /opt/aetherforecast/backend-image-uri.txt ]; then cat /opt/aetherforecast/backend-image-uri.txt; else echo "backend-image-uri.txt not found"; fi',
            "docker ps --format 'table {{.Names}}\t{{.Image}}\t{{.Status}}'",
            "curl -fsS http://127.0.0.1/health"
        )
    }

    "ssm" {
        Write-Section "Mo SSM session"
        Require-Value -Name "BackendEc2InstanceId" -Value $BackendEc2InstanceId
        & aws ssm start-session --target $BackendEc2InstanceId --region $AwsRegion
        if ($LASTEXITCODE -ne 0) {
            throw "Khong mo duoc SSM session"
        }
    }

    default {
        throw "Action khong hop le"
    }
}

