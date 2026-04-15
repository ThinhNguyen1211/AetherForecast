[CmdletBinding()]
param(
    [string]$AwsRegion = "ap-southeast-1",
    [string]$StackName = "AetherForecastStack",
    [string]$BackendEcrRepository = "aetherforecast/backend",
    [string]$BackendImageTag = "",
    [string]$ApiDomainName = ":80",
    [string]$LetsEncryptEmail = "",
    [string]$CorsOrigins = "*",
    [string]$AllowCidr = "",
    [switch]$SkipNpmInstall,
    [switch]$SkipDockerBuildPush,
    [switch]$SkipCdkDeploy,
    [switch]$ConfigureGitHub,
    [string]$GitHubRepo = "",
    [string]$GitHubOidcRoleName = "",
    [string]$GitHubOidcRoleArn = "",
    [string]$ProdApprovers = "",
    [switch]$ShowPlanOnly
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$ArtifactsDir = Join-Path $ProjectRoot "artifacts"

function Write-Section {
    param([string]$Message)
    Write-Host ""
    Write-Host ("=== {0} ===" -f $Message) -ForegroundColor Cyan
}

function Require-Command {
    param([string]$Name)
    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        throw "Khong tim thay lenh '$Name'. Hay cai dat truoc khi chay script."
    }
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

function Ensure-EcrRepository {
    param(
        [string]$Repository,
        [string]$Region
    )

    Write-Section "Kiem tra ECR repository backend"
    $repositoryExists = $true
    try {
        $null = Invoke-Tool -Exe "aws" -Arguments @(
            "ecr", "describe-repositories",
            "--repository-names", $Repository,
            "--region", $Region,
            "--output", "json"
        ) -CaptureOutput
    }
    catch {
        $repositoryExists = $false
    }

    if (-not $repositoryExists) {
        Write-Host "Repository chua ton tai. Dang tao moi..." -ForegroundColor Yellow
        Invoke-Tool -Exe "aws" -Arguments @(
            "ecr", "create-repository",
            "--repository-name", $Repository,
            "--image-scanning-configuration", "scanOnPush=true",
            "--region", $Region
        )
    }
    else {
        Write-Host "Repository da ton tai." -ForegroundColor Green
    }
}

function Get-StackOutputsMap {
    param(
        [string]$Name,
        [string]$Region
    )

    $json = Invoke-Tool -Exe "aws" -Arguments @(
        "cloudformation", "describe-stacks",
        "--stack-name", $Name,
        "--region", $Region,
        "--output", "json"
    ) -CaptureOutput

    $obj = $json | ConvertFrom-Json
    $map = @{}

    if (-not $obj.Stacks -or $obj.Stacks.Count -eq 0) {
        throw "Khong tim thay stack '$Name' trong region '$Region'."
    }

    foreach ($item in $obj.Stacks[0].Outputs) {
        $map[$item.OutputKey] = $item.OutputValue
    }

    return $map
}

function Get-CloudFrontDistributionIdFromDomain {
    param([string]$Domain)

    if ([string]::IsNullOrWhiteSpace($Domain)) {
        return ""
    }

    $query = "DistributionList.Items[?DomainName=='$Domain'].Id | [0]"
    $id = Invoke-Tool -Exe "aws" -Arguments @(
        "cloudfront", "list-distributions",
        "--query", $query,
        "--output", "text"
    ) -CaptureOutput

    if ($id -eq "None") {
        return ""
    }

    return $id.Trim()
}

function Write-OutputsArtifacts {
    param(
        [hashtable]$Data,
        [string]$Dir
    )

    New-Item -ItemType Directory -Path $Dir -Force | Out-Null

    $jsonPath = Join-Path $Dir "deploy-outputs.json"
    $envPath = Join-Path $Dir "deploy-outputs.env"

    ($Data | ConvertTo-Json -Depth 5) | Set-Content -Path $jsonPath -Encoding UTF8

    $lines = @()
    foreach ($key in $Data.Keys) {
        $value = $Data[$key]
        if ($null -eq $value) {
            $value = ""
        }
        $lines += ("{0}={1}" -f $key, $value)
    }
    $lines | Set-Content -Path $envPath -Encoding ASCII

    return @{ Json = $jsonPath; Env = $envPath }
}

function Set-GitHubVariableSafe {
    param(
        [string]$Repo,
        [string]$Name,
        [string]$Value
    )

    if ([string]::IsNullOrWhiteSpace($Value)) {
        Write-Host ("Bo qua variable {0} vi chua co gia tri." -f $Name) -ForegroundColor Yellow
        return
    }

    Invoke-Tool -Exe "gh" -Arguments @("variable", "set", $Name, "--repo", $Repo, "--body", $Value)
    Write-Host ("Da set variable {0}" -f $Name) -ForegroundColor Green
}

function Resolve-AllowCidr {
    param([string]$InputValue)

    if (-not [string]::IsNullOrWhiteSpace($InputValue)) {
        return $InputValue.Trim()
    }

    try {
        $ip = (Invoke-RestMethod -Uri "https://checkip.amazonaws.com").Trim()
        if ([string]::IsNullOrWhiteSpace($ip)) {
            throw "Khong lay duoc public IP"
        }
        return ("{0}/32" -f $ip)
    }
    catch {
        Write-Host "Khong tu lay duoc IP. Dung tam 0.0.0.0/0 (chi nen dung de bootstrap)." -ForegroundColor Yellow
        return "0.0.0.0/0"
    }
}

if ([string]::IsNullOrWhiteSpace($BackendImageTag)) {
    $BackendImageTag = Get-Date -Format "yyyyMMddHHmmss"
}

$AllowCidr = Resolve-AllowCidr -InputValue $AllowCidr

if ($ShowPlanOnly) {
    Write-Section "Ke hoach chay"
    Write-Host ("ProjectRoot: {0}" -f $ProjectRoot)
    Write-Host ("Region: {0}" -f $AwsRegion)
    Write-Host ("StackName: {0}" -f $StackName)
    Write-Host ("BackendEcrRepository: {0}" -f $BackendEcrRepository)
    Write-Host ("BackendImageTag: {0}" -f $BackendImageTag)
    Write-Host ("ApiDomainName: {0}" -f $ApiDomainName)
    Write-Host ("AllowCidr: {0}" -f $AllowCidr)
    Write-Host ("SkipNpmInstall: {0}" -f $SkipNpmInstall.IsPresent)
    Write-Host ("SkipDockerBuildPush: {0}" -f $SkipDockerBuildPush.IsPresent)
    Write-Host ("SkipCdkDeploy: {0}" -f $SkipCdkDeploy.IsPresent)
    Write-Host ("ConfigureGitHub: {0}" -f $ConfigureGitHub.IsPresent)
    exit 0
}

Write-Section "Kiem tra cong cu"
Require-Command -Name "aws"
Require-Command -Name "docker"
Require-Command -Name "npm"
Require-Command -Name "npx"
Require-Command -Name "cdk"
if ($ConfigureGitHub) {
    Require-Command -Name "gh"
}

Push-Location $ProjectRoot
try {
    Write-Section "Xac nhan AWS account"
    $AwsAccountId = (Invoke-Tool -Exe "aws" -Arguments @("sts", "get-caller-identity", "--query", "Account", "--output", "text") -CaptureOutput).Trim()
    if ([string]::IsNullOrWhiteSpace($AwsAccountId)) {
        throw "Khong lay duoc AWS account id"
    }
    Write-Host ("AWS Account: {0}" -f $AwsAccountId) -ForegroundColor Green

    $BackendRegistry = "{0}.dkr.ecr.{1}.amazonaws.com" -f $AwsAccountId, $AwsRegion
    $BackendImageUriSha = "{0}/{1}:{2}" -f $BackendRegistry, $BackendEcrRepository, $BackendImageTag
    $BackendImageUriLatest = "{0}/{1}:latest" -f $BackendRegistry, $BackendEcrRepository

    if (-not $SkipDockerBuildPush) {
        Ensure-EcrRepository -Repository $BackendEcrRepository -Region $AwsRegion

        Write-Section "Dang login ECR"
        $loginPassword = Invoke-Tool -Exe "aws" -Arguments @("ecr", "get-login-password", "--region", $AwsRegion) -CaptureOutput
        $loginPassword | & docker login --username AWS --password-stdin $BackendRegistry
        if ($LASTEXITCODE -ne 0) {
            throw "Docker login ECR that bai"
        }

        Write-Section "Build backend image (runtime)"
        Invoke-Tool -Exe "docker" -Arguments @(
            "build",
            "--target", "runtime",
            "-f", "packages/backend/Dockerfile",
            "-t", $BackendImageUriSha,
            "-t", $BackendImageUriLatest,
            "packages/backend"
        )

        Write-Section "Push backend image"
        Invoke-Tool -Exe "docker" -Arguments @("push", $BackendImageUriSha)
        Invoke-Tool -Exe "docker" -Arguments @("push", $BackendImageUriLatest)
    }
    else {
        Write-Host "Bo qua build/push backend image theo yeu cau." -ForegroundColor Yellow
    }

    if (-not $SkipNpmInstall) {
        Write-Section "Cai dependencies"
        Invoke-Tool -Exe "npm" -Arguments @("ci", "--workspaces", "--include-workspace-root")
    }
    else {
        Write-Host "Bo qua npm install theo yeu cau." -ForegroundColor Yellow
    }

    Write-Section "Build + synth + bootstrap infra"
    Invoke-Tool -Exe "npm" -Arguments @("--workspace", "@aetherforecast/infra", "run", "build")
    Invoke-Tool -Exe "npm" -Arguments @("--workspace", "@aetherforecast/infra", "run", "synth")
    Invoke-Tool -Exe "npm" -Arguments @("--workspace", "@aetherforecast/infra", "run", "bootstrap")

    if (-not $SkipCdkDeploy) {
        Write-Section "Deploy CDK stack"
        $env:CDK_DISABLE_VERSION_CHECK = "1"

        $deployArgs = @(
            "cdk", "deploy", $StackName,
            "--require-approval", "never",
            "--parameters", "BackendImageUri=$BackendImageUriLatest",
            "--parameters", "ApiDomainName=$ApiDomainName",
            "--parameters", "CorsOrigins=$CorsOrigins",
            "--parameters", "AllowCidr=$AllowCidr"
        )

        if (-not [string]::IsNullOrWhiteSpace($LetsEncryptEmail)) {
            $deployArgs += @("--parameters", "LetsEncryptEmail=$LetsEncryptEmail")
        }

        Push-Location (Join-Path $ProjectRoot "packages/infra")
        try {
            Invoke-Tool -Exe "npx" -Arguments $deployArgs
        }
        finally {
            Pop-Location
        }
    }
    else {
        Write-Host "Bo qua CDK deploy theo yeu cau." -ForegroundColor Yellow
    }

    Write-Section "Thu thap outputs sau deploy"
    $stackOutputs = Get-StackOutputsMap -Name $StackName -Region $AwsRegion
    $cloudFrontDomain = if ($stackOutputs.ContainsKey("CloudFrontDistributionDomain")) { $stackOutputs["CloudFrontDistributionDomain"] } else { "" }
    $cloudFrontDistributionId = Get-CloudFrontDistributionIdFromDomain -Domain $cloudFrontDomain

    $apiBaseUrl = ""
    if ($ApiDomainName -eq ":80") {
        if ($stackOutputs.ContainsKey("BackendElasticIp")) {
            $apiBaseUrl = "http://{0}" -f $stackOutputs["BackendElasticIp"]
        }
    }
    else {
        $apiBaseUrl = "https://{0}" -f $ApiDomainName
    }

    $result = [ordered]@{
        GeneratedAtUtc = (Get-Date).ToUniversalTime().ToString("o")
        AwsRegion = $AwsRegion
        AwsAccountId = $AwsAccountId
        StackName = $StackName
        AllowCidr = $AllowCidr
        ApiDomainName = $ApiDomainName
        ApiBaseUrl = $apiBaseUrl
        BackendImageUriSha = $BackendImageUriSha
        BackendImageUriLatest = $BackendImageUriLatest
        ParquetDataBucketName = $stackOutputs["ParquetDataBucketName"]
        MlModelBucketName = $stackOutputs["MlModelBucketName"]
        TrainingEcrRepositoryUri = $stackOutputs["TrainingEcrRepositoryUri"]
        BackendEc2InstanceId = $stackOutputs["BackendEc2InstanceId"]
        BackendElasticIp = $stackOutputs["BackendElasticIp"]
        BackendEc2LogGroupName = $stackOutputs["BackendEc2LogGroupName"]
        BatchJobQueueArn = $stackOutputs["BatchJobQueueArn"]
        BatchJobDefinitionArn = $stackOutputs["BatchJobDefinitionArn"]
        CognitoUserPoolId = $stackOutputs["CognitoUserPoolId"]
        CognitoAppClientId = $stackOutputs["CognitoAppClientId"]
        FrontendBucketName = $stackOutputs["FrontendBucketName"]
        CloudFrontDistributionDomain = $cloudFrontDomain
        CloudFrontDistributionId = $cloudFrontDistributionId
        OperationsDashboardName = $stackOutputs["OperationsDashboardName"]
        OperationsAlarmTopicArn = $stackOutputs["OperationsAlarmTopicArn"]
        OperationsDashboardUrl = $stackOutputs["OperationsDashboardUrl"]
    }

    $paths = Write-OutputsArtifacts -Data $result -Dir $ArtifactsDir
    Write-Host ("Da ghi outputs JSON: {0}" -f $paths.Json) -ForegroundColor Green
    Write-Host ("Da ghi outputs ENV : {0}" -f $paths.Env) -ForegroundColor Green

    if ($ConfigureGitHub) {
        if ([string]::IsNullOrWhiteSpace($GitHubRepo)) {
            throw "Ban bat ConfigureGitHub nhung chua truyen -GitHubRepo (owner/repo)."
        }

        if ([string]::IsNullOrWhiteSpace($GitHubOidcRoleArn) -and -not [string]::IsNullOrWhiteSpace($GitHubOidcRoleName)) {
            $GitHubOidcRoleArn = "arn:aws:iam::{0}:role/{1}" -f $AwsAccountId, $GitHubOidcRoleName
        }

        if (-not [string]::IsNullOrWhiteSpace($GitHubOidcRoleName)) {
            Write-Section "Gan policy rollout vao OIDC role"
            $policyPath = Join-Path $ProjectRoot ".github/iam/oidc-role-ec2-rollout-policy.json"
            Invoke-Tool -Exe "aws" -Arguments @(
                "iam", "put-role-policy",
                "--role-name", $GitHubOidcRoleName,
                "--policy-name", "AetherForecastEc2Rollout",
                "--policy-document", ("file://{0}" -f $policyPath)
            )
        }

        Write-Section "Cap nhat GitHub Repository Variables"
        Set-GitHubVariableSafe -Repo $GitHubRepo -Name "AWS_GITHUB_OIDC_ROLE_ARN" -Value $GitHubOidcRoleArn
        Set-GitHubVariableSafe -Repo $GitHubRepo -Name "BACKEND_EC2_INSTANCE_ID" -Value $result.BackendEc2InstanceId
        Set-GitHubVariableSafe -Repo $GitHubRepo -Name "FRONTEND_S3_BUCKET" -Value $result.FrontendBucketName
        Set-GitHubVariableSafe -Repo $GitHubRepo -Name "CLOUDFRONT_DISTRIBUTION_ID" -Value $result.CloudFrontDistributionId
        Set-GitHubVariableSafe -Repo $GitHubRepo -Name "PROD_APPROVERS" -Value $ProdApprovers
        Set-GitHubVariableSafe -Repo $GitHubRepo -Name "ALLOW_CIDR" -Value $AllowCidr
        Set-GitHubVariableSafe -Repo $GitHubRepo -Name "VITE_API_BASE_URL" -Value $apiBaseUrl
        Set-GitHubVariableSafe -Repo $GitHubRepo -Name "COGNITO_USER_POOL_ID" -Value $result.CognitoUserPoolId
        Set-GitHubVariableSafe -Repo $GitHubRepo -Name "COGNITO_APP_CLIENT_ID" -Value $result.CognitoAppClientId
    }

    Write-Section "Buoc thu cong con lai"
    Write-Host "1) DNS nha dang ky domain (bat buoc thu cong):"
    Write-Host ("   - Frontend root/www -> {0}" -f $result.CloudFrontDistributionDomain)
    Write-Host ("   - API A record      -> {0}" -f $result.BackendElasticIp)
    Write-Host "2) Neu CloudFront dung custom domain, ban can cap ACM cert o us-east-1 va gan vao distribution."
    Write-Host "3) GitHub secrets webhook (Slack/Teams) can set thu cong neu su dung."
    Write-Host "4) Sau khi DNS API xong, deploy lai voi -ApiDomainName api.<domain> de Caddy cap HTTPS tu dong."

    Write-Section "Hoan tat"
    Write-Host "Script deploy hoan tat."
}
finally {
    Pop-Location
}

