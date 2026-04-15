# AetherForecast - Huong Dan Deploy + Van Hanh Tu Dong (VN)

Tai lieu nay di kem 2 script:
- scripts/deploy-e2e.ps1: Tu dong build/push/deploy/lay outputs/cap nhat GitHub variables.
- scripts/af-ops.ps1: Lenh van hanh nhanh (health, logs, cron, train, alarms, SSM).

## 1) Thong tin ban can dien truoc

| Bien | Bat buoc | Vi du |
|---|---|---|
| LetsEncryptEmail | Co (neu deploy domain HTTPS) | ops@aetherforcast.io.vn |
| ApiDomainName | Co | :80 (bootstrap) hoac api.aetherforcast.io.vn |
| GitHubRepo | Co neu set variable tu dong | owner/repo |
| GitHubOidcRoleName | Co neu muon script gan policy rollout | github-actions-aetherforecast |
| ProdApprovers | Khuyen nghi | user1,user2 |
| AllowCidr | Khuyen nghi | x.x.x.x/32 |

Neu khong truyen AllowCidr, script se co gang tu lay public IP va set /32.

## 2) Chay full auto (lan dau)

Buoc nay se tu dong:
- Kiem tra tool can thiet
- Build + push backend image len ECR
- Build/synth/bootstrap/deploy CDK
- Lay CloudFormation outputs
- Ghi ra artifacts/deploy-outputs.json va artifacts/deploy-outputs.env

~~~powershell
powershell -ExecutionPolicy Bypass -File .\scripts\deploy-e2e.ps1 \
  -LetsEncryptEmail "ops@aetherforcast.io.vn" \
  -ApiDomainName ":80" \
  -CorsOrigins "*"
~~~

## 3) Chay full auto + set GitHub variables + gan IAM rollout policy

~~~powershell
powershell -ExecutionPolicy Bypass -File .\scripts\deploy-e2e.ps1 \
  -LetsEncryptEmail "ops@aetherforcast.io.vn" \
  -ApiDomainName ":80" \
  -ConfigureGitHub \
  -GitHubRepo "owner/repo" \
  -GitHubOidcRoleName "github-actions-aetherforecast" \
  -ProdApprovers "user1,user2"
~~~

## 4) Buoc bat buoc thu cong

1. DNS tai nha dang ky domain
- Frontend root/www tro CloudFrontDistributionDomain
- API A record tro BackendElasticIp

2. Neu dung custom domain cho CloudFront
- Tao ACM cert o us-east-1
- Gan cert vao distribution

3. GitHub Secrets webhook (neu dung)
- SLACK_WEBHOOK_URL
- TEAMS_WEBHOOK_URL

## 5) Buoc can cho ket qua roi moi lam tiep

1. Sau CDK deploy:
- Phai co artifacts/deploy-outputs.env truoc khi chay script van hanh

2. Sau cap DNS:
- Cho propagate (thuong 5-30 phut, co the lau hon)
- Sau do moi deploy lai voi ApiDomainName=api.aetherforcast.io.vn

3. Sau push backend de test auto-rollout:
- Cho GitHub Actions backend workflow xong
- Moi chay rollout-check de xac nhan

## 6) Chuyen tu HTTP bootstrap sang HTTPS API domain

~~~powershell
powershell -ExecutionPolicy Bypass -File .\scripts\deploy-e2e.ps1 \
  -SkipDockerBuildPush \
  -SkipNpmInstall \
  -ApiDomainName "api.aetherforcast.io.vn" \
  -LetsEncryptEmail "ops@aetherforcast.io.vn" \
  -CorsOrigins "https://aetherforcast.io.vn,https://www.aetherforcast.io.vn"
~~~

## 7) Lenh van hanh nhanh (scripts/af-ops.ps1)

### 7.1 Kiem tra health
~~~powershell
powershell -ExecutionPolicy Bypass -File .\scripts\af-ops.ps1 -Action health -AutoLoadOutputs
~~~

### 7.2 Theo doi logs backend
~~~powershell
powershell -ExecutionPolicy Bypass -File .\scripts\af-ops.ps1 -Action tail-logs -TailMinutes 60 -AutoLoadOutputs
~~~

### 7.3 Trigger cron data fetch thu cong
~~~powershell
powershell -ExecutionPolicy Bypass -File .\scripts\af-ops.ps1 -Action trigger-cron -AutoLoadOutputs
~~~

### 7.4 Kiem tra parquet S3
~~~powershell
powershell -ExecutionPolicy Bypass -File .\scripts\af-ops.ps1 -Action check-s3 -Symbol BTCUSDT -AutoLoadOutputs
~~~

### 7.5 Submit train job thu cong
~~~powershell
powershell -ExecutionPolicy Bypass -File .\scripts\af-ops.ps1 -Action submit-train -AutoLoadOutputs
~~~

### 7.6 Kiem tra trang thai train job
~~~powershell
powershell -ExecutionPolicy Bypass -File .\scripts\af-ops.ps1 -Action check-train -JobId <JOB_ID> -AutoLoadOutputs
~~~

### 7.7 Kiem tra alarms
~~~powershell
powershell -ExecutionPolicy Bypass -File .\scripts\af-ops.ps1 -Action alarms -AutoLoadOutputs
~~~

### 7.8 Mo dashboard
~~~powershell
powershell -ExecutionPolicy Bypass -File .\scripts\af-ops.ps1 -Action dashboard -AutoLoadOutputs
~~~

### 7.9 Kiem tra rollout backend tren EC2
~~~powershell
powershell -ExecutionPolicy Bypass -File .\scripts\af-ops.ps1 -Action rollout-check -AutoLoadOutputs
~~~

### 7.10 Vao EC2 bang SSM
~~~powershell
powershell -ExecutionPolicy Bypass -File .\scripts\af-ops.ps1 -Action ssm -AutoLoadOutputs
~~~

## 8) Lenh quan trong can nho khi van hanh

1. Trigger cron thu cong:
~~~bash
sudo /usr/local/bin/aetherforecast-fetch-cron.sh
~~~

2. Kiem tra health local tren EC2:
~~~bash
curl -fsS http://127.0.0.1/health
~~~

3. Kiem tra container:
~~~bash
docker ps
docker logs --tail 200 aetherforecast-backend
~~~

4. Submit train bang AWS CLI truc tiep:
~~~powershell
aws batch submit-job --job-name af-train-manual --job-queue <BatchJobQueueArn_or_name> --job-definition <BatchJobDefinitionArn_or_name> --region ap-southeast-1
~~~

5. Kiem tra log group backend:
~~~powershell
aws logs tail /aetherforecast/ec2-backend --since 30m --follow --region ap-southeast-1
~~~

6. Kiem tra alarms:
~~~powershell
aws cloudwatch describe-alarms --alarm-name-prefix aetherforecast- --region ap-southeast-1 --query "MetricAlarms[].[AlarmName,StateValue]" --output table
~~~
