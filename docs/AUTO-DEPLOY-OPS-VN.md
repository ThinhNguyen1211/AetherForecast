# AetherForecast - Huong Dan Deploy + Van Hanh Tu Dong (VN)

Tai lieu nay su dung CDK CLI truc tiep thay vi script wrapper (cac script PowerShell cu da duoc loai bo trong phien ban hien tai).

## 1) Thong tin ban can dien truoc

| Bien | Bat buoc | Vi du |
|---|---|---|
| LetsEncryptEmail | Co (neu deploy domain HTTPS) | ops@aetherforcast.io.vn |
| ApiDomainName | Co | :80 (bootstrap) hoac api.aetherforcast.io.vn |
| GitHubRepo | Co neu set variable tu dong | owner/repo |
| GitHubOidcRoleName | Co neu muon script gan policy rollout | github-actions-aetherforecast |
| ProdApprovers | Khuyen nghi | user1,user2 |
| AllowCidr | Khuyen nghi | x.x.x.x/32 |

Neu khong truyen AllowCidr, CDK se su dung IP public hien tai /32.

## 2) Deploy lan dau (CDK CLI)

Buoc nay:
- Build/push backend image len ECR (neu can)
- Build/synth/bootstrap/deploy CDK
- Lay CloudFormation outputs tu AWS Console hoac CLI

~~~powershell
# 1. Build + push image (neu chua co)
$env:AWS_REGION = "ap-southeast-1"
docker build -t aetherforecast/backend:latest -f packages/backend/Dockerfile packages/backend
docker push <ecr-repo-uri>/aetherforecast-backend:latest

# 2. Deploy CDK
npm install
npm --workspace @aetherforecast/infra run build
npm --workspace @aetherforecast/infra run bootstrap
npm --workspace @aetherforecast/infra run deploy -- `
  -c backendImageUri=<ecr-repo-uri>/aetherforecast-backend:latest `
  -c apiDomainName=:80 `
  -c corsOrigins="*" `
  -c letsEncryptEmail=ops@aetherforcast.io.vn
~~~

## 3) Cau hinh GitHub Actions (thu cong)

Sau khi CDK deploy xong, lay outputs va tao GitHub repository variables/secrets:

~~~powershell
aws cloudformation describe-stacks --stack-name AetherForecastStack --region ap-southeast-1 --query "Stacks[0].Outputs"
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
- Luu CloudFormation outputs (Console hoac CLI) de dung trong cac lenh SSM/AWS CLI ben duoi.

2. Sau cap DNS:
- Cho propagate (thuong 5-30 phut, co the lau hon)
- Sau do moi deploy lai voi ApiDomainName=api.aetherforcast.io.vn

3. Sau push backend de test auto-rollout:
- Cho GitHub Actions backend workflow xong
- Moi chay rollout-check de xac nhan

## 6) Chuyen tu HTTP bootstrap sang HTTPS API domain

~~~powershell
npm --workspace @aetherforecast/infra run deploy -- `
  -c backendImageUri=<ecr-repo-uri>/aetherforecast-backend:latest `
  -c apiDomainName=api.aetherforcast.io.vn `
  -c corsOrigins="https://aetherforcast.io.vn,https://www.aetherforcast.io.vn" `
  -c letsEncryptEmail=ops@aetherforcast.io.vn
~~~

## 7) Lenh van hanh nhanh (AWS CLI / SSM)

Lay InstanceId tu CloudFormation outputs truoc khi chay SSM.

### 7.1 Kiem tra health
~~~powershell
$ApiUrl = "https://api.aetherforcast.io.vn"  # hoac EIP khi chua co domain
Invoke-RestMethod -Uri "$ApiUrl/health" -Method GET
~~~

### 7.2 Theo doi logs backend
~~~powershell
aws logs tail /aetherforecast/ec2-backend --since 30m --follow --region ap-southeast-1
~~~

### 7.3 Trigger cron data fetch thu cong
~~~powershell
aws ssm send-command --instance-ids <InstanceId> --document-name "AWS-RunShellScript" `
  --parameters commands="sudo /usr/local/bin/aetherforecast-fetch-cron.sh" `
  --region ap-southeast-1
~~~

### 7.4 Kiem tra parquet S3
~~~powershell
aws s3 ls s3://<ParquetDataBucketName>/market/klines/BTCUSDT/ --recursive --summarize --region ap-southeast-1
~~~

### 7.5 Submit train job thu cong
~~~powershell
aws batch submit-job --job-name af-train-manual --job-queue <BatchJobQueueArn> `
  --job-definition <BatchJobDefinitionArn> --region ap-southeast-1
~~~

### 7.6 Kiem tra trang thai train job
~~~powershell
aws batch describe-jobs --jobs <JobId> --region ap-southeast-1
~~~

### 7.7 Kiem tra alarms
~~~powershell
aws cloudwatch describe-alarms --alarm-name-prefix aetherforecast- --region ap-southeast-1 `
  --query "MetricAlarms[].[AlarmName,StateValue]" --output table
~~~

### 7.8 Mo dashboard
~~~powershell
Start-Process "https://ap-southeast-1.console.aws.amazon.com/cloudwatch/home?region=ap-southeast-1#dashboards:name=AetherForecast-Operations"
~~~

### 7.9 Kiem tra rollout backend tren EC2
~~~powershell
aws ssm send-command --instance-ids <InstanceId> --document-name "AWS-RunShellScript" `
  --parameters commands="docker ps --format 'table {{.Image}}\t{{.Status}}\t{{.Names}}'" `
  --region ap-southeast-1
~~~

### 7.10 Vao EC2 bang SSM
~~~powershell
aws ssm start-session --target <InstanceId> --region ap-southeast-1
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
