# AetherForecast Runbook

## 1. Purpose
This runbook describes day-2 operations for AetherForecast production workloads on AWS.

## 2. Production Components
- Frontend: S3 static hosting + CloudFront distribution.
- Backend API and realtime WebSocket: Single EC2 t3.micro host running Dockerized FastAPI behind Caddy.
- Data ingestion: Host cron (every 30 minutes) executes fetch pipeline in backend container and writes partitioned parquet to S3.
- Training: AWS Batch Spot GPU jobs with checkpointing and model promotion manifest.
- Inference: FastAPI endpoint loads active model from S3 manifest.
- Monitoring: CloudWatch dashboard + alarms + SNS notifications.

## 3. Critical Health Endpoints
- API health: GET /health on backend EC2 public endpoint (EIP/domain).
- API docs: GET /docs.
- WebSocket stream: /ws/{symbol}.

## 4. Routine Operations
### Daily
- Confirm CloudWatch dashboard widgets are green:
  - EC2 CPU and status checks
  - API request/5xx/latency custom metrics
  - WebSocket connections
  - Batch failed jobs
  - Cron fetch runs and failures
  - Parquet write and model promotion custom metrics
- Confirm latest cron run produced parquet objects under:
  - s3://<data-bucket>/market/klines/symbol=<SYMBOL>/year=<YYYY>/month=<MM>/day=<DD>/

### Weekly
- Review failed/slow requests in backend logs.
- Review Batch queue health and Spot interruptions.
- Review S3 object growth and retention costs.

## 5. Deploy Operations
### Infra deploy
```bash
cd packages/infra
npx cdk deploy --all
```

### Backend image rollout
1. Build and push image to ECR.
2. Backend GitHub Actions workflow automatically rolls out image to EC2 via AWS SSM Run Command.
3. Verify /health, /docs, and websocket stream before traffic confirmation.

### Frontend rollout
1. Build frontend dist.
2. Sync dist to S3 bucket.
3. Invalidate CloudFront.

## 6. Rollback Procedures
### Backend rollback
- Pull previous known-good image tag on EC2 host.
- Recreate backend container with the previous image tag.

### Frontend rollback
- Re-sync previous build artifact to S3.
- Invalidate CloudFront.

### Model rollback
- Update manifest latest.json active_model_s3_uri to previous model version path.

## 7. Incident Response
### P1: API unavailable (5xx spike)
1. Check EC2 instance status checks and security group ingress.
2. Check backend container status and logs (`docker ps`, `docker logs aetherforecast-backend`).
3. Check Caddy runtime (`docker exec aetherforecast-backend caddy validate --config /etc/caddy/Caddyfile`).
4. Roll back to previous image tag if needed.
5. Verify /health and API latency recovery.

### P1: Cron pipeline failure
1. Check host cron status and latest logs (`systemctl status crond`, `tail -n 200 /var/log/aetherforecast-cron.log`).
2. Verify DATA_BUCKET and EC2 IAM permissions.
3. Trigger one manual cron run (`/usr/local/bin/aetherforecast-fetch-cron.sh`) to validate recovery.

### P1: Training failures
1. Check Batch job logs and queue state.
2. Confirm Spot capacity and checkpoint writes.
3. Relaunch with conservative hyperparameters if needed.

## 8. Logging Standards
- Backend logs are JSON structured via structlog.
- Batch training log group is configured for 30-day retention.
- EC2 host-level cron logs are in `/var/log/aetherforecast-cron.log`.
- EC2 backend host + container logs are shipped to CloudWatch Logs group `/aetherforecast/ec2-backend` with 30-day retention.

## 9. Monitoring URLs
Use AWS Console links with your region:
- CloudWatch Dashboard: https://<region>.console.aws.amazon.com/cloudwatch/home?region=<region>#dashboards:name=AetherForecast-Operations
- CloudWatch Alarms: https://<region>.console.aws.amazon.com/cloudwatch/home?region=<region>#alarmsV2:

## 10. Useful Commands
### Trigger cron task manually on EC2
```bash
sudo /usr/local/bin/aetherforecast-fetch-cron.sh
```

### Check backend container health on EC2
```bash
docker ps
docker logs --tail 200 aetherforecast-backend
curl -fsS http://127.0.0.1/health
```

### Trigger manual SSM rollout (if needed)
```bash
aws ssm send-command --document-name AWS-RunShellScript --instance-ids <ec2-instance-id> --parameters commands='["docker ps","curl -fsS http://127.0.0.1/health"]'
```

### Trigger training job manually
```bash
aws batch submit-job --job-name af-train-manual --job-queue aetherforecast-training-queue --job-definition aetherforecast-training-job
```

### Check promoted model
```bash
aws s3 cp s3://<model-bucket>/chronos-v1/model/manifest/latest.json -
```
