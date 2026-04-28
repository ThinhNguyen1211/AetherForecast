# AetherForecast Handover

## 1. Project Summary
AetherForecast is a full-stack production-ready forecasting platform for crypto and gold-market style time-series with:
- AWS CDK infrastructure (network, storage, compute, auth, frontend hosting, monitoring)
- FastAPI backend with Cognito JWT protection
- Realtime WebSocket market updates
- Partitioned S3 parquet data lake ingestion
- Hugging Face-based forecasting inference
- AWS Batch Spot GPU LoRA training and S3 manifest-based promotion
- React + Vite trading dashboard UI
- GitHub Actions CI/CD for backend, frontend, and infrastructure

## 2. Architecture Diagram (Text)
```text
[User Browser]
   |  HTTPS
   v
[CloudFront] --> [S3 Frontend Bucket]
   |
   v
[EC2 t3.micro + Docker]
   |
   v
[Caddy -> FastAPI Service] <--> [Cognito JWT Validation]
   |           |\
   |           | \--> [WebSocket Binance Stream]
   |           \----> [S3 Model Manifest + Artifacts]
   |
   \----> [/predict inference]

[EC2 Host Cron (*/30min)]
   -> [Container cronjob.sh: data fetch]
   -> [S3 Parquet partitioned lake]

[AWS Batch Spot GPU]
   -> [LoRA training + checkpointing]
   -> [S3 model version]
   -> [Manifest promotion]

[CloudWatch Dashboard + Alarms] -> [SNS Notifications]
```

## 3. Cost Estimate (Low Traffic, Approx)
Estimated monthly range in ap-southeast-1 with low traffic and periodic training:
- EC2 backend (t3.micro + EBS + transfer): $12-$35
- S3 (frontend + parquet + models): $10-$40
- CloudFront: $5-$20
- CloudWatch logs/metrics/alarms: $10-$30
- AWS Batch Spot GPU training (occasional): $20-$120

Total estimated monthly range: $65-$315 depending on traffic and training frequency.

## 4. Retraining Procedure
1. Push latest training image to ECR.
2. Submit Batch job to aetherforecast-training-queue.
3. Monitor logs and job status.
4. Verify manifest update in:
   - s3://<model-bucket>/chronos-v1/model/manifest/latest.json
5. Validate /predict output in API.

## 5. Rollback Strategy
- API rollback: redeploy previous backend image tag on EC2 container.
- Frontend rollback: redeploy previous dist artifact to S3 + CloudFront invalidation.
- Model rollback: update manifest active_model_s3_uri to prior version.
- Infra rollback: CDK deploy previous known commit.

## 6. Incident Response Model
- Severity 1: production outage, sustained 5xx, no ingest, or failed model serving.
- Severity 2: degraded latency, intermittent cron or batch failure.
- Severity 3: minor UI/data lag with no customer-impact outage.

For each incident:
1. Detect via alarms/dashboard.
2. Triage and isolate component.
3. Mitigate (rollback/restart/redeploy).
4. Validate recovery via checklist.
5. Capture postmortem and preventive action.

## 7. Ownership And Handover Notes
- Recommended owners:
  - Platform/Infra owner
  - Backend/API owner
  - ML training/inference owner
  - Frontend owner
- Keep runbook and env-var docs updated after every production change.
