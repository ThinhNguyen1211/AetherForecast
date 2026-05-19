---
trigger: always_on
---

# AWS Infrastructure & DevOps Rules

## Core Principles
- Infrastructure as Code first (AWS CDK TypeScript).
- Least privilege everywhere (IAM roles, security groups).
- Cost optimization: Spot cho train, giảm S3 requests (cron 30p+), scale-to-zero khi có thể.
- Observability: structured logs (structlog), CloudWatch alarms + dashboards, X-Ray nếu cần.

## CDK Best Practices
- Stack tách biệt: Network, Storage, Compute, ML-Batch, Scheduler, Auth, Monitoring.
- Resource naming consistent (resource-naming.ts).
- Tags global: Project, Environment, Owner.
- Deterministic synth & deploy.

## Backend & Container
- FastAPI + Caddy reverse proxy trên EC2 hoặc Fargate.
- Docker multi-stage, slim image, không chứa training deps ở runtime.
- Cronjob trên EC2/host cron hoặc EventBridge.

## ML Training Rules
- Train trên GPU (RTX 3050 hoặc g4dn), batch nhỏ, LoRA rank vừa phải.
- Data từ S3 Parquet partitioned.
- Promote model qua manifest/latest.json để zero-downtime update.

## Security & Compliance
- Không hardcode secret, không commit .env.
- Scan secret, dependency, Docker image (Trivy, Bandit).
- Cognito cho auth, JWT ngắn hạn hoặc refresh token hợp lý.

## CI/CD
- GitHub Actions + OIDC.
- Matrix test, lint/type/security scan trước merge.
- Manual approval cho production deploy.