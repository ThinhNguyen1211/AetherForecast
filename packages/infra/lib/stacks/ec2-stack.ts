import { CfnOutput, CfnParameter, RemovalPolicy, Stack } from "aws-cdk-lib";
import * as ec2 from "aws-cdk-lib/aws-ec2";
import * as iam from "aws-cdk-lib/aws-iam";
import * as logs from "aws-cdk-lib/aws-logs";
import * as s3 from "aws-cdk-lib/aws-s3";
import { Construct } from "constructs";

export interface Ec2StackProps {
  readonly vpc: ec2.IVpc;
  readonly parquetDataBucket: s3.IBucket;
  readonly mlModelBucket: s3.IBucket;
  readonly cognitoUserPoolId: string;
  readonly cognitoClientId: string;
}

export class Ec2Stack extends Construct {
  public readonly instance: ec2.Instance;
  public readonly securityGroup: ec2.SecurityGroup;
  public readonly elasticIp: ec2.CfnEIP;
  public readonly backendLogGroup: logs.LogGroup;

  constructor(scope: Construct, id: string, props: Ec2StackProps) {
    super(scope, id);

    const backendImageUriParam = new CfnParameter(this, "BackendImageUri", {
      type: "String",
      default: "ghcr.io/aetherforecast/backend:latest",
      description: "Container image URI for backend (must include Caddy + FastAPI).",
    });
    backendImageUriParam.overrideLogicalId("BackendImageUri");

    const apiDomainParam = new CfnParameter(this, "ApiDomainName", {
      type: "String",
      default: ":80",
      description: "Caddy site address (:80 for HTTP-only bootstrap, or api.example.com for automatic HTTPS).",
    });
    apiDomainParam.overrideLogicalId("ApiDomainName");

    const letsEncryptEmailParam = new CfnParameter(this, "LetsEncryptEmail", {
      type: "String",
      default: "ops@example.com",
      description: "Email used for Let's Encrypt registration in Caddy.",
    });
    letsEncryptEmailParam.overrideLogicalId("LetsEncryptEmail");

    const corsOriginsParam = new CfnParameter(this, "CorsOrigins", {
      type: "String",
      default: "*",
      description: "Comma-separated CORS origins for backend API.",
    });
    corsOriginsParam.overrideLogicalId("CorsOrigins");

    const allowCidrParam = new CfnParameter(this, "AllowCidr", {
      type: "String",
      default: "0.0.0.0/0",
      description:
        "CIDR allowed to access backend 80/443. 0.0.0.0/0 is temporary for testing.",
    });
    allowCidrParam.overrideLogicalId("AllowCidr");

    const ec2InstanceTypeParam = new CfnParameter(this, "BackendEc2InstanceType", {
      type: "String",
      default: "t4g.small",
      description: "EC2 instance type for backend host. t4g.small (Arm64) is the default for cost-efficient API + cron.",
    });
    ec2InstanceTypeParam.overrideLogicalId("BackendEc2InstanceType");

    const rootVolumeGiBParam = new CfnParameter(this, "BackendRootVolumeGiB", {
      type: "Number",
      default: 40,
      minValue: 20,
      maxValue: 200,
      description: "Root EBS size in GiB for backend host.",
    });
    rootVolumeGiBParam.overrideLogicalId("BackendRootVolumeGiB");

    this.securityGroup = new ec2.SecurityGroup(this, "BackendEc2SecurityGroup", {
      vpc: props.vpc,
      allowAllOutbound: true,
      description: "Security group for single EC2 backend host",
    });

    this.securityGroup.addIngressRule(
      ec2.Peer.ipv4(allowCidrParam.valueAsString),
      ec2.Port.tcp(80),
      "Allow HTTP from AllowCidr (temporary for testing)",
    );
    this.securityGroup.addIngressRule(
      ec2.Peer.ipv4(allowCidrParam.valueAsString),
      ec2.Port.tcp(443),
      "Allow HTTPS from AllowCidr (temporary for testing)",
    );

    this.backendLogGroup = new logs.LogGroup(this, "Ec2BackendLogGroup", {
      logGroupName: "/aetherforecast/ec2-backend",
      retention: logs.RetentionDays.ONE_MONTH,
      removalPolicy: RemovalPolicy.RETAIN,
    });

    const role = new iam.Role(this, "BackendEc2Role", {
      assumedBy: new iam.ServicePrincipal("ec2.amazonaws.com"),
      description: "EC2 role for backend host with S3, CloudWatch, Batch and SSM access",
    });

    role.addManagedPolicy(
      iam.ManagedPolicy.fromAwsManagedPolicyName("AmazonSSMManagedInstanceCore"),
    );
    role.addManagedPolicy(
      iam.ManagedPolicy.fromAwsManagedPolicyName("CloudWatchAgentServerPolicy"),
    );
    role.addManagedPolicy(
      iam.ManagedPolicy.fromAwsManagedPolicyName("AmazonEC2ContainerRegistryReadOnly"),
    );

    role.addToPolicy(
      new iam.PolicyStatement({
        actions: [
          "batch:SubmitJob",
          "batch:DescribeJobs",
          "batch:ListJobs",
          "batch:DescribeJobDefinitions",
          "batch:DescribeJobQueues",
        ],
        resources: ["*"],
      }),
    );

    role.addToPolicy(
      new iam.PolicyStatement({
        actions: ["s3:ListBucket"],
        resources: [props.parquetDataBucket.bucketArn, props.mlModelBucket.bucketArn],
      }),
    );

    role.addToPolicy(
      new iam.PolicyStatement({
        actions: ["s3:GetObject", "s3:PutObject", "s3:DeleteObject"],
        resources: [
          `${props.parquetDataBucket.bucketArn}/*`,
          `${props.mlModelBucket.bucketArn}/*`,
        ],
      }),
    );

    role.addToPolicy(
      new iam.PolicyStatement({
        actions: ["cloudwatch:PutMetricData"],
        resources: ["*"],
      }),
    );

    this.backendLogGroup.grantWrite(role);

    const stackRegion = Stack.of(this).region;

    const machineImage = ec2.MachineImage.latestAmazonLinux2023({
      cpuType: ec2.AmazonLinuxCpuType.ARM_64,
    });

    this.instance = new ec2.Instance(this, "BackendInstance", {
      vpc: props.vpc,
      vpcSubnets: { subnetType: ec2.SubnetType.PUBLIC },
      instanceType: new ec2.InstanceType(ec2InstanceTypeParam.valueAsString),
      machineImage,
      securityGroup: this.securityGroup,
      role,
      detailedMonitoring: true,
      blockDevices: [
        {
          deviceName: "/dev/xvda",
          volume: ec2.BlockDeviceVolume.ebs(rootVolumeGiBParam.valueAsNumber, {
            encrypted: true,
            deleteOnTermination: true,
          }),
        },
      ],
    });

    this.instance.userData.addCommands(
      "#!/bin/bash",
      "set -euxo pipefail",
      "dnf update -y",
      "dnf install -y docker cronie jq util-linux amazon-cloudwatch-agent amazon-ssm-agent awscli",
      "systemctl enable amazon-ssm-agent",
      "systemctl start amazon-ssm-agent",
      "mkdir -p /etc/docker",
      "cat > /etc/docker/daemon.json <<'EOF'\n{\n  \"log-driver\": \"json-file\",\n  \"log-opts\": {\n    \"max-size\": \"20m\",\n    \"max-file\": \"3\"\n  }\n}\nEOF",
      "systemctl enable docker",
      "systemctl start docker",
      "systemctl restart docker",
      "systemctl enable crond",
      "systemctl start crond",
      "mkdir -p /var/lib/aetherforecast",
      "touch /var/log/aetherforecast-cron.log /var/log/aetherforecast-health.log",
      "mkdir -p /opt/aetherforecast/caddy_data /opt/aetherforecast/caddy_config",
      "chmod -R 0777 /opt/aetherforecast/caddy_data /opt/aetherforecast/caddy_config",
      `cat > /opt/aws/amazon-cloudwatch-agent/etc/amazon-cloudwatch-agent.json <<'EOF'\n{\n  "agent": {\n    "run_as_user": "root"\n  },\n  "logs": {\n    "logs_collected": {\n      "files": {\n        "collect_list": [\n          {\n            "file_path": "/var/log/aetherforecast-cron.log",\n            "log_group_name": "${this.backendLogGroup.logGroupName}",\n            "log_stream_name": "{instance_id}/cron"\n          },\n          {\n            "file_path": "/var/log/aetherforecast-health.log",\n            "log_group_name": "${this.backendLogGroup.logGroupName}",\n            "log_stream_name": "{instance_id}/health"\n          },\n          {\n            "file_path": "/var/lib/docker/containers/*/*.log",\n            "log_group_name": "${this.backendLogGroup.logGroupName}",\n            "log_stream_name": "{instance_id}/docker"\n          }\n        ]\n      }\n    }\n  },\n  "metrics": {\n    "append_dimensions": {\n      "InstanceId": "\${aws:InstanceId}"\n    },\n    "metrics_collected": {\n      "cpu": {\n        "resources": ["*"],\n        "measurement": [\n          "cpu_usage_user",\n          "cpu_usage_system",\n          "cpu_usage_iowait"\n        ],\n        "totalcpu": true,\n        "metrics_collection_interval": 60\n      },\n      "mem": {\n        "measurement": ["mem_used_percent"],\n        "metrics_collection_interval": 60\n      }\n    }\n  }\n}\nEOF`,
      "/opt/aws/amazon-cloudwatch-agent/bin/amazon-cloudwatch-agent-ctl -a stop || true",
      "systemctl enable amazon-cloudwatch-agent",
      "/opt/aws/amazon-cloudwatch-agent/bin/amazon-cloudwatch-agent-ctl -a fetch-config -m ec2 -c file:/opt/aws/amazon-cloudwatch-agent/etc/amazon-cloudwatch-agent.json -s",
      `cat > /opt/aetherforecast/.env <<'EOF'\nAPP_ENV=prod\nLOG_LEVEL=INFO\nPORT=8000\nUVICORN_WORKERS=1\nAWS_REGION=${stackRegion}\nDATA_BUCKET=${props.parquetDataBucket.bucketName}\nMODEL_BUCKET=${props.mlModelBucket.bucketName}\nMODEL_S3_URI=s3://${props.mlModelBucket.bucketName}/chronos-v1/model/\nREQUIRE_S3_MODEL=true\nCOGNITO_USER_POOL_ID=${props.cognitoUserPoolId}\nCOGNITO_CLIENT_ID=${props.cognitoClientId}\nCOGNITO_REGION=${stackRegion}\nCORS_ORIGINS=${corsOriginsParam.valueAsString}\nAPP_DOMAIN=${apiDomainParam.valueAsString}\nLETSENCRYPT_EMAIL=${letsEncryptEmailParam.valueAsString}\nPARQUET_PREFIX=market/klines\nSYMBOLS=\nKLINE_INTERVAL=1m\nFETCH_CONCURRENCY=24\nFETCH_SYMBOL_LIMIT=0\nMAX_KLINE_PAGES=2\nBOOTSTRAP_LOOKBACK_MINUTES=180\nFETCH_LOOP_SECONDS=0\nSENTIMENT_MODE=simple\nEXTERNAL_SENTIMENT_ENABLED=true\nEOF`,
      `BACKEND_IMAGE_URI='${backendImageUriParam.valueAsString}'`,
      "ECR_REGISTRY=$(echo \"$BACKEND_IMAGE_URI\" | cut -d'/' -f1)",
      [
        "if echo \"$ECR_REGISTRY\" | grep -q '\\.dkr\\.ecr\\.'; then",
        `  aws ecr get-login-password --region ${stackRegion} | docker login --username AWS --password-stdin \"$ECR_REGISTRY\"`,
        "fi",
      ].join("\n"),
      "docker pull $BACKEND_IMAGE_URI",
      "docker rm -f aetherforecast-backend || true",
      [
        "docker run -d --name aetherforecast-backend --restart unless-stopped",
        "-p 80:80 -p 443:443",
        "--env-file /opt/aetherforecast/.env",
        "-v /opt/aetherforecast/caddy_data:/data",
        "-v /opt/aetherforecast/caddy_config:/config",
        "$BACKEND_IMAGE_URI",
      ].join(" "),
      "cat > /usr/local/bin/aetherforecast-fetch-cron.sh <<'EOF'\n#!/bin/bash\nset -euo pipefail\nif ! docker ps --format '{{.Names}}' | grep -q '^aetherforecast-backend$'; then\n  exit 1\nfi\nif flock -n /var/run/aetherforecast-fetch.lock docker exec aetherforecast-backend /app/cronjob.sh; then\n  date +%s > /var/lib/aetherforecast/cron-last-success\n  exit 0\nfi\nexit 1\nEOF",
      "chmod +x /usr/local/bin/aetherforecast-fetch-cron.sh",
      "cat > /etc/cron.d/aetherforecast-fetch <<'EOF'\n*/30 * * * * root /usr/local/bin/aetherforecast-fetch-cron.sh >> /var/log/aetherforecast-cron.log 2>&1\nEOF",
      "chmod 0644 /etc/cron.d/aetherforecast-fetch",
      `cat > /usr/local/bin/aetherforecast-health-metrics.sh <<'EOF'\n#!/bin/bash\nset -euo pipefail\nAWS_REGION="${stackRegion}"\nTOKEN=$(curl -sS -X PUT http://169.254.169.254/latest/api/token -H 'X-aws-ec2-metadata-token-ttl-seconds: 21600')\nINSTANCE_ID=$(curl -sS -H "X-aws-ec2-metadata-token: $TOKEN" http://169.254.169.254/latest/meta-data/instance-id)\nCONTAINER_HEALTH=0\nif docker ps --filter name=^aetherforecast-backend$ --format '{{.Names}}' | grep -q '^aetherforecast-backend$'; then\n  CONTAINER_HEALTH=1\nfi\nCRON_HEALTH=0\nCRON_AGE=999999\nif [ -f /var/lib/aetherforecast/cron-last-success ]; then\n  LAST_SUCCESS=$(cat /var/lib/aetherforecast/cron-last-success || echo 0)\n  NOW=$(date +%s)\n  CRON_AGE=$((NOW - LAST_SUCCESS))\n  if [ "$CRON_AGE" -le 1800 ]; then\n    CRON_HEALTH=1\n  fi\nfi\nMETRIC_JSON=$(jq -n --arg iid "$INSTANCE_ID" --argjson ch "$CONTAINER_HEALTH" --argjson crh "$CRON_HEALTH" --argjson cra "$CRON_AGE" '[{"MetricName":"ContainerHealthy","Dimensions":[{"Name":"InstanceId","Value":$iid}],"Value":$ch,"Unit":"Count"},{"MetricName":"CronHealthy","Dimensions":[{"Name":"InstanceId","Value":$iid}],"Value":$crh,"Unit":"Count"},{"MetricName":"CronLastSuccessAgeSeconds","Dimensions":[{"Name":"InstanceId","Value":$iid}],"Value":$cra,"Unit":"Seconds"}]')\naws cloudwatch put-metric-data --region "$AWS_REGION" --namespace AetherForecast/Host --metric-data "$METRIC_JSON"\nEOF`,
      "chmod +x /usr/local/bin/aetherforecast-health-metrics.sh",
      "cat > /etc/cron.d/aetherforecast-health-metrics <<'EOF'\n*/5 * * * * root /usr/local/bin/aetherforecast-health-metrics.sh >> /var/log/aetherforecast-health.log 2>&1\nEOF",
      "chmod 0644 /etc/cron.d/aetherforecast-health-metrics",
      "/usr/local/bin/aetherforecast-health-metrics.sh || true",
      "systemctl restart crond",
    );

    this.elasticIp = new ec2.CfnEIP(this, "BackendElasticIp", {
      domain: "vpc",
    });

    new ec2.CfnEIPAssociation(this, "BackendEipAssociation", {
      allocationId: this.elasticIp.attrAllocationId,
      instanceId: this.instance.instanceId,
    });

    new CfnOutput(this, "Ec2BackendInstanceId", {
      value: this.instance.instanceId,
    });

    new CfnOutput(this, "Ec2BackendElasticIp", {
      value: this.elasticIp.ref,
    });

    new CfnOutput(this, "Ec2BackendApiDomain", {
      value: apiDomainParam.valueAsString,
    });

    new CfnOutput(this, "Ec2BackendCorsOrigins", {
      value: corsOriginsParam.valueAsString,
    });

    new CfnOutput(this, "Ec2BackendAllowCidr", {
      value: allowCidrParam.valueAsString,
    });

    new CfnOutput(this, "Ec2BackendInstanceType", {
      value: ec2InstanceTypeParam.valueAsString,
    });

    new CfnOutput(this, "Ec2BackendRootVolumeGiB", {
      value: rootVolumeGiBParam.valueAsString,
    });

    new CfnOutput(this, "Ec2BackendLogGroupName", {
      value: this.backendLogGroup.logGroupName,
    });

    new CfnOutput(this, "Ec2BackendHealthCheckHint", {
      value: "https://<your-domain-or-eip>/health",
    });
  }
}
