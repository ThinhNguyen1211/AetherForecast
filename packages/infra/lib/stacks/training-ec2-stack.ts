import { CfnParameter, RemovalPolicy, Stack } from "aws-cdk-lib";
import * as ec2 from "aws-cdk-lib/aws-ec2";
import * as ecr from "aws-cdk-lib/aws-ecr";
import * as iam from "aws-cdk-lib/aws-iam";
import * as logs from "aws-cdk-lib/aws-logs";
import * as s3 from "aws-cdk-lib/aws-s3";
import { Construct } from "constructs";

export interface TrainingEc2StackProps {
  readonly vpc: ec2.IVpc;
  readonly trainingRepository: ecr.IRepository;
  readonly parquetDataBucket: s3.IBucket;
  readonly mlModelBucket: s3.IBucket;
}

export class TrainingEc2Stack extends Construct {
  public readonly instance: ec2.Instance;
  public readonly securityGroup: ec2.SecurityGroup;
  public readonly trainingLogGroup: logs.LogGroup;
  public readonly elasticIp: ec2.CfnEIP;

  constructor(scope: Construct, id: string, props: TrainingEc2StackProps) {
    super(scope, id);

    const instanceTypeParam = new CfnParameter(this, "TrainingEc2InstanceType", {
      type: "String",
      default: "g4dn.xlarge",
      description: "Dedicated EC2 instance type for training workloads.",
    });
    instanceTypeParam.overrideLogicalId("TrainingEc2InstanceType");

    const rootVolumeGiBParam = new CfnParameter(this, "TrainingEc2RootVolumeGiB", {
      type: "Number",
      default: 150,
      minValue: 50,
      maxValue: 500,
      description: "Root EBS volume size in GiB for dedicated training instance.",
    });
    rootVolumeGiBParam.overrideLogicalId("TrainingEc2RootVolumeGiB");

    const allowIngressCidrParam = new CfnParameter(this, "TrainingEc2AllowIngressCidr", {
      type: "String",
      default: "0.0.0.0/0",
      description: "CIDR allowed to access training host 22 and 443 for temporary operational access.",
    });
    allowIngressCidrParam.overrideLogicalId("TrainingEc2AllowIngressCidr");

    this.securityGroup = new ec2.SecurityGroup(this, "TrainingEc2SecurityGroup", {
      vpc: props.vpc,
      allowAllOutbound: true,
      description: "Security group for dedicated EC2 GPU training host",
    });

    this.securityGroup.addIngressRule(
      ec2.Peer.ipv4(allowIngressCidrParam.valueAsString),
      ec2.Port.tcp(22),
      "Temporary SSH access to training host",
    );

    this.securityGroup.addIngressRule(
      ec2.Peer.ipv4(allowIngressCidrParam.valueAsString),
      ec2.Port.tcp(443),
      "Temporary HTTPS access to training host",
    );

    this.trainingLogGroup = new logs.LogGroup(this, "TrainingEc2LogGroup", {
      retention: logs.RetentionDays.ONE_MONTH,
      removalPolicy: RemovalPolicy.RETAIN,
    });

    const role = new iam.Role(this, "TrainingEc2Role", {
      assumedBy: new iam.ServicePrincipal("ec2.amazonaws.com"),
      description: "Role for training EC2 host with S3, CloudWatch, ECR, and SSM access",
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
        actions: ["s3:*"],
        resources: [props.parquetDataBucket.bucketArn, props.mlModelBucket.bucketArn],
      }),
    );

    role.addToPolicy(
      new iam.PolicyStatement({
        actions: ["s3:*"],
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

    props.trainingRepository.grantPull(role);
    this.trainingLogGroup.grantWrite(role);

    const machineImage = ec2.MachineImage.fromSsmParameter(
      "/aws/service/ecs/optimized-ami/amazon-linux-2023/gpu/recommended/image_id",
      {
        os: ec2.OperatingSystemType.LINUX,
      },
    );

    this.instance = new ec2.Instance(this, "TrainingEc2Instance", {
      vpc: props.vpc,
      vpcSubnets: { subnetType: ec2.SubnetType.PUBLIC },
      instanceType: new ec2.InstanceType(instanceTypeParam.valueAsString),
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

    const stackRegion = Stack.of(this).region;

    this.instance.userData.addCommands(
      "#!/bin/bash",
      "set -euxo pipefail",
      "dnf update -y",
      "dnf install -y docker jq awscli amazon-cloudwatch-agent amazon-ssm-agent",
      "systemctl enable amazon-ssm-agent",
      "systemctl start amazon-ssm-agent",
      "systemctl enable docker",
      "systemctl start docker",
      "mkdir -p /opt/aetherforecast-training /var/log/aetherforecast-training /var/lib/aetherforecast-training",
      "mkdir -p /var/lib/aetherforecast-training/hf-home /var/lib/aetherforecast-training/hf-cache",
      "touch /var/log/aetherforecast-training/training.log",
      `cat > /opt/aws/amazon-cloudwatch-agent/etc/amazon-cloudwatch-agent.json <<'EOF'\n{\n  "agent": {\n    "run_as_user": "root"\n  },\n  "logs": {\n    "logs_collected": {\n      "files": {\n        "collect_list": [\n          {\n            "file_path": "/var/log/aetherforecast-training/training.log",\n            "log_group_name": "${this.trainingLogGroup.logGroupName}",\n            "log_stream_name": "{instance_id}/training"\n          }\n        ]\n      }\n    }\n  }\n}\nEOF`,
      "/opt/aws/amazon-cloudwatch-agent/bin/amazon-cloudwatch-agent-ctl -a stop || true",
      "systemctl enable amazon-cloudwatch-agent",
      "/opt/aws/amazon-cloudwatch-agent/bin/amazon-cloudwatch-agent-ctl -a fetch-config -m ec2 -c file:/opt/aws/amazon-cloudwatch-agent/etc/amazon-cloudwatch-agent.json -s",
      "cat > /opt/aetherforecast-training/run-training.sh <<'EOF'\n#!/bin/bash\nset -euo pipefail\n\n: \"${TRAIN_IMAGE_URI:?TRAIN_IMAGE_URI is required}\"\n: \"${AWS_REGION:?AWS_REGION is required}\"\n: \"${DATA_S3_BUCKET:?DATA_S3_BUCKET is required}\"\n: \"${MODEL_S3_URI:?MODEL_S3_URI is required}\"\n: \"${CHECKPOINT_S3_URI:?CHECKPOINT_S3_URI is required}\"\n\nLOG_FILE=/var/log/aetherforecast-training/training.log\nmkdir -p \"$(dirname \"$LOG_FILE\")\"\ntouch \"$LOG_FILE\"\n\nECR_REGISTRY=$(echo \"$TRAIN_IMAGE_URI\" | cut -d'/' -f1)\nif echo \"$ECR_REGISTRY\" | grep -q '\\.dkr\\.ecr\\.'; then\n  aws ecr get-login-password --region \"$AWS_REGION\" | docker login --username AWS --password-stdin \"$ECR_REGISTRY\"\nfi\n\ndocker pull \"$TRAIN_IMAGE_URI\"\ndocker rm -f aetherforecast-training-run >/dev/null 2>&1 || true\n\ndocker run --name aetherforecast-training-run --rm --gpus all \\\n  -e TRAIN_MODE=true \\\n  -e AWS_REGION=\"$AWS_REGION\" \\\n  -e DATA_S3_BUCKET=\"$DATA_S3_BUCKET\" \\\n  -e DATA_BUCKET=\"$DATA_S3_BUCKET\" \\\n  -e MODEL_S3_URI=\"$MODEL_S3_URI\" \\\n  -e CHECKPOINT_S3_URI=\"$CHECKPOINT_S3_URI\" \\\n  -e SYMBOLS=\"${SYMBOLS:-BTCUSDT,ETHUSDT,SOLUSDT}\" \\\n  -e TIMEFRAME=\"${TIMEFRAME:-1h}\" \\\n  -e TRAINING_HORIZON=\"${TRAINING_HORIZON:-7}\" \\\n  -e CONTEXT_LENGTH=\"${CONTEXT_LENGTH:-96}\" \\\n  -e MAX_ROWS_PER_SYMBOL=\"${MAX_ROWS_PER_SYMBOL:-20000}\" \\\n  -e EPOCHS=\"${EPOCHS:-3}\" \\\n  -e LEARNING_RATE=\"${LEARNING_RATE:-0.0002}\" \\\n  -e BATCH_SIZE=\"${BATCH_SIZE:-4}\" \\\n  -e GRAD_ACCUM_STEPS=\"${GRAD_ACCUM_STEPS:-8}\" \\\n  -e WARMUP_RATIO=\"${WARMUP_RATIO:-0.03}\" \\\n  -e WEIGHT_DECAY=\"${WEIGHT_DECAY:-0.01}\" \\\n  -e SAVE_STEPS=\"${SAVE_STEPS:-100}\" \\\n  -e EVAL_STEPS=\"${EVAL_STEPS:-100}\" \\\n  -e LOGGING_STEPS=\"${LOGGING_STEPS:-20}\" \\\n  -e LORA_R=\"${LORA_R:-16}\" \\\n  -e LORA_ALPHA=\"${LORA_ALPHA:-32}\" \\\n  -e LORA_DROPOUT=\"${LORA_DROPOUT:-0.05}\" \\\n  -e MAX_SEQ_LENGTH=\"${MAX_SEQ_LENGTH:-512}\" \\\n  -e BASE_MODEL_ID=\"${BASE_MODEL_ID:-amazon/chronos-2}\" \\\n  -e BASE_MODEL_FALLBACK_ID=\"${BASE_MODEL_FALLBACK_ID:-amazon/chronos-t5-large}\" \\\n  -e HF_HOME=\"/var/lib/aetherforecast-training/hf-home\" \\\n  -e HF_CACHE_DIR=\"/var/lib/aetherforecast-training/hf-cache\" \\\n  \"$TRAIN_IMAGE_URI\" 2>&1 | tee -a \"$LOG_FILE\"\nEOF",
      "chmod +x /opt/aetherforecast-training/run-training.sh",
    );

    this.elasticIp = new ec2.CfnEIP(this, "TrainingEc2ElasticIp", {
      domain: "vpc",
    });

    new ec2.CfnEIPAssociation(this, "TrainingEc2EipAssociation", {
      allocationId: this.elasticIp.attrAllocationId,
      instanceId: this.instance.instanceId,
    });
  }
}