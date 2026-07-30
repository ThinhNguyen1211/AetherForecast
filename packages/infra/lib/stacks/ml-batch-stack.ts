import { Stack } from "aws-cdk-lib";
import * as batch from "aws-cdk-lib/aws-batch";
import * as ec2 from "aws-cdk-lib/aws-ec2";
import * as ecr from "aws-cdk-lib/aws-ecr";
import * as events from "aws-cdk-lib/aws-events";
import * as iam from "aws-cdk-lib/aws-iam";
import * as logs from "aws-cdk-lib/aws-logs";
import * as s3 from "aws-cdk-lib/aws-s3";
import { Construct } from "constructs";

export interface MlBatchStackProps {
  readonly vpc: ec2.IVpc;
  readonly trainingRepository: ecr.IRepository;
  readonly trainingImageUri: string;
  readonly parquetDataBucket: s3.IBucket;
  readonly mlModelBucket: s3.IBucket;
}

export class MlBatchStack extends Construct {
  public readonly computeEnvironment: batch.CfnComputeEnvironment;
  public readonly jobQueue: batch.CfnJobQueue;
  public readonly jobDefinition: batch.CfnJobDefinition;
  public readonly jobQueueName: string;
  public readonly trainingLogGroup: logs.LogGroup;
  public readonly monthlyScheduleRule: events.CfnRule;
  public readonly monthlyScheduleRuleName: string;

  constructor(scope: Construct, id: string, props: MlBatchStackProps) {
    super(scope, id);

    const batchSecurityGroup = new ec2.SecurityGroup(this, "BatchSecurityGroup", {
      vpc: props.vpc,
      allowAllOutbound: true,
      description: "Security group for AWS Batch GPU compute",
    });

    const batchServiceRole = new iam.Role(this, "BatchServiceRole", {
      assumedBy: new iam.ServicePrincipal("batch.amazonaws.com"),
      managedPolicies: [
        iam.ManagedPolicy.fromAwsManagedPolicyName("service-role/AWSBatchServiceRole"),
      ],
    });

    const ecsInstanceRole = new iam.Role(this, "BatchInstanceRole", {
      assumedBy: new iam.ServicePrincipal("ec2.amazonaws.com"),
      managedPolicies: [
        iam.ManagedPolicy.fromAwsManagedPolicyName(
          "service-role/AmazonEC2ContainerServiceforEC2Role",
        ),
      ],
    });

    const instanceProfile = new iam.CfnInstanceProfile(this, "BatchInstanceProfile", {
      roles: [ecsInstanceRole.roleName],
      instanceProfileName: "aetherforecast-batch-instance-profile",
    });

    const trainingJobRole = new iam.Role(this, "TrainingJobRole", {
      assumedBy: new iam.ServicePrincipal("ecs-tasks.amazonaws.com"),
      description: "Role used by training containers at runtime",
    });

    trainingJobRole.addToPolicy(
      new iam.PolicyStatement({
        actions: ["s3:ListBucket"],
        resources: [props.parquetDataBucket.bucketArn, props.mlModelBucket.bucketArn],
      }),
    );

    trainingJobRole.addToPolicy(
      new iam.PolicyStatement({
        actions: ["s3:GetObject"],
        resources: [`${props.parquetDataBucket.bucketArn}/*`],
      }),
    );

    trainingJobRole.addToPolicy(
      new iam.PolicyStatement({
        actions: ["s3:GetObject", "s3:PutObject", "s3:DeleteObject"],
        resources: [`${props.mlModelBucket.bucketArn}/*`],
      }),
    );

    const trainingExecutionRole = new iam.Role(this, "TrainingExecutionRole", {
      assumedBy: new iam.ServicePrincipal("ecs-tasks.amazonaws.com"),
      managedPolicies: [
        iam.ManagedPolicy.fromAwsManagedPolicyName(
          "service-role/AmazonECSTaskExecutionRolePolicy",
        ),
      ],
      description: "Execution role for pulling training images and writing logs",
    });

    props.trainingRepository.grantPull(trainingExecutionRole);

    this.trainingLogGroup = new logs.LogGroup(this, "BatchTrainingLogGroup", {
      retention: logs.RetentionDays.ONE_MONTH,
    });

    const privateSubnetIds = props.vpc
      .selectSubnets({ subnetType: ec2.SubnetType.PUBLIC })
      .subnetIds;

    this.computeEnvironment = new batch.CfnComputeEnvironment(this, "TrainingSpotComputeEnvironment", {
      computeEnvironmentName: "aetherforecast-training-ce-spot",
      type: "MANAGED",
      serviceRole: batchServiceRole.roleArn,
      state: "ENABLED",
      computeResources: {
        type: "SPOT",
        allocationStrategy: "SPOT_CAPACITY_OPTIMIZED",
        minvCpus: 0,
        desiredvCpus: 0,
        // Capped well below the previous 128 — a single training job only
        // needs 8 vCPUs (one g4dn.2xlarge); this still allows a few
        // concurrent/retry jobs while bounding worst-case runaway cost.
        maxvCpus: 32,
        bidPercentage: 100,
        instanceRole: instanceProfile.attrArn,
        instanceTypes: ["g4dn.2xlarge"],
        securityGroupIds: [batchSecurityGroup.securityGroupId],
        subnets: privateSubnetIds,
        ec2Configuration: [{ imageType: "ECS_AL2_NVIDIA" }],
        tags: {
          Workload: "training",
        },
      },
    });

    this.jobQueueName = "aetherforecast-training-queue";

    this.jobQueue = new batch.CfnJobQueue(this, "TrainingJobQueue", {
      jobQueueName: this.jobQueueName,
      priority: 1,
      state: "ENABLED",
      // Spot-only per explicit user directive: no On-Demand fallback, to
      // eliminate any risk of runaway On-Demand GPU cost. Tradeoff accepted:
      // if Spot capacity is unavailable for g4dn.2xlarge, a scheduled run
      // will stay queued until capacity frees up rather than falling back.
      computeEnvironmentOrder: [
        {
          order: 1,
          computeEnvironment: this.computeEnvironment.ref,
        },
      ],
    });

    this.jobDefinition = new batch.CfnJobDefinition(this, "TrainingJobDefinition", {
      jobDefinitionName: "aetherforecast-training-job",
      type: "container",
      platformCapabilities: ["EC2"],
      retryStrategy: { attempts: 2 },
      timeout: { attemptDurationSeconds: 60 * 60 * 24 },
      containerProperties: {
        image: props.trainingImageUri,
        command: ["/app/entrypoint.sh"],
        executionRoleArn: trainingExecutionRole.roleArn,
        jobRoleArn: trainingJobRole.roleArn,
        resourceRequirements: [
          { type: "VCPU", value: "8" },
          { type: "MEMORY", value: "30720" },
          { type: "GPU", value: "1" },
        ],
        environment: [
          { name: "TRAIN_MODE", value: "true" },
          { name: "AWS_REGION", value: Stack.of(this).region },
          { name: "DATA_S3_BUCKET", value: props.parquetDataBucket.bucketName },
          { name: "DATA_BUCKET", value: props.parquetDataBucket.bucketName },
          {
            name: "MODEL_S3_URI",
            value: `s3://${props.mlModelBucket.bucketName}/chronos-v1/model/`,
          },
          { name: "MODEL_BUCKET", value: props.mlModelBucket.bucketName },
          {
            name: "MODEL_OUTPUT_PREFIX",
            value: "chronos/models/",
          },
          {
            name: "CHECKPOINT_S3_URI",
            value: `s3://${props.mlModelBucket.bucketName}/checkpoints/`,
          },
          { name: "SYMBOLS", value: "BTCUSDT,ETHUSDT,SOLUSDT" },
          { name: "TIMEFRAME", value: "1h" },
          { name: "TRAINING_HORIZON", value: "7" },
          { name: "CONTEXT_LENGTH", value: "96" },
          { name: "MAX_ROWS_PER_SYMBOL", value: "20000" },
          { name: "EPOCHS", value: "3" },
          { name: "LEARNING_RATE", value: "0.0002" },
          { name: "BATCH_SIZE", value: "4" },
          { name: "GRAD_ACCUM_STEPS", value: "8" },
          { name: "WARMUP_RATIO", value: "0.03" },
          { name: "WEIGHT_DECAY", value: "0.01" },
          { name: "SAVE_STEPS", value: "100" },
          { name: "EVAL_STEPS", value: "100" },
          { name: "LOGGING_STEPS", value: "20" },
          { name: "LORA_R", value: "16" },
          { name: "LORA_ALPHA", value: "32" },
          { name: "LORA_DROPOUT", value: "0.05" },
          { name: "MAX_SEQ_LENGTH", value: "512" },
          { name: "BASE_MODEL_ID", value: "amazon/chronos-2" },
          { name: "BASE_MODEL_FALLBACK_ID", value: "amazon/chronos-t5-large" },
        ],
        logConfiguration: {
          logDriver: "awslogs",
          options: {
            "awslogs-group": this.trainingLogGroup.logGroupName,
            "awslogs-region": Stack.of(this).region,
            "awslogs-stream-prefix": "training",
          },
        },
      },
    });

    this.jobQueue.addDependency(this.computeEnvironment);

    // --- Monthly automated trigger ---
    // Nothing previously scheduled this job at all; scheduler-stack.ts is a
    // separate, unrelated 30-minute market-data-fetch cron. Hand-rolled with
    // CfnRule (L1) rather than the aws-events-targets.BatchJob L2 helper, to
    // stay consistent with this file's existing all-L1 Batch resources
    // (CfnComputeEnvironment/CfnJobQueue/CfnJobDefinition) and avoid an
    // untested L1/L2 interop assumption.
    const eventBridgeBatchRole = new iam.Role(this, "TrainingScheduleInvokeRole", {
      assumedBy: new iam.ServicePrincipal("events.amazonaws.com"),
      description: "Allows EventBridge to submit the monthly fine-tuning job to AWS Batch",
    });

    eventBridgeBatchRole.addToPolicy(
      new iam.PolicyStatement({
        actions: ["batch:SubmitJob"],
        resources: [this.jobQueue.attrJobQueueArn, this.jobDefinition.attrJobDefinitionArn],
      }),
    );

    this.monthlyScheduleRuleName = "aetherforecast-training-monthly-schedule";

    // cron(0 0 ? * 1L *) = last Sunday of every month at 00:00 UTC (AWS cron:
    // day-of-week 1=Sunday, "1L" = last occurrence of that weekday in the month).
    // Deliberately NOT a fixed rate(28 days) — that drifts across weekdays
    // over time instead of landing on the same calendar anchor each month.
    this.monthlyScheduleRule = new events.CfnRule(this, "MonthlyTrainingScheduleRule", {
      name: this.monthlyScheduleRuleName,
      description:
        "Triggers the AWS Batch Chronos-2 fine-tuning job on the last Sunday of every month at 00:00 UTC",
      scheduleExpression: "cron(0 0 ? * 1L *)",
      // Starts DISABLED — this is new, cost-incurring automation that hasn't
      // been through a manual test run yet. Flip to "ENABLED" once you've
      // confirmed a manual `batch submit-job` against this queue/definition
      // succeeds end-to-end.
      state: "DISABLED",
      targets: [
        {
          id: "MonthlyFineTuneBatchTarget",
          arn: this.jobQueue.attrJobQueueArn,
          roleArn: eventBridgeBatchRole.roleArn,
          batchParameters: {
            jobDefinition: this.jobDefinition.attrJobDefinitionArn,
            jobName: "aetherforecast-monthly-finetune",
          },
        },
      ],
    });

    this.monthlyScheduleRule.addDependency(this.jobQueue);
    this.monthlyScheduleRule.addDependency(this.jobDefinition);
  }
}
