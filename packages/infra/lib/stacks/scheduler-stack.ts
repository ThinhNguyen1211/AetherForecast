import * as ec2 from "aws-cdk-lib/aws-ec2";
import * as ecr from "aws-cdk-lib/aws-ecr";
import * as ecs from "aws-cdk-lib/aws-ecs";
import * as ecsPatterns from "aws-cdk-lib/aws-ecs-patterns";
import * as events from "aws-cdk-lib/aws-events";
import * as iam from "aws-cdk-lib/aws-iam";
import * as logs from "aws-cdk-lib/aws-logs";
import * as s3 from "aws-cdk-lib/aws-s3";
import { Construct } from "constructs";

export interface SchedulerStackProps {
  readonly vpc: ec2.IVpc;
  readonly cluster: ecs.ICluster;
  readonly backendRepository: ecr.IRepository;
  readonly parquetDataBucket: s3.IBucket;
}

export class SchedulerStack extends Construct {
  public readonly scheduledTask: ecsPatterns.ScheduledFargateTask;
  public readonly cronLogGroup: logs.LogGroup;
  public readonly scheduleRuleName: string;

  constructor(scope: Construct, id: string, props: SchedulerStackProps) {
    super(scope, id);

    this.cronLogGroup = new logs.LogGroup(this, "CronTaskLogGroup", {
      logGroupName: "/aws/ecs/aetherforecast/data-fetch-cron",
      retention: logs.RetentionDays.ONE_MONTH,
    });

    const securityGroup = new ec2.SecurityGroup(this, "CronTaskSecurityGroup", {
      vpc: props.vpc,
      allowAllOutbound: true,
      description: "Security group for scheduled data fetch task",
    });

    const taskRole = new iam.Role(this, "CronTaskRole", {
      assumedBy: new iam.ServicePrincipal("ecs-tasks.amazonaws.com"),
      description: "Role for scheduled market data fetch container",
    });

    taskRole.addToPolicy(
      new iam.PolicyStatement({
        actions: ["s3:ListBucket"],
        resources: [props.parquetDataBucket.bucketArn],
      }),
    );

    taskRole.addToPolicy(
      new iam.PolicyStatement({
        actions: ["s3:PutObject", "s3:GetObject"],
        resources: [`${props.parquetDataBucket.bucketArn}/*`],
      }),
    );

    const executionRole = new iam.Role(this, "CronExecutionRole", {
      assumedBy: new iam.ServicePrincipal("ecs-tasks.amazonaws.com"),
      managedPolicies: [
        iam.ManagedPolicy.fromAwsManagedPolicyName(
          "service-role/AmazonECSTaskExecutionRolePolicy",
        ),
      ],
      description: "Execution role for scheduled ECS task",
    });

    props.backendRepository.grantPull(executionRole);

    const taskDefinition = new ecs.FargateTaskDefinition(this, "CronTaskDefinition", {
      cpu: 512,
      memoryLimitMiB: 1024,
      taskRole,
      executionRole,
      runtimePlatform: {
        operatingSystemFamily: ecs.OperatingSystemFamily.LINUX,
        cpuArchitecture: ecs.CpuArchitecture.X86_64,
      },
    });

    taskDefinition.addContainer("CronContainer", {
      image: ecs.ContainerImage.fromEcrRepository(props.backendRepository, "latest"),
      command: ["python", "-m", "app.jobs.fetch_data"],
      environment: {
        APP_ENV: "prod",
        DATA_BUCKET: props.parquetDataBucket.bucketName,
      },
      logging: ecs.LogDrivers.awsLogs({
        streamPrefix: "data-fetch-cron",
        logGroup: this.cronLogGroup,
      }),
    });

    this.scheduledTask = new ecsPatterns.ScheduledFargateTask(this, "DataFetchCron", {
      cluster: props.cluster,
      desiredTaskCount: 1,
      schedule: events.Schedule.cron({ minute: "0/30" }),
      subnetSelection: { subnetType: ec2.SubnetType.PRIVATE_WITH_EGRESS },
      securityGroups: [securityGroup],
      scheduledFargateTaskDefinitionOptions: {
        taskDefinition,
      },
    });

    this.scheduleRuleName = this.scheduledTask.eventRule.ruleName;
  }
}
