import { Duration } from "aws-cdk-lib";
import * as ec2 from "aws-cdk-lib/aws-ec2";
import * as ecr from "aws-cdk-lib/aws-ecr";
import * as ecs from "aws-cdk-lib/aws-ecs";
import * as ecsPatterns from "aws-cdk-lib/aws-ecs-patterns";
import * as elbv2 from "aws-cdk-lib/aws-elasticloadbalancingv2";
import * as iam from "aws-cdk-lib/aws-iam";
import * as logs from "aws-cdk-lib/aws-logs";
import * as s3 from "aws-cdk-lib/aws-s3";
import { Construct } from "constructs";

export interface ComputeStackProps {
  readonly vpc: ec2.IVpc;
  readonly backendRepository: ecr.IRepository;
  readonly parquetDataBucket: s3.IBucket;
  readonly mlModelBucket: s3.IBucket;
}

export class ComputeStack extends Construct {
  public readonly cluster: ecs.Cluster;
  public readonly backendService: ecs.FargateService;
  public readonly backendTaskDefinition: ecs.FargateTaskDefinition;
  public readonly backendTaskRole: iam.Role;
  public readonly backendApiLogGroup: logs.LogGroup;
  public readonly inferenceLogGroup: logs.LogGroup;
  public readonly backendLoadBalancer: elbv2.ApplicationLoadBalancer;
  public readonly backendTargetGroup: elbv2.ApplicationTargetGroup;
  public readonly backendLoadBalancerDnsName: string;

  constructor(scope: Construct, id: string, props: ComputeStackProps) {
    super(scope, id);

    this.cluster = new ecs.Cluster(this, "Cluster", {
      vpc: props.vpc,
      containerInsights: true,
      clusterName: "aetherforecast-cluster",
    });

    const serviceSecurityGroup = new ec2.SecurityGroup(this, "BackendServiceSecurityGroup", {
      vpc: props.vpc,
      allowAllOutbound: true,
      description: "Security group for backend + inference Fargate tasks",
    });

    this.backendTaskRole = new iam.Role(this, "BackendTaskRole", {
      assumedBy: new iam.ServicePrincipal("ecs-tasks.amazonaws.com"),
      description: "Task role for backend and inference containers",
    });

    this.backendTaskRole.addToPolicy(
      new iam.PolicyStatement({
        actions: ["s3:ListBucket"],
        resources: [props.parquetDataBucket.bucketArn, props.mlModelBucket.bucketArn],
      }),
    );

    this.backendTaskRole.addToPolicy(
      new iam.PolicyStatement({
        actions: ["s3:GetObject"],
        resources: [
          `${props.parquetDataBucket.bucketArn}/*`,
          `${props.mlModelBucket.bucketArn}/*`,
        ],
      }),
    );

    const executionRole = new iam.Role(this, "BackendExecutionRole", {
      assumedBy: new iam.ServicePrincipal("ecs-tasks.amazonaws.com"),
      managedPolicies: [
        iam.ManagedPolicy.fromAwsManagedPolicyName(
          "service-role/AmazonECSTaskExecutionRolePolicy",
        ),
      ],
      description: "Execution role for pulling images and writing logs",
    });

    props.backendRepository.grantPull(executionRole);

    this.backendTaskDefinition = new ecs.FargateTaskDefinition(this, "BackendTaskDefinition", {
      cpu: 1024,
      memoryLimitMiB: 2048,
      taskRole: this.backendTaskRole,
      executionRole,
      runtimePlatform: {
        operatingSystemFamily: ecs.OperatingSystemFamily.LINUX,
        cpuArchitecture: ecs.CpuArchitecture.X86_64,
      },
    });

    this.backendApiLogGroup = new logs.LogGroup(this, "BackendApiLogGroup", {
      logGroupName: "/aws/ecs/aetherforecast/backend-api",
      retention: logs.RetentionDays.ONE_MONTH,
    });

    this.inferenceLogGroup = new logs.LogGroup(this, "InferenceLogGroup", {
      logGroupName: "/aws/ecs/aetherforecast/inference",
      retention: logs.RetentionDays.ONE_MONTH,
    });

    const backendContainer = this.backendTaskDefinition.addContainer("BackendApiContainer", {
      image: ecs.ContainerImage.fromEcrRepository(props.backendRepository, "latest"),
      logging: ecs.LogDrivers.awsLogs({
        streamPrefix: "backend-api",
        logGroup: this.backendApiLogGroup,
      }),
      environment: {
        APP_ENV: "prod",
        MODEL_BUCKET: props.mlModelBucket.bucketName,
        PARQUET_BUCKET: props.parquetDataBucket.bucketName,
      },
      healthCheck: {
        command: ["CMD-SHELL", "wget -q -O - http://localhost:8000/health || exit 1"],
        interval: Duration.seconds(30),
        timeout: Duration.seconds(5),
        retries: 3,
        startPeriod: Duration.seconds(30),
      },
    });

    backendContainer.addPortMappings({
      containerPort: 8000,
      protocol: ecs.Protocol.TCP,
    });

    const inferenceContainer = this.backendTaskDefinition.addContainer("InferenceContainer", {
      image: ecs.ContainerImage.fromEcrRepository(props.backendRepository, "latest"),
      command: ["python", "-m", "app.inference.server"],
      logging: ecs.LogDrivers.awsLogs({
        streamPrefix: "inference",
        logGroup: this.inferenceLogGroup,
      }),
      environment: {
        APP_ENV: "prod",
        MODEL_BUCKET: props.mlModelBucket.bucketName,
      },
      essential: false,
      healthCheck: {
        command: ["CMD-SHELL", "wget -q -O - http://localhost:8080/health || exit 1"],
        interval: Duration.seconds(30),
        timeout: Duration.seconds(5),
        retries: 3,
        startPeriod: Duration.seconds(30),
      },
    });

    inferenceContainer.addPortMappings({
      containerPort: 8080,
      protocol: ecs.Protocol.TCP,
    });

    backendContainer.addContainerDependencies({
      container: inferenceContainer,
      condition: ecs.ContainerDependencyCondition.START,
    });

    const service = new ecsPatterns.ApplicationLoadBalancedFargateService(this, "BackendService", {
      cluster: this.cluster,
      taskDefinition: this.backendTaskDefinition,
      desiredCount: 2,
      publicLoadBalancer: true,
      assignPublicIp: false,
      securityGroups: [serviceSecurityGroup],
      taskSubnets: { subnetType: ec2.SubnetType.PRIVATE_WITH_EGRESS },
      healthCheckGracePeriod: Duration.seconds(90),
      circuitBreaker: { rollback: true },
    });

    service.targetGroup.configureHealthCheck({
      path: "/health",
      healthyHttpCodes: "200",
      interval: Duration.seconds(30),
    });

    const scaling = service.service.autoScaleTaskCount({
      minCapacity: 2,
      maxCapacity: 10,
    });

    scaling.scaleOnCpuUtilization("CpuScaling", {
      targetUtilizationPercent: 60,
      scaleInCooldown: Duration.seconds(60),
      scaleOutCooldown: Duration.seconds(60),
    });

    this.backendService = service.service;
    this.backendLoadBalancer = service.loadBalancer;
    this.backendTargetGroup = service.targetGroup;
    this.backendLoadBalancerDnsName = service.loadBalancer.loadBalancerDnsName;
  }
}
