import { CfnOutput, CfnParameter, Stack, StackProps, Tags } from "aws-cdk-lib";
import { Construct } from "constructs";
import { AuthStack } from "./stacks/auth-stack";
import { ContainerRegistryStack } from "./stacks/container-registry-stack";
import { Ec2Stack } from "./stacks/ec2-stack";
import { FrontendHostingStack } from "./stacks/frontend-hosting-stack";
import { MlBatchStack } from "./stacks/ml-batch-stack";
import { MonitoringStack } from "./stacks/monitoring-stack";
import { NetworkStack } from "./stacks/network-stack";
import { StorageStack } from "./stacks/storage-stack";
import { TrainingEc2Stack } from "./stacks/training-ec2-stack";

export class AetherForecastStack extends Stack {
  constructor(scope: Construct, id: string, props?: StackProps) {
    super(scope, id, props);

    Tags.of(this).add("Project", "AetherForecast");
    Tags.of(this).add("Environment", "prod");

    const network = new NetworkStack(this, "NetworkStack", {
      maxAzs: 2,
    });

    const storage = new StorageStack(this, "StorageStack");

    const auth = new AuthStack(this, "AuthStack");

    const registries = new ContainerRegistryStack(this, "ContainerRegistryStack");

    const trainingImageTag = new CfnParameter(this, "TrainingImageTag", {
      type: "String",
      default: "latest",
      description: "Training container image tag to use in AWS Batch job definition",
    });

    const trainingImageUri = `${registries.trainingRepository.repositoryUri}:${trainingImageTag.valueAsString}`;

    const ec2Backend = new Ec2Stack(this, "Ec2Stack", {
      vpc: network.vpc,
      parquetDataBucket: storage.parquetDataBucket,
      mlModelBucket: storage.mlModelBucket,
      cognitoUserPoolId: auth.userPool.userPoolId,
      cognitoClientId: auth.userPoolClient.userPoolClientId,
    });

    const trainingEc2 = new TrainingEc2Stack(this, "TrainingEc2Stack", {
      vpc: network.vpc,
      trainingRepository: registries.trainingRepository,
      parquetDataBucket: storage.parquetDataBucket,
      mlModelBucket: storage.mlModelBucket,
    });

    const mlBatch = new MlBatchStack(this, "MlBatchStack", {
      vpc: network.vpc,
      trainingRepository: registries.trainingRepository,
      trainingImageUri,
      parquetDataBucket: storage.parquetDataBucket,
      mlModelBucket: storage.mlModelBucket,
    });

    const monitoring = new MonitoringStack(this, "MonitoringStack", {
      backendInstance: ec2Backend.instance,
      batchJobQueueName: mlBatch.jobQueueName,
      parquetDataBucket: storage.parquetDataBucket,
      mlModelBucket: storage.mlModelBucket,
    });

    const frontend = new FrontendHostingStack(this, "FrontendHostingStack");

    new CfnOutput(this, "ParquetDataBucketName", {
      value: storage.parquetDataBucket.bucketName,
    });

    new CfnOutput(this, "MlModelBucketName", {
      value: storage.mlModelBucket.bucketName,
    });

    new CfnOutput(this, "TrainingEcrRepositoryUri", {
      value: registries.trainingRepository.repositoryUri,
    });

    new CfnOutput(this, "TrainingImageUri", {
      value: trainingImageUri,
    });

    new CfnOutput(this, "BackendEc2InstanceId", {
      value: ec2Backend.instance.instanceId,
    });

    new CfnOutput(this, "BackendElasticIp", {
      value: ec2Backend.elasticIp.ref,
    });

    new CfnOutput(this, "BackendEc2LogGroupName", {
      value: ec2Backend.backendLogGroup.logGroupName,
    });

    new CfnOutput(this, "TrainingEc2InstanceId", {
      value: trainingEc2.instance.instanceId,
    });

    new CfnOutput(this, "TrainingEc2ElasticIp", {
      value: trainingEc2.elasticIp.ref,
    });

    new CfnOutput(this, "TrainingEc2SecurityGroupId", {
      value: trainingEc2.securityGroup.securityGroupId,
    });

    new CfnOutput(this, "TrainingEc2LogGroupName", {
      value: trainingEc2.trainingLogGroup.logGroupName,
    });

    new CfnOutput(this, "BatchJobQueueArn", {
      value: mlBatch.jobQueue.attrJobQueueArn,
    });

    new CfnOutput(this, "BatchJobDefinitionArn", {
      value: mlBatch.jobDefinition.attrJobDefinitionArn,
    });

    new CfnOutput(this, "CognitoUserPoolId", {
      value: auth.userPool.userPoolId,
    });

    new CfnOutput(this, "CognitoAppClientId", {
      value: auth.userPoolClient.userPoolClientId,
    });

    new CfnOutput(this, "FrontendBucketName", {
      value: frontend.frontendBucket.bucketName,
    });

    new CfnOutput(this, "CloudFrontDistributionDomain", {
      value: frontend.distribution.distributionDomainName,
    });

    new CfnOutput(this, "OperationsDashboardName", {
      value: monitoring.dashboardName,
    });

    new CfnOutput(this, "OperationsAlarmTopicArn", {
      value: monitoring.alarmTopic.topicArn,
    });

    new CfnOutput(this, "OperationsDashboardUrl", {
      value: `https://${this.region}.console.aws.amazon.com/cloudwatch/home?region=${this.region}#dashboards:name=${monitoring.dashboardName}`,
    });
  }
}
