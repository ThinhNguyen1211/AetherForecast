import { Duration, Stack } from "aws-cdk-lib";
import * as cloudwatch from "aws-cdk-lib/aws-cloudwatch";
import * as cloudwatchActions from "aws-cdk-lib/aws-cloudwatch-actions";
import * as ec2 from "aws-cdk-lib/aws-ec2";
import * as s3 from "aws-cdk-lib/aws-s3";
import * as sns from "aws-cdk-lib/aws-sns";
import { Construct } from "constructs";
import * as fs from "fs";
import * as path from "path";

export interface MonitoringStackProps {
  readonly backendInstance: ec2.Instance;
  readonly batchJobQueueName: string;
  readonly parquetDataBucket: s3.IBucket;
  readonly mlModelBucket: s3.IBucket;
}

export class MonitoringStack extends Construct {
  public readonly alarmTopic: sns.Topic;
  public readonly dashboardName: string;

  constructor(scope: Construct, id: string, props: MonitoringStackProps) {
    super(scope, id);

    this.dashboardName = "AetherForecast-Operations";

    this.alarmTopic = new sns.Topic(this, "OperationsAlarmTopic", {
      topicName: "aetherforecast-ops-alarms",
      displayName: "AetherForecast Production Alarm Notifications",
    });

    const ec2CpuMetric = new cloudwatch.Metric({
      namespace: "AWS/EC2",
      metricName: "CPUUtilization",
      dimensionsMap: {
        InstanceId: props.backendInstance.instanceId,
      },
      statistic: "Average",
      period: Duration.minutes(5),
    });

    const api5xxMetric = new cloudwatch.Metric({
      namespace: "AetherForecast/API",
      metricName: "Api5xx",
      dimensionsMap: {
        Service: "backend",
      },
      statistic: "Sum",
      period: Duration.minutes(5),
    });

    const apiLatencyMetric = new cloudwatch.Metric({
      namespace: "AetherForecast/API",
      metricName: "ApiLatencyMs",
      dimensionsMap: {
        Service: "backend",
      },
      statistic: "Average",
      period: Duration.minutes(5),
    });

    const fetchErrorsMetric = new cloudwatch.Metric({
      namespace: "AetherForecast/Pipeline",
      metricName: "FetchErrors",
      dimensionsMap: {
        Pipeline: "data-fetch-cron",
      },
      statistic: "Sum",
      period: Duration.minutes(5),
    });

    const batchFailedMetric = new cloudwatch.Metric({
      namespace: "AWS/Batch",
      metricName: "FailedJobs",
      dimensionsMap: {
        JobQueue: props.batchJobQueueName,
      },
      statistic: "Sum",
      period: Duration.minutes(5),
    });

    const cronRunsMetric = new cloudwatch.Metric({
      namespace: "AetherForecast/Pipeline",
      metricName: "FetchRuns",
      dimensionsMap: {
        Pipeline: "data-fetch-cron",
      },
      statistic: "Sum",
      period: Duration.minutes(5),
    });

    const containerHealthyMetric = new cloudwatch.Metric({
      namespace: "AetherForecast/Host",
      metricName: "ContainerHealthy",
      dimensionsMap: {
        InstanceId: props.backendInstance.instanceId,
      },
      statistic: "Average",
      period: Duration.minutes(5),
    });

    const cronHealthyMetric = new cloudwatch.Metric({
      namespace: "AetherForecast/Host",
      metricName: "CronHealthy",
      dimensionsMap: {
        InstanceId: props.backendInstance.instanceId,
      },
      statistic: "Average",
      period: Duration.minutes(5),
    });

    const cronLastSuccessAgeMetric = new cloudwatch.Metric({
      namespace: "AetherForecast/Host",
      metricName: "CronLastSuccessAgeSeconds",
      dimensionsMap: {
        InstanceId: props.backendInstance.instanceId,
      },
      statistic: "Maximum",
      period: Duration.minutes(5),
    });

    const highCpuAlarm = new cloudwatch.Alarm(this, "BackendHighCpuAlarm", {
      alarmName: "aetherforecast-backend-high-cpu",
      alarmDescription: "Backend EC2 CPU utilization is above 80% for 10 minutes",
      metric: ec2CpuMetric,
      threshold: 80,
      evaluationPeriods: 2,
      datapointsToAlarm: 2,
      comparisonOperator: cloudwatch.ComparisonOperator.GREATER_THAN_OR_EQUAL_TO_THRESHOLD,
      treatMissingData: cloudwatch.TreatMissingData.NOT_BREACHING,
    });

    const highErrorRateAlarm = new cloudwatch.Alarm(this, "BackendHighErrorAlarm", {
      alarmName: "aetherforecast-backend-high-5xx",
      alarmDescription: "Backend FastAPI 5xx custom metric exceeded threshold",
      metric: api5xxMetric,
      threshold: 5,
      evaluationPeriods: 1,
      datapointsToAlarm: 1,
      comparisonOperator: cloudwatch.ComparisonOperator.GREATER_THAN_OR_EQUAL_TO_THRESHOLD,
      treatMissingData: cloudwatch.TreatMissingData.NOT_BREACHING,
    });

    const highLatencyAlarm = new cloudwatch.Alarm(this, "BackendHighLatencyAlarm", {
      alarmName: "aetherforecast-backend-high-latency",
      alarmDescription: "Backend API average latency is above 2000 ms",
      metric: apiLatencyMetric,
      threshold: 2000,
      evaluationPeriods: 2,
      datapointsToAlarm: 2,
      comparisonOperator: cloudwatch.ComparisonOperator.GREATER_THAN_OR_EQUAL_TO_THRESHOLD,
      treatMissingData: cloudwatch.TreatMissingData.NOT_BREACHING,
    });

    const batchFailureAlarm = new cloudwatch.Alarm(this, "BatchFailureAlarm", {
      alarmName: "aetherforecast-batch-job-failure",
      alarmDescription: "Batch training job failures detected",
      metric: batchFailedMetric,
      threshold: 1,
      evaluationPeriods: 1,
      datapointsToAlarm: 1,
      comparisonOperator: cloudwatch.ComparisonOperator.GREATER_THAN_OR_EQUAL_TO_THRESHOLD,
      treatMissingData: cloudwatch.TreatMissingData.NOT_BREACHING,
    });

    const cronFailureAlarm = new cloudwatch.Alarm(this, "CronFailureAlarm", {
      alarmName: "aetherforecast-cron-failure",
      alarmDescription: "Cron data fetch error events detected",
      metric: fetchErrorsMetric,
      threshold: 1,
      evaluationPeriods: 1,
      datapointsToAlarm: 1,
      comparisonOperator: cloudwatch.ComparisonOperator.GREATER_THAN_OR_EQUAL_TO_THRESHOLD,
      treatMissingData: cloudwatch.TreatMissingData.NOT_BREACHING,
    });

    const cronMissingRunAlarm = new cloudwatch.Alarm(this, "CronMissingRunAlarm", {
      alarmName: "aetherforecast-cron-missing-runs",
      alarmDescription: "No cron fetch runs detected in the last 30 minutes",
      metric: cronRunsMetric,
      threshold: 1,
      evaluationPeriods: 2,
      datapointsToAlarm: 2,
      comparisonOperator: cloudwatch.ComparisonOperator.LESS_THAN_THRESHOLD,
      treatMissingData: cloudwatch.TreatMissingData.BREACHING,
    });

    const containerUnhealthyAlarm = new cloudwatch.Alarm(this, "ContainerUnhealthyAlarm", {
      alarmName: "aetherforecast-backend-container-unhealthy",
      alarmDescription: "Backend container health metric indicates unhealthy runtime",
      metric: containerHealthyMetric,
      threshold: 1,
      evaluationPeriods: 2,
      datapointsToAlarm: 2,
      comparisonOperator: cloudwatch.ComparisonOperator.LESS_THAN_THRESHOLD,
      treatMissingData: cloudwatch.TreatMissingData.BREACHING,
    });

    const cronUnhealthyAlarm = new cloudwatch.Alarm(this, "CronUnhealthyAlarm", {
      alarmName: "aetherforecast-cron-unhealthy",
      alarmDescription: "Host cron health metric is unhealthy",
      metric: cronHealthyMetric,
      threshold: 1,
      evaluationPeriods: 2,
      datapointsToAlarm: 2,
      comparisonOperator: cloudwatch.ComparisonOperator.LESS_THAN_THRESHOLD,
      treatMissingData: cloudwatch.TreatMissingData.BREACHING,
    });

    const cronStaleAlarm = new cloudwatch.Alarm(this, "CronStaleAlarm", {
      alarmName: "aetherforecast-cron-last-success-stale",
      alarmDescription: "Last successful cron run is stale",
      metric: cronLastSuccessAgeMetric,
      threshold: 1800,
      evaluationPeriods: 2,
      datapointsToAlarm: 2,
      comparisonOperator: cloudwatch.ComparisonOperator.GREATER_THAN_THRESHOLD,
      treatMissingData: cloudwatch.TreatMissingData.BREACHING,
    });

    const alarmAction = new cloudwatchActions.SnsAction(this.alarmTopic);
    highCpuAlarm.addAlarmAction(alarmAction);
    highErrorRateAlarm.addAlarmAction(alarmAction);
    highLatencyAlarm.addAlarmAction(alarmAction);
    batchFailureAlarm.addAlarmAction(alarmAction);
    cronFailureAlarm.addAlarmAction(alarmAction);
    cronMissingRunAlarm.addAlarmAction(alarmAction);
    containerUnhealthyAlarm.addAlarmAction(alarmAction);
    cronUnhealthyAlarm.addAlarmAction(alarmAction);
    cronStaleAlarm.addAlarmAction(alarmAction);

    const dashboardTemplatePath = path.resolve(__dirname, "../monitoring/dashboard.json");
    const dashboardTemplate = fs.readFileSync(dashboardTemplatePath, "utf8");

    const dashboardBody = this.renderDashboardBody(dashboardTemplate, {
      "__REGION__": Stack.of(this).region,
      "__INSTANCE_ID__": props.backendInstance.instanceId,
      "__BATCH_JOB_QUEUE_NAME__": props.batchJobQueueName,
      "__PARQUET_BUCKET_NAME__": props.parquetDataBucket.bucketName,
      "__MODEL_BUCKET_NAME__": props.mlModelBucket.bucketName,
    });

    new cloudwatch.CfnDashboard(this, "OperationsDashboard", {
      dashboardName: this.dashboardName,
      dashboardBody: dashboardBody,
    });
  }

  private renderDashboardBody(template: string, replacements: Record<string, string>): string {
    let output = template;
    for (const [key, value] of Object.entries(replacements)) {
      output = output.split(key).join(value);
    }
    return output;
  }
}
