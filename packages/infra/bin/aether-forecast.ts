#!/usr/bin/env node
import * as cdk from "aws-cdk-lib";
import { AetherForecastStack } from "../lib/aether-forecast-stack";

const app = new cdk.App();

const account = process.env.CDK_DEFAULT_ACCOUNT;
const region = process.env.CDK_DEFAULT_REGION ?? "ap-southeast-1";

new AetherForecastStack(app, "AetherForecastStack", {
  env: { account, region },
  description: "AetherForecast production infrastructure",
});
