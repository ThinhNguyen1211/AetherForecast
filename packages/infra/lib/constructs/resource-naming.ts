import { Stack } from "aws-cdk-lib";
import { Construct } from "constructs";

export const PROJECT_SLUG = "aetherforecast";

export function accountRegionSuffix(scope: Construct): string {
  const stack = Stack.of(scope);
  return `${stack.account}-${stack.region}`;
}
