import * as ec2 from "aws-cdk-lib/aws-ec2";
import { Construct } from "constructs";

export interface NetworkStackProps {
  readonly maxAzs?: number;
}

export class NetworkStack extends Construct {
  public readonly vpc: ec2.Vpc;

  constructor(scope: Construct, id: string, props: NetworkStackProps = {}) {
    super(scope, id);

    this.vpc = new ec2.Vpc(this, "AetherVpc", {
      vpcName: "aetherforecast-vpc",
      maxAzs: props.maxAzs ?? 2,
      natGateways: 0,
      subnetConfiguration: [
        {
          cidrMask: 24,
          name: "public",
          subnetType: ec2.SubnetType.PUBLIC,
        },
        {
          cidrMask: 24,
          name: "private-data",
          subnetType: ec2.SubnetType.PRIVATE_ISOLATED,
        },
      ],
    });
  }
}
