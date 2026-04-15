import * as ecr from "aws-cdk-lib/aws-ecr";
import { Construct } from "constructs";

export class ContainerRegistryStack extends Construct {
  public readonly trainingRepository: ecr.Repository;

  constructor(scope: Construct, id: string) {
    super(scope, id);

    this.trainingRepository = new ecr.Repository(this, "TrainingRepository", {
      repositoryName: "aetherforecast/training",
      imageScanOnPush: true,
      imageTagMutability: ecr.TagMutability.MUTABLE,
      lifecycleRules: [
        {
          maxImageCount: 200,
          description: "Retain recent training images",
        },
      ],
    });
  }
}
