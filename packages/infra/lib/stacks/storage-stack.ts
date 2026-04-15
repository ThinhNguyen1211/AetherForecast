import { Duration, RemovalPolicy } from "aws-cdk-lib";
import * as s3 from "aws-cdk-lib/aws-s3";
import { Construct } from "constructs";
import { PROJECT_SLUG, accountRegionSuffix } from "../constructs/resource-naming";

export class StorageStack extends Construct {
  public readonly parquetDataBucket: s3.Bucket;
  public readonly mlModelBucket: s3.Bucket;

  constructor(scope: Construct, id: string) {
    super(scope, id);

    this.parquetDataBucket = new s3.Bucket(this, "ParquetDataBucket", {
      bucketName: `${PROJECT_SLUG}-data-${accountRegionSuffix(this)}`,
      encryption: s3.BucketEncryption.S3_MANAGED,
      versioned: true,
      enforceSSL: true,
      blockPublicAccess: s3.BlockPublicAccess.BLOCK_ALL,
      removalPolicy: RemovalPolicy.RETAIN,
      lifecycleRules: [
        {
          id: "ParquetPartitionLifecycle",
          prefix: "symbol=",
          enabled: true,
          transitions: [
            {
              storageClass: s3.StorageClass.INFREQUENT_ACCESS,
              transitionAfter: Duration.days(30),
            },
            {
              storageClass: s3.StorageClass.GLACIER_INSTANT_RETRIEVAL,
              transitionAfter: Duration.days(180),
            },
          ],
          noncurrentVersionExpiration: Duration.days(365),
        },
      ],
    });

    this.mlModelBucket = new s3.Bucket(this, "MlModelBucket", {
      bucketName: `${PROJECT_SLUG}-models-${accountRegionSuffix(this)}`,
      encryption: s3.BucketEncryption.S3_MANAGED,
      versioned: true,
      enforceSSL: true,
      blockPublicAccess: s3.BlockPublicAccess.BLOCK_ALL,
      removalPolicy: RemovalPolicy.RETAIN,
      lifecycleRules: [
        {
          id: "ModelAndCheckpointLifecycle",
          enabled: true,
          transitions: [
            {
              storageClass: s3.StorageClass.INTELLIGENT_TIERING,
              transitionAfter: Duration.days(30),
            },
          ],
          abortIncompleteMultipartUploadAfter: Duration.days(7),
        },
      ],
    });
  }
}
