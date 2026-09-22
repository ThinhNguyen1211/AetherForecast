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
          // The ingestion job writes with mode="overwrite_partitions", which
          // deletes and re-creates every partition file it touches. With
          // versioning on, each rewrite leaves a noncurrent version plus a
          // delete marker behind. Those dead versions are the entire storage
          // bill for this bucket, so they are purged aggressively.
          id: "PurgeNoncurrentParquetVersions",
          prefix: "market/klines/",
          enabled: true,
          noncurrentVersionExpiration: Duration.days(1),
          expiredObjectDeleteMarker: true,
          abortIncompleteMultipartUploadAfter: Duration.days(7),
        },
        {
          // Watermark JSONs are rewritten in place on every flush; same problem,
          // different prefix (the old rule's "symbol=" prefix matched neither).
          id: "PurgeNoncurrentWatermarks",
          prefix: "_metadata/",
          enabled: true,
          noncurrentVersionExpiration: Duration.days(1),
          expiredObjectDeleteMarker: true,
        },
        {
          // Cold-partition tiering. Note: the bucket keeps the default
          // TransitionDefaultMinimumObjectSize=all_storage_classes_128K, and the
          // parquet parts average ~13 KB, so this is currently a no-op. It is
          // kept for when partitions are compacted into larger files.
          id: "ParquetPartitionTiering",
          prefix: "market/klines/",
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
        {
          // Superseded model/LoRA artifacts: keep a rollback window, then purge.
          // Live artifacts are ~1.5 GB; retained noncurrent versions were ~7.8 GB.
          id: "PurgeNoncurrentModelVersions",
          enabled: true,
          noncurrentVersionExpiration: Duration.days(30),
          noncurrentVersionsToRetain: 3,
          expiredObjectDeleteMarker: true,
        },
      ],
    });
  }
}
