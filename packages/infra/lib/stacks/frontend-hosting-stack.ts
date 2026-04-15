import { Duration, RemovalPolicy } from "aws-cdk-lib";
import * as cloudfront from "aws-cdk-lib/aws-cloudfront";
import * as origins from "aws-cdk-lib/aws-cloudfront-origins";
import * as s3 from "aws-cdk-lib/aws-s3";
import { Construct } from "constructs";
import { PROJECT_SLUG, accountRegionSuffix } from "../constructs/resource-naming";

export class FrontendHostingStack extends Construct {
  public readonly frontendBucket: s3.Bucket;
  public readonly distribution: cloudfront.Distribution;

  constructor(scope: Construct, id: string) {
    super(scope, id);

    this.frontendBucket = new s3.Bucket(this, "FrontendBucket", {
      bucketName: `${PROJECT_SLUG}-frontend-${accountRegionSuffix(this)}`,
      encryption: s3.BucketEncryption.S3_MANAGED,
      enforceSSL: true,
      blockPublicAccess: s3.BlockPublicAccess.BLOCK_ALL,
      versioned: true,
      removalPolicy: RemovalPolicy.RETAIN,
    });

    const oai = new cloudfront.OriginAccessIdentity(this, "FrontendOai", {
      comment: "OAI for AetherForecast frontend bucket",
    });

    this.frontendBucket.grantRead(oai);

    this.distribution = new cloudfront.Distribution(this, "FrontendDistribution", {
      comment: "AetherForecast static frontend distribution",
      defaultBehavior: {
        origin: new origins.S3Origin(this.frontendBucket, { originAccessIdentity: oai }),
        viewerProtocolPolicy: cloudfront.ViewerProtocolPolicy.REDIRECT_TO_HTTPS,
        cachePolicy: cloudfront.CachePolicy.CACHING_OPTIMIZED,
      },
      defaultRootObject: "index.html",
      errorResponses: [
        {
          httpStatus: 403,
          responseHttpStatus: 200,
          responsePagePath: "/index.html",
          ttl: Duration.minutes(5),
        },
        {
          httpStatus: 404,
          responseHttpStatus: 200,
          responsePagePath: "/index.html",
          ttl: Duration.minutes(5),
        },
      ],
    });
  }
}
