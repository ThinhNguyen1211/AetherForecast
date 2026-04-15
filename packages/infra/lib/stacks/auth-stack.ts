import { Duration } from "aws-cdk-lib";
import * as cognito from "aws-cdk-lib/aws-cognito";
import { Construct } from "constructs";

export class AuthStack extends Construct {
  public readonly userPool: cognito.UserPool;
  public readonly userPoolClient: cognito.UserPoolClient;

  constructor(scope: Construct, id: string) {
    super(scope, id);

    this.userPool = new cognito.UserPool(this, "UserPool", {
      userPoolName: "aetherforecast-users",
      // Custom frontend auth flow requires self-service registration.
      selfSignUpEnabled: true,
      autoVerify: {
        email: true,
      },
      signInAliases: {
        email: true,
        username: false,
        phone: false,
      },
      standardAttributes: {
        email: {
          required: true,
          mutable: false,
        },
      },
      passwordPolicy: {
        minLength: 12,
        requireDigits: true,
        requireLowercase: true,
        requireUppercase: true,
        requireSymbols: true,
      },
      mfa: cognito.Mfa.OPTIONAL,
    });

    this.userPoolClient = this.userPool.addClient("WebAppClient", {
      userPoolClientName: "aetherforecast-web-client",
      authFlows: {
        userPassword: true,
        userSrp: true,
      },
      // Cognito limits access/id tokens to 24 hours. 30-day persistence is achieved via refresh token.
      accessTokenValidity: Duration.hours(24),
      idTokenValidity: Duration.hours(24),
      refreshTokenValidity: Duration.days(30),
      authSessionValidity: Duration.minutes(15),
      preventUserExistenceErrors: true,
      generateSecret: false,
    });
  }
}
