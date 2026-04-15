interface CognitoErrorPayload {
  __type?: string;
  code?: string;
  message?: string;
}

type CognitoOperation = "InitiateAuth" | "SignUp" | "ConfirmSignUp" | "ResendConfirmationCode";

export const COGNITO_MIN_PASSWORD_LENGTH = 12;

interface CognitoAuthenticationResult {
  AccessToken?: string;
  IdToken?: string;
  RefreshToken?: string;
  ExpiresIn?: number;
}

interface InitiateAuthResponse {
  AuthenticationResult?: CognitoAuthenticationResult;
}

interface SignUpResponse {
  UserConfirmed?: boolean;
}

export interface CognitoConfig {
  region: string;
  userPoolId: string;
  clientId: string;
  isConfigured: boolean;
  missingVariables: string[];
}

export interface CognitoSignInResult {
  token: string;
  idToken: string;
  accessToken: string;
  refreshToken: string;
  expiresIn: number;
}

function toSignInResult(
  auth: CognitoAuthenticationResult | undefined,
  fallbackRefreshToken = "",
): CognitoSignInResult {
  if (!auth?.IdToken || !auth.AccessToken) {
    throw new Error("Cognito did not return authentication tokens.");
  }

  return {
    token: auth.IdToken,
    idToken: auth.IdToken,
    accessToken: auth.AccessToken,
    refreshToken: auth.RefreshToken ?? fallbackRefreshToken,
    expiresIn: auth.ExpiresIn ?? 3600,
  };
}

function getCognitoTarget(operation: string): string {
  return `AWSCognitoIdentityProviderService.${operation}`;
}

function parseCognitoError(error: unknown, operation: CognitoOperation): string {
  if (typeof error !== "object" || error === null) {
    return "Unable to complete Cognito request.";
  }

  const payload = error as CognitoErrorPayload;
  const rawType = payload.__type ?? payload.code ?? "";
  const type = rawType.split("#").pop() ?? "";
  const message = payload.message?.trim();

  switch (type) {
    case "UserNotFoundException":
      return "Incorrect email or password.";
    case "NotAuthorizedException": {
      if (operation === "SignUp") {
        return "Sign up is not enabled for this app client. Please contact support.";
      }
      if (message?.toLowerCase().includes("refresh token")) {
        return "Session has expired. Please sign in again.";
      }
      if (message?.toLowerCase().includes("signup is not permitted")) {
        return "Sign up is currently disabled. Please contact support.";
      }
      return "Incorrect email or password.";
    }
    case "UsernameExistsException":
      return "This email is already registered.";
    case "InvalidPasswordException":
      return "Password does not meet Cognito policy requirements.";
    case "InvalidParameterException":
      return message || "Invalid sign-up information. Please check your inputs.";
    case "UserNotConfirmedException":
      return "Your account is not confirmed. Enter verification code to continue.";
    case "CodeMismatchException":
      return "Verification code is incorrect.";
    case "ExpiredCodeException":
      return "Verification code has expired. Request a new one.";
    case "TooManyRequestsException":
      return "Too many requests. Please try again in a minute.";
    default:
      return message || "Cognito authentication failed.";
  }
}

export function getCognitoConfig(): CognitoConfig {
  const region = import.meta.env.VITE_COGNITO_REGION?.trim() || "ap-southeast-1";
  const userPoolId = import.meta.env.VITE_COGNITO_USER_POOL_ID?.trim() || "";
  const clientId = import.meta.env.VITE_COGNITO_CLIENT_ID?.trim() || "";

  const missingVariables: string[] = [];
  if (!userPoolId) {
    missingVariables.push("VITE_COGNITO_USER_POOL_ID");
  }
  if (!clientId) {
    missingVariables.push("VITE_COGNITO_CLIENT_ID");
  }

  return {
    region,
    userPoolId,
    clientId,
    isConfigured: missingVariables.length === 0,
    missingVariables,
  };
}

async function callCognito<T>(
  operation: CognitoOperation,
  payload: Record<string, unknown>,
  config: CognitoConfig,
): Promise<T> {
  if (!config.isConfigured) {
    throw new Error(`Missing Cognito config: ${config.missingVariables.join(", ")}`);
  }

  const endpoint = `https://cognito-idp.${config.region}.amazonaws.com/`;

  const response = await fetch(endpoint, {
    method: "POST",
    headers: {
      "Content-Type": "application/x-amz-json-1.1",
      "X-Amz-Target": getCognitoTarget(operation),
    },
    body: JSON.stringify(payload),
  });

  const body = (await response.json().catch(() => ({}))) as unknown;

  if (!response.ok) {
    throw new Error(parseCognitoError(body, operation));
  }

  const errorPayload = body as CognitoErrorPayload;
  if (errorPayload.__type || errorPayload.code) {
    throw new Error(parseCognitoError(errorPayload, operation));
  }

  return body as T;
}

export async function signInWithPassword(email: string, password: string): Promise<CognitoSignInResult> {
  const config = getCognitoConfig();
  const normalizedEmail = email.trim().toLowerCase();

  const result = await callCognito<InitiateAuthResponse>(
    "InitiateAuth",
    {
      AuthFlow: "USER_PASSWORD_AUTH",
      ClientId: config.clientId,
      AuthParameters: {
        USERNAME: normalizedEmail,
        PASSWORD: password,
      },
    },
    config,
  );

  return toSignInResult(result.AuthenticationResult);
}

export async function refreshCognitoSession(refreshToken: string): Promise<CognitoSignInResult> {
  const normalizedRefreshToken = refreshToken.trim();
  if (!normalizedRefreshToken) {
    throw new Error("Refresh token is missing.");
  }

  const config = getCognitoConfig();
  const result = await callCognito<InitiateAuthResponse>(
    "InitiateAuth",
    {
      AuthFlow: "REFRESH_TOKEN_AUTH",
      ClientId: config.clientId,
      AuthParameters: {
        REFRESH_TOKEN: normalizedRefreshToken,
      },
    },
    config,
  );

  return toSignInResult(result.AuthenticationResult, normalizedRefreshToken);
}

export async function signUpWithPassword(email: string, password: string): Promise<{ userConfirmed: boolean }> {
  const config = getCognitoConfig();
  const normalizedEmail = email.trim().toLowerCase();

  const result = await callCognito<SignUpResponse>(
    "SignUp",
    {
      ClientId: config.clientId,
      Username: normalizedEmail,
      Password: password,
      UserAttributes: [{ Name: "email", Value: normalizedEmail }],
    },
    config,
  );

  return { userConfirmed: Boolean(result.UserConfirmed) };
}

export async function confirmSignUpCode(email: string, code: string): Promise<void> {
  const config = getCognitoConfig();
  await callCognito(
    "ConfirmSignUp",
    {
      ClientId: config.clientId,
      Username: email.trim().toLowerCase(),
      ConfirmationCode: code.trim(),
    },
    config,
  );
}

export async function resendSignUpCode(email: string): Promise<void> {
  const config = getCognitoConfig();
  await callCognito(
    "ResendConfirmationCode",
    {
      ClientId: config.clientId,
      Username: email.trim().toLowerCase(),
    },
    config,
  );
}