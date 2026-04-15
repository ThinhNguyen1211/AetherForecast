import { FormEvent, useEffect, useMemo, useState } from "react";

import {
  COGNITO_MIN_PASSWORD_LENGTH,
  confirmSignUpCode,
  getCognitoConfig,
  resendSignUpCode,
  signInWithPassword,
  signUpWithPassword,
} from "@/services/cognito-auth";
import { saveAuthSession } from "@/services/api";

interface AuthTokenModalProps {
  open: boolean;
  defaultMode?: "signin" | "signup";
  hasToken: boolean;
  onAuthenticate: (token: string) => void;
  onSignOut: () => void;
  onClose: () => void;
}

type AuthMode = "signin" | "signup" | "confirm";

function isEmailLike(value: string): boolean {
  return /.+@.+\..+/.test(value.trim());
}

function EyeIcon({ visible }: { visible: boolean }) {
  return (
    <svg
      viewBox="0 0 24 24"
      className="h-4 w-4"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      {visible ? (
        <>
          <path d="M2 12s3.5-7 10-7 10 7 10 7-3.5 7-10 7-10-7-10-7z" />
          <circle cx="12" cy="12" r="3" />
        </>
      ) : (
        <>
          <path d="M3 3l18 18" />
          <path d="M10.6 10.6A3 3 0 0013.4 13.4" />
          <path d="M9.4 5.4A10.9 10.9 0 0112 5c6.5 0 10 7 10 7a17.6 17.6 0 01-3.2 4.6" />
          <path d="M6.7 6.7C4.1 8.2 2.5 12 2.5 12s3.5 7 9.5 7a9.8 9.8 0 004.2-.9" />
        </>
      )}
    </svg>
  );
}

export default function AuthTokenModal({
  open,
  defaultMode = "signin",
  hasToken,
  onAuthenticate,
  onSignOut,
  onClose,
}: AuthTokenModalProps) {
  const [mode, setMode] = useState<AuthMode>("signin");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [verificationCode, setVerificationCode] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [errorMessage, setErrorMessage] = useState("");
  const [successMessage, setSuccessMessage] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [showConfirmPassword, setShowConfirmPassword] = useState(false);

  const cognitoConfig = useMemo(() => getCognitoConfig(), []);

  useEffect(() => {
    if (!open) {
      setPassword("");
      setConfirmPassword("");
      setVerificationCode("");
      setErrorMessage("");
      setSuccessMessage("");
      setShowPassword(false);
      setShowConfirmPassword(false);
      if (hasToken) {
        setMode("signin");
      }
      return;
    }

    if (!hasToken) {
      setMode(defaultMode);
    }
  }, [open, hasToken, defaultMode]);

  if (!open) {
    return null;
  }

  const modeTitle =
    mode === "signin"
      ? "Sign In"
      : mode === "signup"
        ? "Create Account"
        : "Confirm Account";

  const modeDescription =
    mode === "signin"
      ? "Sign in with Cognito email/password to unlock protected routes."
      : mode === "signup"
        ? "Create a Cognito account with your email and a strong password."
        : "Enter the verification code sent to your email.";

  const resetMessages = () => {
    setErrorMessage("");
    setSuccessMessage("");
  };

  const switchMode = (nextMode: AuthMode) => {
    setMode(nextMode);
    resetMessages();
    setPassword("");
    setConfirmPassword("");
    setShowPassword(false);
    setShowConfirmPassword(false);
    if (nextMode !== "confirm") {
      setVerificationCode("");
    }
  };

  const validateShared = (): boolean => {
    if (!isEmailLike(email)) {
      setErrorMessage("Please enter a valid email address.");
      return false;
    }

    if (mode !== "confirm" && password.length < COGNITO_MIN_PASSWORD_LENGTH) {
      setErrorMessage(`Password must have at least ${COGNITO_MIN_PASSWORD_LENGTH} characters.`);
      return false;
    }

    return true;
  };

  const handleSignIn = async () => {
    if (!validateShared()) {
      return;
    }

    const authResult = await signInWithPassword(email, password);
    saveAuthSession(authResult);
    onAuthenticate(authResult.token);
    setSuccessMessage("Signed in successfully.");
    onClose();
  };

  const handleSignUp = async () => {
    if (!validateShared()) {
      return;
    }

    if (password !== confirmPassword) {
      setErrorMessage("Password confirmation does not match.");
      return;
    }

    const signUpResult = await signUpWithPassword(email, password);
    if (signUpResult.userConfirmed) {
      const authResult = await signInWithPassword(email, password);
      saveAuthSession(authResult);
      onAuthenticate(authResult.token);
      setSuccessMessage("Account created and signed in.");
      onClose();
      return;
    }

    setSuccessMessage("Verification code sent. Confirm account to continue.");
    setMode("confirm");
  };

  const handleConfirm = async () => {
    if (!isEmailLike(email)) {
      setErrorMessage("Please enter a valid email address.");
      return;
    }

    if (!verificationCode.trim()) {
      setErrorMessage("Verification code is required.");
      return;
    }

    await confirmSignUpCode(email, verificationCode);
    const authResult = await signInWithPassword(email, password);
    saveAuthSession(authResult);
    onAuthenticate(authResult.token);
    setSuccessMessage("Account confirmed and signed in.");
    onClose();
  };

  const handleSubmit = async () => {
    resetMessages();
    setIsSubmitting(true);

    try {
      if (mode === "signin") {
        await handleSignIn();
      } else if (mode === "signup") {
        await handleSignUp();
      } else {
        await handleConfirm();
      }
    } catch (error) {
      const nextMessage = error instanceof Error ? error.message : "Authentication failed.";
      setErrorMessage(nextMessage);
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleResendCode = async () => {
    resetMessages();
    setIsSubmitting(true);

    try {
      await resendSignUpCode(email);
      setSuccessMessage("A new verification code has been sent.");
    } catch (error) {
      const nextMessage = error instanceof Error ? error.message : "Unable to resend code.";
      setErrorMessage(nextMessage);
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleFormSubmit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    void handleSubmit();
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 p-4">
      <div className="glass-panel w-full max-w-xl rounded-2xl p-6">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <h2 className="text-xl font-semibold text-neon-cyan">{modeTitle}</h2>
            <p className="mt-1 text-sm text-violet-200/80">{modeDescription}</p>
          </div>
          {hasToken && (
            <button
              type="button"
              className="rounded-lg border border-violet-400/35 px-3 py-2 text-xs text-violet-200 transition hover:border-violet-300/60"
              onClick={onSignOut}
            >
              Sign out
            </button>
          )}
        </div>

        {!cognitoConfig.isConfigured && (
          <div className="mt-4 rounded-lg border border-rose-400/40 bg-rose-500/10 px-3 py-2 text-xs text-rose-100">
            Missing Cognito env: {cognitoConfig.missingVariables.join(", ")}
          </div>
        )}

        <div className="mt-4 flex gap-2">
          <button
            type="button"
            className={`rounded-lg border px-3 py-2 text-xs transition ${
              mode === "signin"
                ? "border-cyan-300/80 bg-cyan-400/15 text-cyan-100"
                : "border-violet-400/35 text-violet-200 hover:border-violet-300/60"
            }`}
            onClick={() => switchMode("signin")}
          >
            Sign in
          </button>
          <button
            type="button"
            className={`rounded-lg border px-3 py-2 text-xs transition ${
              mode === "signup"
                ? "border-cyan-300/80 bg-cyan-400/15 text-cyan-100"
                : "border-violet-400/35 text-violet-200 hover:border-violet-300/60"
            }`}
            onClick={() => switchMode("signup")}
          >
            Register
          </button>
        </div>

        <form onSubmit={handleFormSubmit}>
          <div className="mt-4 space-y-3">
            <input
              className="w-full rounded-xl border border-violet-400/40 bg-cosmic-900/80 p-3 text-sm text-violet-50 outline-none ring-neon-cyan/50 transition focus:ring"
              placeholder="Email"
              value={email}
              onChange={(event) => setEmail(event.target.value)}
              autoComplete="email"
            />

            {mode !== "confirm" && (
              <div className="relative">
                <input
                  type={showPassword ? "text" : "password"}
                  className="w-full rounded-xl border border-violet-400/40 bg-cosmic-900/80 p-3 pr-12 text-sm text-violet-50 outline-none ring-neon-cyan/50 transition focus:ring"
                  placeholder="Password"
                  value={password}
                  onChange={(event) => setPassword(event.target.value)}
                  autoComplete={mode === "signin" ? "current-password" : "new-password"}
                />
                <button
                  type="button"
                  onClick={() => setShowPassword((value) => !value)}
                  className="absolute right-3 top-1/2 -translate-y-1/2 text-violet-200/80 transition hover:text-cyan-200"
                  aria-label={showPassword ? "Hide password" : "Show password"}
                >
                  <EyeIcon visible={showPassword} />
                </button>
              </div>
            )}

            {mode === "signup" && (
              <div className="relative">
                <input
                  type={showConfirmPassword ? "text" : "password"}
                  className="w-full rounded-xl border border-violet-400/40 bg-cosmic-900/80 p-3 pr-12 text-sm text-violet-50 outline-none ring-neon-cyan/50 transition focus:ring"
                  placeholder="Confirm password"
                  value={confirmPassword}
                  onChange={(event) => setConfirmPassword(event.target.value)}
                  autoComplete="new-password"
                />
                <button
                  type="button"
                  onClick={() => setShowConfirmPassword((value) => !value)}
                  className="absolute right-3 top-1/2 -translate-y-1/2 text-violet-200/80 transition hover:text-cyan-200"
                  aria-label={showConfirmPassword ? "Hide confirm password" : "Show confirm password"}
                >
                  <EyeIcon visible={showConfirmPassword} />
                </button>
              </div>
            )}

            {mode === "confirm" && (
              <input
                className="w-full rounded-xl border border-violet-400/40 bg-cosmic-900/80 p-3 text-sm text-violet-50 outline-none ring-neon-cyan/50 transition focus:ring"
                placeholder="Verification code"
                value={verificationCode}
                onChange={(event) => setVerificationCode(event.target.value)}
                autoComplete="one-time-code"
              />
            )}
          </div>

          {errorMessage && (
            <div className="mt-4 rounded-lg border border-rose-400/40 bg-rose-500/10 px-3 py-2 text-sm text-rose-100">
              {errorMessage}
            </div>
          )}

          {successMessage && (
            <div className="mt-4 rounded-lg border border-cyan-400/35 bg-cyan-500/10 px-3 py-2 text-sm text-cyan-100">
              {successMessage}
            </div>
          )}

          <div className="mt-5 flex justify-end gap-2">
            {mode === "confirm" && (
              <button
                type="button"
                className="rounded-lg border border-violet-400/35 px-4 py-2 text-sm text-violet-100 transition hover:border-violet-300/60 disabled:cursor-not-allowed disabled:opacity-50"
                onClick={handleResendCode}
                disabled={isSubmitting || !cognitoConfig.isConfigured}
              >
                Resend code
              </button>
            )}
            <button
              type="button"
              className="rounded-lg border border-violet-400/35 px-4 py-2 text-sm text-violet-100 transition hover:border-violet-300/60"
              onClick={onClose}
            >
              Close
            </button>
            <button
              type="submit"
              className="rounded-lg border border-cyan-400/70 bg-cyan-500/10 px-4 py-2 text-sm font-medium text-cyan-200 transition hover:bg-cyan-400/20 disabled:cursor-not-allowed disabled:opacity-50"
              disabled={isSubmitting || !cognitoConfig.isConfigured}
            >
              {isSubmitting
                ? "Processing..."
                : mode === "signin"
                  ? "Sign in"
                  : mode === "signup"
                    ? "Create account"
                    : "Verify and sign in"}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
