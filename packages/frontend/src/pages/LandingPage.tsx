import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { useTranslation } from "react-i18next";

import AuthTokenModal from "@/components/ui/AuthTokenModal";
import { useMarketStore } from "@/hooks/useMarketStore";
import logoEye from "@/assets/logo-eye.svg";

export default function LandingPage() {
  const navigate = useNavigate();
  const { t } = useTranslation();
  const highlights = t("landing.highlights", { returnObjects: true }) as string[];
  const { token, setToken, clearToken } = useMarketStore();
  const [showAuthModal, setShowAuthModal] = useState(false);
  const [authMode, setAuthMode] = useState<"signin" | "signup">("signin");

  const handleAuthenticated = (jwt: string) => {
    setToken(jwt);
    setShowAuthModal(false);
    navigate("/dashboard", { replace: true });
  };

  return (
    <div className="cosmic-shell min-h-screen px-4 py-6 lg:px-8 lg:py-10">
      <div className="mx-auto flex min-h-[calc(100vh-3rem)] w-full max-w-6xl items-center justify-center">
        <section className="glass-panel grid w-full gap-8 rounded-3xl p-6 lg:grid-cols-[1.3fr_1fr] lg:p-10">
          <div>
            <div className="mb-5 inline-flex items-center gap-3 rounded-full border border-cyan-300/35 bg-cyan-400/8 px-3 py-1 text-xs tracking-[0.12em] text-cyan-200">
              <img src={logoEye} alt="AetherForecast" className="h-5 w-8" />
              {t("landing.badge")}
            </div>

            <h1 className="max-w-3xl text-3xl font-semibold leading-tight text-violet-50 lg:text-5xl">
              {t("landing.title")}
            </h1>

            <p className="mt-5 max-w-2xl text-sm leading-7 text-violet-100/85 lg:text-base">
              {t("landing.description")}
            </p>

            <div className="mt-8 flex flex-wrap gap-3">
              <button
                type="button"
                onClick={() => {
                  setAuthMode("signin");
                  setShowAuthModal(true);
                }}
                className="rounded-xl border border-cyan-300/80 bg-cyan-400/15 px-5 py-3 text-sm font-semibold text-cyan-100 transition hover:bg-cyan-300/25"
              >
                {t("landing.signIn")}
              </button>
              <button
                type="button"
                onClick={() => {
                  setAuthMode("signup");
                  setShowAuthModal(true);
                }}
                className="rounded-xl border border-violet-300/45 bg-violet-500/10 px-5 py-3 text-sm font-semibold text-violet-100 transition hover:border-violet-200/70"
              >
                {t("landing.signUp")}
              </button>
            </div>
          </div>

          <div className="rounded-2xl border border-violet-300/25 bg-cosmic-900/60 p-5">
            <p className="muted-label">{t("landing.highlightsTitle")}</p>
            <ul className="mt-3 space-y-3 text-sm text-violet-100/90">
              {highlights.map((item) => (
                <li key={item} className="rounded-lg border border-violet-300/20 bg-violet-500/5 px-3 py-2 leading-6">
                  {item}
                </li>
              ))}
            </ul>
            <div className="mt-5 rounded-xl border border-cyan-300/30 bg-cyan-500/8 p-3 text-xs leading-6 text-cyan-100/90">
              {t("landing.protectedNotice")}
            </div>
          </div>
        </section>
      </div>

      <AuthTokenModal
        open={showAuthModal}
        defaultMode={authMode}
        hasToken={Boolean(token)}
        onAuthenticate={handleAuthenticated}
        onSignOut={clearToken}
        onClose={() => setShowAuthModal(false)}
      />
    </div>
  );
}
