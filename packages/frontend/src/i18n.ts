import i18n from "i18next";
import { initReactI18next } from "react-i18next";

import en from "@/locales/en.json";
import vi from "@/locales/vi.json";

export const LANGUAGE_STORAGE_KEY = "aether_lang";

const resources = {
  en: { translation: en },
  vi: { translation: vi },
};

function getStoredLanguage(): "en" | "vi" {
  try {
    const stored = localStorage.getItem(LANGUAGE_STORAGE_KEY);
    if (stored === "en" || stored === "vi") {
      return stored;
    }
  } catch {
    // localStorage may be disabled in some environments.
  }
  return "en";
}

i18n.use(initReactI18next).init({
  resources,
  lng: getStoredLanguage(),
  fallbackLng: "en",
  interpolation: {
    escapeValue: false,
  },
});

i18n.on("languageChanged", (lng) => {
  try {
    localStorage.setItem(LANGUAGE_STORAGE_KEY, lng);
  } catch {
    // Ignore localStorage write failures.
  }
});

export default i18n;
