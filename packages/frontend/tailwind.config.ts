import type { Config } from "tailwindcss";

export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        cosmic: {
          950: "#0b0616",
          900: "#12081f",
          800: "#1d1035",
        },
        neon: {
          cyan: "#00ffff",
          purple: "#8c52ff",
          lilac: "#b088f5",
        },
      },
      fontFamily: {
        sans: ["Sora", "Inter", "sans-serif"],
        mono: ["Space Mono", "monospace"],
      },
      boxShadow: {
        glow: "0 0 0 1px rgba(140,82,255,0.3), 0 0 24px rgba(0,255,255,0.12)",
      },
      backgroundImage: {
        cosmic:
          "radial-gradient(1200px 700px at 75% -10%, rgba(140,82,255,0.22), transparent 50%), radial-gradient(900px 500px at -10% 90%, rgba(0,255,255,0.16), transparent 45%), linear-gradient(145deg, #0b0616 0%, #12081f 55%, #1d1035 100%)",
      },
    },
  },
  plugins: [],
} satisfies Config;
