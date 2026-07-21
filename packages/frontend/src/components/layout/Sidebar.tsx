import { useTranslation } from "react-i18next";

interface SidebarProps {
  symbols: string[];
  selectedSymbol: string;
  searchQuery: string;
  loading: boolean;
  onSearchQueryChange: (value: string) => void;
  onSelectSymbol: (symbol: string) => void;
}

export default function Sidebar({
  symbols,
  selectedSymbol,
  searchQuery,
  loading,
  onSearchQueryChange,
  onSelectSymbol,
}: SidebarProps) {
  const { t } = useTranslation();

  return (
    <aside className="glass-panel flex h-full flex-col rounded-2xl p-3">
      <div>
        <p className="muted-label">{t("sidebar.markets")}</p>
        <input
          value={searchQuery}
          onChange={(event) => onSearchQueryChange(event.target.value.toUpperCase())}
          placeholder={t("sidebar.filterPlaceholder")}
          className="mt-2 w-full rounded-lg border border-violet-400/35 bg-cosmic-900/70 px-3 py-2 text-sm outline-none ring-neon-cyan/50 transition focus:ring"
        />
      </div>

      <div className="scrollbar-slim mt-3 flex-1 overflow-y-auto rounded-lg border border-violet-400/20 bg-cosmic-900/50">
        {loading ? (
          <p className="px-3 py-4 text-sm text-violet-200/75">{t("sidebar.loadingSymbols")}</p>
        ) : symbols.length === 0 ? (
          <p className="px-3 py-4 text-sm text-violet-200/75">{t("sidebar.noSymbols")}</p>
        ) : (
          symbols.map((symbol) => (
            <button
              type="button"
              key={symbol}
              onClick={() => onSelectSymbol(symbol)}
              className={`flex w-full items-center justify-between border-b border-violet-400/10 px-3 py-2 text-left text-sm transition ${
                selectedSymbol === symbol
                  ? "bg-cyan-400/12 text-cyan-100"
                  : "text-violet-100 hover:bg-violet-500/10"
              }`}
            >
              <span className="font-medium tracking-wide">{symbol}</span>
              <span className="text-[10px] uppercase tracking-[0.14em] text-violet-300/70">{t("sidebar.spot")}</span>
            </button>
          ))
        )}
      </div>
    </aside>
  );
}
