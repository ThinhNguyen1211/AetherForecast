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
  return (
    <aside className="glass-panel flex h-[72vh] min-h-[28rem] flex-col rounded-2xl p-3 lg:h-[calc(100vh-10.5rem)]">
      <div>
        <p className="muted-label">Markets</p>
        <input
          value={searchQuery}
          onChange={(event) => onSearchQueryChange(event.target.value.toUpperCase())}
          placeholder="Filter symbol"
          className="mt-2 w-full rounded-lg border border-violet-400/35 bg-cosmic-900/70 px-3 py-2 text-sm outline-none ring-neon-cyan/50 transition focus:ring"
        />
      </div>

      <div className="scrollbar-slim mt-3 flex-1 overflow-y-auto rounded-lg border border-violet-400/20 bg-cosmic-900/50">
        {loading ? (
          <p className="px-3 py-4 text-sm text-violet-200/75">Loading symbols...</p>
        ) : symbols.length === 0 ? (
          <p className="px-3 py-4 text-sm text-violet-200/75">No symbols available.</p>
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
              <span className="text-[10px] uppercase tracking-[0.14em] text-violet-300/70">spot</span>
            </button>
          ))
        )}
      </div>
    </aside>
  );
}
