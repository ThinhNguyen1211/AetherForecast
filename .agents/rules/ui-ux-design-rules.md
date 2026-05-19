---
trigger: always_on
---

# UI/UX Design Rules for Trading/ML Dashboard

## General UI/UX Philosophy
- Học từ Binance, TradingView, Coinbase: sạch sẽ, thông tin dày đặc nhưng không rối, focus vào chart.
- Ưu tiên **speed & clarity**: user phải thấy chart ngay trong <1.5s khi đổi coin.
- Cosmic/dark theme: nền tối (#0a0a0f), accent tím neon (#a855f7), xanh cyan (#22d3ee), glow nhẹ.
- Typography: Inter hoặc system font, font-weight rõ ràng, kích thước hợp lý.

## Chart Specific
- Sử dụng Lightweight Charts hoặc tối ưu tương tự TradingView.
- Realtime: update mượt, chỉ update last candle, không gây giật viewport khi user kéo lịch sử.
- Lazy loading: initial load 600-800 candles nhanh → background load thêm khi scroll.
- Prediction overlay: tím mờ (opacity 0.75-0.85), variance thực tế, confidence bands rõ ràng.
- Indicators: RSI, MACD, Bollinger, volume – có thể toggle.

## Component & Layout Rules
- Sidebar: searchable symbols list, volume/price thay đổi.
- Prediction panel: sentiment score rõ nguồn (market + external), horizon selector, confidence interval.
- Loading: shimmer/skeleton trên chart area, không hiện "Syncing..." khi lazy load.
- Error: graceful, retry button, không crash toàn dashboard.

## Best Practices từ ông lớn
- Binance: minimal icons, dark mode chuyên nghiệp, realtime cực mượt.
- TradingView: chart tương tác mạnh, overlay thông minh, nhiều timeframe.
- Apple/Google: attention to micro-interactions, consistent spacing, high polish.
- Luôn nghĩ "mobile-first" nhưng ưu tiên desktop trading experience.

## Accessibility & Polish
- ARIA labels cho chart elements.
- Keyboard navigation.
- Hover states, focus states rõ ràng.
- Consistent spacing (Tailwind spacing scale).