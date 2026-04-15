# Frontend (Step 3)

AetherForecast trading dashboard frontend built with React + Vite + TypeScript + TailwindCSS.

## Features

- Cosmic dark trading UI inspired by the eye animation identity.
- Lightweight Charts candlestick chart with MA, RSI, MACD, and Bollinger indicators.
- Prediction overlay with confidence interval.
- Zustand market state for symbol/timeframe/token.
- Custom Cognito sign-in/sign-up/verification form (no hosted default UI).
- Axios API client with Cognito JWT bearer token.

## Local Development

1. Install dependencies:
	npm install

2. Start dev server:
	npm run dev

3. Build production bundle:
	npm run build

4. Preview production build:
	npm run preview

## Environment

Set API base URL in .env:

VITE_API_BASE_URL=https://api.example.com
VITE_COGNITO_REGION=ap-southeast-1
VITE_COGNITO_USER_POOL_ID=ap-southeast-1_example
VITE_COGNITO_CLIENT_ID=exampleclientid123
