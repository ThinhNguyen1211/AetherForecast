import React from "react";
import ReactDOM from "react-dom/client";
import { BrowserRouter } from "react-router-dom";
import App from "./App";
import "./index.css";

interface AppErrorBoundaryState {
  hasError: boolean;
}

class AppErrorBoundary extends React.Component<React.PropsWithChildren, AppErrorBoundaryState> {
  constructor(props: React.PropsWithChildren) {
    super(props);
    this.state = { hasError: false };
  }

  static getDerivedStateFromError(): AppErrorBoundaryState {
    return { hasError: true };
  }

  componentDidCatch(error: unknown): void {
    console.error("[AetherForecast] Unhandled render error", error);
  }

  render(): React.ReactNode {
    if (this.state.hasError) {
      return (
        <div className="cosmic-shell min-h-screen px-4 py-10">
          <div className="mx-auto max-w-3xl rounded-2xl border border-rose-400/45 bg-rose-500/10 p-6 text-rose-100">
            <h1 className="text-lg font-semibold">Dashboard failed to render.</h1>
            <p className="mt-2 text-sm text-rose-100/90">
              Please refresh the page. If the issue continues, clear browser cache and try again.
            </p>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <AppErrorBoundary>
      <BrowserRouter>
        <App />
      </BrowserRouter>
    </AppErrorBoundary>
  </React.StrictMode>,
);
