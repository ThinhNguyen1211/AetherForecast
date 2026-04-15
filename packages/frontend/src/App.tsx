import { Navigate, Route, Routes } from "react-router-dom";

import ProtectedRoute from "@/components/routing/ProtectedRoute";
import { useMarketStore } from "@/hooks/useMarketStore";
import Dashboard from "@/pages/Dashboard";
import LandingPage from "@/pages/LandingPage";

export default function App() {
  const token = useMarketStore((state) => state.token);

  return (
    <Routes>
      <Route path="/" element={token ? <Navigate to="/dashboard" replace /> : <LandingPage />} />
      <Route
        path="/dashboard"
        element={
          <ProtectedRoute>
            <Dashboard />
          </ProtectedRoute>
        }
      />
      <Route path="*" element={<Navigate to={token ? "/dashboard" : "/"} replace />} />
    </Routes>
  );
}
