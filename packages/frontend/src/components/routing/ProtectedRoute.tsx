import { ReactElement } from "react";
import { Navigate, useLocation } from "react-router-dom";

import { useMarketStore } from "@/hooks/useMarketStore";

interface ProtectedRouteProps {
  children: ReactElement;
}

export default function ProtectedRoute({ children }: ProtectedRouteProps) {
  const token = useMarketStore((state) => state.token);
  const location = useLocation();

  if (!token) {
    return <Navigate to="/" replace state={{ from: location.pathname }} />;
  }

  return children;
}
