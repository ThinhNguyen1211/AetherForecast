from functools import lru_cache
import time
from typing import Any

import requests
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt

from src.core.config import Settings, get_settings


bearer_scheme = HTTPBearer(auto_error=False)


class CognitoJWTVerifier:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._jwks: dict[str, dict[str, Any]] = {}
        self._jwks_last_loaded = 0.0

    def _jwks_url(self) -> str:
        return f"{self.settings.cognito_issuer}/.well-known/jwks.json"

    def _load_jwks(self, force_refresh: bool = False) -> None:
        cache_ttl_seconds = 3600
        cache_expired = (time.time() - self._jwks_last_loaded) > cache_ttl_seconds

        if self._jwks and not cache_expired and not force_refresh:
            return

        response = requests.get(self._jwks_url(), timeout=5)
        response.raise_for_status()
        payload = response.json()

        keys = payload.get("keys", [])
        self._jwks = {str(item["kid"]): item for item in keys if "kid" in item}
        self._jwks_last_loaded = time.time()

    def verify_token(self, token: str) -> dict[str, Any]:
        if not self.settings.cognito_user_pool_id:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Cognito User Pool is not configured",
            )

        try:
            unverified_header = jwt.get_unverified_header(token)
            kid = str(unverified_header.get("kid", ""))

            self._load_jwks()
            key = self._jwks.get(kid)
            if key is None:
                self._load_jwks(force_refresh=True)
                key = self._jwks.get(kid)

            if key is None:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Unknown token signing key",
                )

            claims = jwt.decode(
                token,
                key,
                algorithms=["RS256"],
                issuer=self.settings.cognito_issuer,
                options={"verify_aud": False},
            )

            token_use = claims.get("token_use")
            if token_use not in {"id", "access"}:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid token_use claim",
                )

            client_id = self.settings.cognito_client_id
            if client_id and token_use == "id" and claims.get("aud") != client_id:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Token audience does not match app client",
                )

            if client_id and token_use == "access" and claims.get("client_id") != client_id:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Token client_id does not match app client",
                )

            return claims
        except HTTPException:
            raise
        except (JWTError, requests.RequestException) as exc:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"Token validation failed: {exc}",
            ) from exc


@lru_cache
def get_cognito_verifier() -> CognitoJWTVerifier:
    return CognitoJWTVerifier(get_settings())


def require_authenticated_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
    verifier: CognitoJWTVerifier = Depends(get_cognito_verifier),
) -> dict[str, Any]:
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Bearer token is required",
        )
    return verifier.verify_token(credentials.credentials)
