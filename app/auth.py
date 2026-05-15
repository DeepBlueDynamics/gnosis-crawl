"""
Authentication integration with nuts-auth service.
Validates ahp_ API tokens by exchanging them at nuts-auth /auth endpoint.
"""
import base64
import hmac
import hashlib
import json
import logging
import httpx
from datetime import datetime, timezone
from typing import Dict, Optional
from fastapi import HTTPException, Header, Depends

from app.config import settings

logger = logging.getLogger(__name__)


def validate_token_from_query(token: str, secret_key: str) -> Dict:
    """Validates a short-lived HMAC-signed internal token from a query parameter."""
    if not secret_key:
        raise ValueError("Secret key cannot be empty.")
    if not token:
        raise HTTPException(status_code=401, detail="Token cannot be empty.")
    try:
        token_parts = token.split(".")
        if len(token_parts) != 2:
            raise HTTPException(status_code=401, detail="Invalid token format.")
        encoded_payload, encoded_signature = token_parts
        sig_gen = hmac.new(secret_key.encode(), encoded_payload.encode(), hashlib.sha256)
        expected_sig = base64.urlsafe_b64encode(sig_gen.digest()).rstrip(b'=').decode()
        if not hmac.compare_digest(encoded_signature, expected_sig):
            raise HTTPException(status_code=401, detail="Invalid token signature.")
        padding = 4 - (len(encoded_payload) % 4)
        if padding != 4:
            encoded_payload += '=' * padding
        payload = json.loads(base64.urlsafe_b64decode(encoded_payload).decode())
        if 'exp' in payload:
            exp_time = datetime.fromisoformat(payload['exp'].replace('Z', '+00:00'))
            if datetime.now(timezone.utc) > exp_time:
                raise HTTPException(status_code=401, detail="Token has expired.")
        return payload
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Internal token validation failed: {e}")
        raise HTTPException(status_code=401, detail="Token validation failed.")


class AuthClient:
    """Validates ahp_ tokens via nuts-auth /auth exchange endpoint."""

    def __init__(self):
        self.auth_url = settings.gnosis_auth_url.rstrip("/")

    async def validate_token(self, token: str) -> Dict:
        # JWTs from the browser magic-link flow are 3-part base64-dot strings.
        # ahp_ tokens are long opaque strings prefixed with "ahp_".
        is_jwt = token.count(".") == 2 and not token.startswith("ahp_")
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                if is_jwt:
                    # Verify the JWT against nuts-auth and decode claims.
                    resp = await client.get(
                        f"{self.auth_url}/api/verify",
                        headers={"Authorization": f"Bearer {token}"},
                    )
                    if resp.status_code != 200:
                        raise HTTPException(status_code=401, detail="Invalid or expired token")
                    payload = resp.json()
                else:
                    # ahp_ token → exchange for a fresh JWT.
                    resp = await client.post(
                        f"{self.auth_url}/auth",
                        data={"token": token},
                    )
                    if resp.status_code != 200:
                        raise HTTPException(status_code=401, detail="Invalid or inactive token")
                    jwt_token = resp.json().get("access_token", "")
                    payload = json.loads(base64.urlsafe_b64decode(jwt_token.split(".")[1] + "=="))

            email = payload.get("sub", "unknown@grub-crawl.local")
            logger.info(f"Token validated for: {email} (jwt={is_jwt})")

            return {
                "subject": email,
                "email": email,
                "user_id": payload.get("user_id", ""),
                "scopes": payload.get("scopes", ["crawl:*"]),
                "valid": True,
            }

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Token validation failed: {e}")
            raise HTTPException(status_code=401, detail="Invalid or expired token")


# Global auth client instance
auth_client = AuthClient()


async def get_current_user(authorization: str = Header(None)) -> Dict:
    """
    FastAPI dependency to get current authenticated user
    
    Args:
        authorization: Authorization header with Bearer token
        
    Returns:
        Dict with user information
        
    Raises:
        HTTPException: If authentication fails
    """
    if not authorization:
        raise HTTPException(
            status_code=401,
            detail="Missing Authorization header",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    if not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail="Invalid authorization header format",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    token = authorization.split(" ")[1]
    if not token:
        raise HTTPException(
            status_code=401,
            detail="Missing bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return await auth_client.validate_token(token)


async def get_user_email(user: Dict = Depends(get_current_user)) -> str:
    """
    Extract user email from authenticated user info
    
    Args:
        user: User info from get_current_user
        
    Returns:
        User email string
    """
    # Extract email from subject or email field
    if "email" in user:
        return user["email"]
    elif user.get("subject", "").startswith("user:"):
        return user["subject"][5:]  # Remove "user:" prefix
    else:
        return "unknown@grub-crawl.local"


def get_customer_identifier(customer_id: Optional[str] = None, user_email: Optional[str] = None) -> str:
    """
    Resolve customer identifier from either customer_id or user_email.
    Prioritizes customer_id if provided, falls back to user_email.
    
    Args:
        customer_id: Optional customer ID from request
        user_email: Optional user email from auth
        
    Returns:
        Customer identifier string for storage partitioning
    """
    if customer_id:
        return customer_id
    elif user_email:
        return user_email
    else:
        return "anonymous@grub-crawl.local"
