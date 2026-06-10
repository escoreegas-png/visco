"""
Production-grade FastAPI backend for Visco Compare.
Adds: phone-exists check, no-verification password reset, wishlist hydration.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import os
import re
import secrets
import time
from collections import defaultdict
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from difflib import SequenceMatcher
from typing import Any, Awaitable, Callable, Optional
from urllib.parse import quote, urlparse

import bcrypt
import httpx
import jwt
from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import BaseModel, Field
from starlette.middleware.base import BaseHTTPMiddleware

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("comparator")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
ADMIN_API_KEY = os.getenv("ADMIN_API_KEY")
JWT_SECRET = os.getenv("JWT_SECRET", secrets.token_urlsafe(64))
JWT_ALG = "HS256"
JWT_EXPIRES_DAYS = 30
PORT = int(os.getenv("PORT", "8000"))
OFFER_SIGNING_SECRET = os.getenv("OFFER_SIGNING_SECRET", secrets.token_urlsafe(48))

CORS_ALLOWED_ORIGINS = [
    o.strip() for o in os.getenv("CORS_ALLOWED_ORIGINS", "*").split(",") if o.strip()
]

PRODUCTS_TABLE = "products"
USERS_TABLE = "users"
WISHLIST_TABLE = "user_wishlist"
ALERTS_TABLE = "user_price_alerts"

CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "120"))
SEARCH_CACHE_TTL = int(os.getenv("SEARCH_CACHE_TTL", "60"))

RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", "120"))
RATE_LIMIT_WINDOW = int(os.getenv("RATE_LIMIT_WINDOW", "60"))
AUTH_RATE_LIMIT = int(os.getenv("AUTH_RATE_LIMIT", "10"))
RESET_RATE_LIMIT = int(os.getenv("RESET_RATE_LIMIT", "5"))

if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
    logger.warning("SUPABASE_URL / SUPABASE_SERVICE_KEY not set — DB calls will fail.")

# ---------------------------------------------------------------------------
# Caches & analytics
# ---------------------------------------------------------------------------
class TTLCache:
    def __init__(self, default_ttl: int = 120, max_size: int = 2048) -> None:
        self._store: dict[str, tuple[float, Any]] = {}
        self._lock = asyncio.Lock()
        self._default_ttl = default_ttl
        self._max_size = max_size

    async def get(self, key: str) -> Any | None:
        async with self._lock:
            item = self._store.get(key)
            if not item:
                return None
            expires, value = item
            if expires < time.time():
                self._store.pop(key, None)
                return None
            return value

    async def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        async with self._lock:
            if len(self._store) >= self._max_size:
                cutoff = sorted(self._store.items(), key=lambda kv: kv[1][0])[
                    : max(1, self._max_size // 10)
                ]
                for k, _ in cutoff:
                    self._store.pop(k, None)
            self._store[key] = (time.time() + (ttl or self._default_ttl), value)

    async def clear(self) -> None:
        async with self._lock:
            self._store.clear()

    def size(self) -> int:
        return len(self._store)


cache = TTLCache(default_ttl=CACHE_TTL_SECONDS)

_offer_registry: dict[str, dict[str, Any]] = {}
_offer_lock = asyncio.Lock()

_search_analytics: dict[str, int] = defaultdict(int)
_click_analytics: dict[str, int] = defaultdict(int)
_click_log: list[dict[str, Any]] = []
_MAX_CLICK_LOG = 5000

_rate_buckets: dict[str, list[float]] = defaultdict(list)
_rate_lock = asyncio.Lock()


# ---------------------------------------------------------------------------
# Supabase client
# ---------------------------------------------------------------------------
class SupabaseClient:
    def __init__(self, url: str, key: str) -> None:
        self._base = f"{url}/rest/v1" if url else ""
        self._headers = {
            "apikey": key,
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        self._client: httpx.AsyncClient | None = None

    async def start(self) -> None:
        if not self._client:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(15.0, connect=5.0),
                limits=httpx.Limits(max_connections=50, max_keepalive_connections=20),
                headers=self._headers,
            )

    async def stop(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    async def _request(
        self,
        method: str,
        path: str,
        params: dict[str, Any] | None = None,
        body: Any | None = None,
        prefer: str | None = None,
        retries: int = 2,
    ) -> tuple[list[dict[str, Any]], int]:
        if not self._client or not self._base:
            raise HTTPException(status_code=503, detail="Database unavailable.")

        url = f"{self._base}{path}"
        query_parts: list[str] = []
        if params:
            for key, value in params.items():
                query_parts.append(
                    f"{quote(str(key), safe='')}={quote(str(value), safe='(),.*:')}"
                )
        query_string = "&".join(query_parts)
        full_url = f"{url}?{query_string}" if query_string else url

        headers = {}
        if prefer:
            headers["Prefer"] = prefer
        else:
            headers["Prefer"] = "count=exact"

        last_exc: Exception | None = None
        for attempt in range(retries + 1):
            try:
                resp = await self._client.request(
                    method, full_url, json=body, headers=headers
                )
                if resp.status_code >= 500:
                    raise httpx.HTTPStatusError(
                        "server error", request=resp.request, response=resp
                    )
                if resp.status_code >= 400:
                    logger.warning(
                        "Supabase %s %s -> %s | body=%s",
                        method,
                        path,
                        resp.status_code,
                        resp.text[:300],
                    )
                    if resp.status_code == 409:
                        raise HTTPException(status_code=409, detail="Conflict.")
                    raise HTTPException(status_code=502, detail="Upstream error.")
                content_range = resp.headers.get("content-range", "")
                total = 0
                if "/" in content_range:
                    try:
                        total = int(content_range.split("/")[-1])
                    except ValueError:
                        total = 0
                data = resp.json() if resp.content else []
                if not isinstance(data, list):
                    data = [data] if data else []
                return data, total
            except (httpx.TransportError, httpx.HTTPStatusError) as exc:
                last_exc = exc
                if attempt < retries:
                    await asyncio.sleep(0.3 * (attempt + 1))
                    continue
                logger.error("Supabase request failed: %s", exc)
                raise HTTPException(
                    status_code=503, detail="Database temporarily unavailable."
                )
        raise HTTPException(
            status_code=503, detail=str(last_exc) if last_exc else "DB error"
        )

    async def select(
        self,
        table: str = PRODUCTS_TABLE,
        columns: str = "*",
        filters: dict[str, str] | None = None,
        order: str | None = None,
        limit: int | None = None,
        offset: int | None = None,
    ) -> tuple[list[dict[str, Any]], int]:
        params: dict[str, Any] = {"select": columns}
        if filters:
            params.update(filters)
        if order:
            params["order"] = order
        if limit is not None:
            params["limit"] = str(limit)
        if offset is not None:
            params["offset"] = str(offset)
        return await self._request("GET", f"/{table}", params=params)

    async def insert(self, table: str, row: dict[str, Any]) -> dict[str, Any]:
        data, _ = await self._request(
            "POST",
            f"/{table}",
            body=row,
            prefer="return=representation",
        )
        return data[0] if data else {}

    async def update(
        self, table: str, filters: dict[str, str], patch: dict[str, Any]
    ) -> list[dict[str, Any]]:
        data, _ = await self._request(
            "PATCH",
            f"/{table}",
            params=filters,
            body=patch,
            prefer="return=representation",
        )
        return data

    async def delete(self, table: str, filters: dict[str, str]) -> None:
        await self._request("DELETE", f"/{table}", params=filters, prefer="return=minimal")


db = SupabaseClient(SUPABASE_URL, SUPABASE_SERVICE_KEY)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
SAFE_COLUMNS = (
    "title,image,rating,reviews,offers,price,price_history,"
    "main_category,sub_category,product_type,brand,search_tags,seo_keywords,"
    "normalized_product_id,lowest_price,highest_price,average_price,"
    "match_score,last_seen_at,created_at"
)


def _safe_json(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (list, dict)):
        return value
    if isinstance(value, (bytes, bytearray)):
        try:
            value = value.decode("utf-8", errors="ignore")
        except Exception:
            return None
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            return None
    return None


def _to_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        cleaned = re.sub(r"[^\d.\-]", "", value)
        try:
            return float(cleaned) if cleaned else default
        except ValueError:
            return default
    return default


def _to_int(value: Any, default: int = 0) -> int:
    return int(_to_float(value, default))


def _store_from_url(url: str) -> str:
    try:
        host = urlparse(url).hostname or ""
        host = host.lower().replace("www.", "")
        parts = host.split(".")
        if len(parts) >= 2:
            return parts[-2].capitalize()
        return host.capitalize() or "Store"
    except Exception:
        return "Store"


def _sign_offer_id(payload: str) -> str:
    digest = hmac.new(
        OFFER_SIGNING_SECRET.encode(), payload.encode(), hashlib.sha256
    ).hexdigest()[:20]
    return f"offer_{digest}"


async def _register_offer(affiliate_url: str, store: str, product_id: str) -> str:
    if not affiliate_url:
        return ""
    payload = f"{product_id}|{store}|{affiliate_url}"
    offer_id = _sign_offer_id(payload)
    async with _offer_lock:
        _offer_registry[offer_id] = {
            "url": affiliate_url,
            "store": store,
            "product_id": product_id,
            "created_at": time.time(),
        }
    return offer_id


async def _normalize_offers(
    raw_offers: Any,
    fallback_affiliate: str | None,
    fallback_url: str | None,
    product_id: str,
) -> list[dict[str, Any]]:
    parsed = _safe_json(raw_offers) or []
    if isinstance(parsed, dict):
        parsed = [parsed]

    if not parsed and (fallback_affiliate or fallback_url):
        parsed = [
            {
                "store": _store_from_url(fallback_affiliate or fallback_url or ""),
                "price": None,
                "affiliate_url": fallback_affiliate or fallback_url,
                "url": fallback_url,
            }
        ]

    best_by_store: dict[str, dict[str, Any]] = {}
    for item in parsed:
        if not isinstance(item, dict):
            continue
        affiliate = (
            item.get("affiliate_url")
            or item.get("affiliateUrl")
            or item.get("url")
            or item.get("link")
            or fallback_affiliate
            or fallback_url
        )
        if not affiliate:
            continue
        store_raw = item.get("store") or item.get("merchant") or item.get("source")
        store = (store_raw or _store_from_url(str(affiliate))).strip().title()
        price = _to_float(item.get("price") or item.get("amount") or item.get("value"))
        if price <= 0:
            continue

        existing = best_by_store.get(store)
        if not existing or price < existing["_price"]:
            best_by_store[store] = {
                "_price": price,
                "_affiliate": str(affiliate),
                "store": store,
            }

    results: list[dict[str, Any]] = []
    for store, entry in best_by_store.items():
        offer_id = await _register_offer(entry["_affiliate"], store, product_id)
        results.append(
            {
                "offerId": offer_id,
                "store": store,
                "price": round(entry["_price"], 2),
            }
        )

    results.sort(key=lambda o: o["price"])
    return results


def _normalize_price_history(raw: Any) -> list[dict[str, Any]]:
    parsed = _safe_json(raw) or []
    if isinstance(parsed, dict):
        parsed = [{"date": k, "price": v} for k, v in parsed.items()]
    cleaned: list[dict[str, Any]] = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        date_raw = item.get("date") or item.get("timestamp") or item.get("time")
        price = _to_float(item.get("price") or item.get("value"))
        if not date_raw or price <= 0:
            continue
        try:
            if isinstance(date_raw, (int, float)):
                dt = datetime.fromtimestamp(float(date_raw), tz=timezone.utc)
            else:
                dt = datetime.fromisoformat(str(date_raw).replace("Z", "+00:00"))
        except Exception:
            continue
        entry = {"date": dt.date().isoformat(), "price": round(price, 2)}
        store = item.get("store") or item.get("merchant")
        if store:
            entry["store"] = str(store)
        cleaned.append(entry)

    cleaned.sort(key=lambda e: e["date"])
    return cleaned


async def _transform_product(row: dict[str, Any], include_extras: bool = True) -> dict[str, Any]:
    product_id = str(row.get("normalized_product_id") or "")
    offers = await _normalize_offers(
        row.get("offers"),
        row.get("affiliate_url"),
        row.get("url"),
        product_id,
    )

    prices = [o["price"] for o in offers]
    lowest = _to_float(row.get("lowest_price")) or (min(prices) if prices else 0.0)
    highest = _to_float(row.get("highest_price")) or (max(prices) if prices else 0.0)
    average = _to_float(row.get("average_price")) or (
        round(sum(prices) / len(prices), 2) if prices else 0.0
    )

    public: dict[str, Any] = {
        "id": product_id,
        "title": row.get("title") or "",
        "image": row.get("image") or "",
        "rating": round(_to_float(row.get("rating")), 2),
        "reviews": _to_int(row.get("reviews")),
        "lowestPrice": round(lowest, 2),
        "highestPrice": round(highest, 2),
        "averagePrice": round(average, 2),
        "offers": offers,
    }

    if include_extras:
        public.update(
            {
                "brand": row.get("brand") or "",
                "mainCategory": row.get("main_category") or "",
                "subCategory": row.get("sub_category") or "",
                "productType": row.get("product_type") or "",
            }
        )

    return public


# ---------------------------------------------------------------------------
# Auth helpers
# ---------------------------------------------------------------------------
PHONE_RE = re.compile(r"^\+?\d{10,15}$")


def _normalize_phone(phone: str) -> str:
    p = re.sub(r"[\s\-()]", "", phone or "")
    if not p.startswith("+") and len(p) == 10:
        p = "+91" + p
    return p


def _hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt(rounds=12)).decode("utf-8")


def _verify_password(password: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8"), hashed.encode("utf-8"))
    except Exception:
        return False


def _create_token(user_id: str, phone: str) -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "sub": user_id,
        "phone": phone,
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(days=JWT_EXPIRES_DAYS)).timestamp()),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALG)


def _decode_token(token: str) -> dict[str, Any]:
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALG])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Session expired.")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid session.")


async def get_current_user(
    authorization: Optional[str] = Header(default=None),
) -> dict[str, Any]:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Authentication required.")
    token = authorization.split(" ", 1)[1].strip()
    payload = _decode_token(token)
    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid session.")
    return {"id": user_id, "phone": payload.get("phone")}


async def _ip_rate_limit(request: Request, bucket_prefix: str, limit: int) -> None:
    ip = (
        request.headers.get("x-forwarded-for", "").split(",")[0].strip()
        or (request.client.host if request.client else "anon")
    )
    key = f"{bucket_prefix}:{ip}"
    now = time.time()
    window_start = now - 60
    async with _rate_lock:
        bucket = _rate_buckets[key]
        bucket[:] = [t for t in bucket if t > window_start]
        if len(bucket) >= limit:
            raise HTTPException(status_code=429, detail="Too many attempts. Try again soon.")
        bucket.append(now)


# ---------------------------------------------------------------------------
# Search engine
# ---------------------------------------------------------------------------
_WORD_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _WORD_RE.findall((text or "").lower())


def _fuzzy_ratio(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def _score_product(row: dict[str, Any], query: str, q_tokens: list[str]) -> float:
    title = (row.get("title") or "").lower()
    brand = (row.get("brand") or "").lower()
    tags_parsed = _safe_json(row.get("search_tags"))
    if isinstance(tags_parsed, list):
        tags = " ".join(str(t) for t in tags_parsed).lower()
    else:
        tags = str(row.get("search_tags") or "").lower()
    keywords_parsed = _safe_json(row.get("seo_keywords"))
    if isinstance(keywords_parsed, list):
        keywords = " ".join(str(k) for k in keywords_parsed).lower()
    else:
        keywords = str(row.get("seo_keywords") or "").lower()
    haystack_meta = (
        f"{tags} {keywords} {row.get('main_category','')} "
        f"{row.get('sub_category','')} {row.get('product_type','')}"
    ).lower()

    score = 0.0
    if query and query in title:
        score += 5.0
    if query and query in brand:
        score += 2.0

    for tok in q_tokens:
        if tok in title:
            score += 2.0
        if tok in brand:
            score += 1.5
        if tok in haystack_meta:
            score += 0.8

    score += _fuzzy_ratio(query, title) * 3.0
    score += _fuzzy_ratio(query, brand) * 1.0

    score += min(_to_float(row.get("rating")), 5.0) * 0.2
    score += min(_to_int(row.get("reviews")) / 1000.0, 2.0)

    return score


# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------
class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable[[Request], Awaitable[Any]]):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["X-XSS-Protection"] = "0"
        if "server" in response.headers:
            try:
                del response.headers["server"]
            except KeyError:
                pass
        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable[[Request], Awaitable[Any]]):
        if request.url.path == "/health":
            return await call_next(request)

        client_ip = (
            request.headers.get("x-forwarded-for", "").split(",")[0].strip()
            or (request.client.host if request.client else "anonymous")
        )
        key = f"{client_ip}:{request.url.path.split('/')[1] if '/' in request.url.path else ''}"
        now = time.time()
        window_start = now - RATE_LIMIT_WINDOW

        async with _rate_lock:
            bucket = _rate_buckets[key]
            bucket[:] = [t for t in bucket if t > window_start]
            if len(bucket) >= RATE_LIMIT_REQUESTS:
                retry_after = int(bucket[0] + RATE_LIMIT_WINDOW - now) + 1
                return JSONResponse(
                    status_code=429,
                    content={"success": False, "error": "Too many requests."},
                    headers={"Retry-After": str(max(1, retry_after))},
                )
            bucket.append(now)

        return await call_next(request)


class AccessLogMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable[[Request], Awaitable[Any]]):
        start = time.time()
        try:
            response = await call_next(request)
        except Exception:
            elapsed = (time.time() - start) * 1000
            logger.exception(
                "Unhandled error on %s %s (%.1fms)",
                request.method,
                request.url.path,
                elapsed,
            )
            raise
        elapsed = (time.time() - start) * 1000
        logger.info(
            "%s %s -> %s (%.1fms)",
            request.method,
            request.url.path,
            response.status_code,
            elapsed,
        )
        return response


# ---------------------------------------------------------------------------
# App lifespan
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    await db.start()
    logger.info("Backend started.")
    try:
        yield
    finally:
        await db.stop()
        logger.info("Backend stopped.")


app = FastAPI(
    title="API",
    version="1.1.0",
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
    lifespan=lifespan,
)

app.add_middleware(AccessLogMiddleware)
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(RateLimitMiddleware)
app.add_middleware(GZipMiddleware, minimum_size=512)
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOWED_ORIGINS or ["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
    max_age=600,
)


@app.exception_handler(HTTPException)
async def http_exception_handler(_: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"success": False, "error": exc.detail},
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(_: Request, exc: Exception):
    logger.exception("Unhandled: %s", exc)
    return JSONResponse(
        status_code=500,
        content={"success": False, "error": "Internal server error."},
    )


def _ok(data: Any, **extra: Any) -> dict[str, Any]:
    return {"success": True, "data": data, **extra}


def require_admin(x_admin_key: Optional[str] = Header(default=None, alias="X-Admin-Key")) -> None:
    if not x_admin_key or not ADMIN_API_KEY or not hmac.compare_digest(x_admin_key, ADMIN_API_KEY):
        raise HTTPException(status_code=404, detail="Not found.")


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------
class SignupBody(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    phone: str = Field(min_length=8, max_length=20)
    password: str = Field(min_length=4, max_length=128)


class LoginBody(BaseModel):
    phone: str = Field(min_length=8, max_length=20)
    password: str = Field(min_length=1, max_length=128)


class CheckPhoneBody(BaseModel):
    phone: str = Field(min_length=8, max_length=20)


class ResetPasswordBody(BaseModel):
    phone: str = Field(min_length=8, max_length=20)
    new_password: str = Field(min_length=4, max_length=128)


class WishlistBody(BaseModel):
    product_id: str = Field(min_length=1, max_length=200)


class AlertBody(BaseModel):
    product_id: str = Field(min_length=1, max_length=200)
    target_price: float = Field(gt=0, lt=10_000_000)


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------
@app.get("/health")
async def health() -> dict[str, Any]:
    return _ok(
        {
            "status": "ok",
            "time": datetime.now(timezone.utc).isoformat(),
        }
    )


@app.get("/favicon.ico")
async def favicon() -> JSONResponse:
    return JSONResponse(status_code=204, content=None)


# ---------------------------------------------------------------------------
# Auth routes
# ---------------------------------------------------------------------------
@app.post("/auth/check-phone")
async def auth_check_phone(body: CheckPhoneBody, request: Request) -> dict[str, Any]:
    """Tell the client whether an account already exists for this phone."""
    await _ip_rate_limit(request, "checkphone", 20)
    phone = _normalize_phone(body.phone)
    if not PHONE_RE.match(phone):
        raise HTTPException(status_code=400, detail="Invalid phone number.")
    rows, _ = await db.select(
        table=USERS_TABLE,
        columns="id",
        filters={"phone": f"eq.{phone}"},
        limit=1,
    )
    return _ok({"exists": bool(rows)})


@app.post("/auth/signup")
async def auth_signup(body: SignupBody, request: Request) -> dict[str, Any]:
    await _ip_rate_limit(request, "auth", AUTH_RATE_LIMIT)

    phone = _normalize_phone(body.phone)
    if not PHONE_RE.match(phone):
        raise HTTPException(status_code=400, detail="Invalid phone number.")

    name = body.name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="Name required.")

    existing, _ = await db.select(
        table=USERS_TABLE,
        columns="id",
        filters={"phone": f"eq.{phone}"},
        limit=1,
    )
    if existing:
        raise HTTPException(
            status_code=409,
            detail="An account with this phone already exists. Please log in instead.",
        )

    pw_hash = _hash_password(body.password)
    user = await db.insert(
        USERS_TABLE,
        {"name": name, "phone": phone, "password_hash": pw_hash},
    )
    if not user:
        raise HTTPException(status_code=500, detail="Could not create account.")

    token = _create_token(str(user["id"]), phone)
    return _ok(
        {
            "token": token,
            "user": {"id": user["id"], "name": user["name"], "phone": user["phone"]},
        }
    )


@app.post("/auth/login")
async def auth_login(body: LoginBody, request: Request) -> dict[str, Any]:
    await _ip_rate_limit(request, "auth", AUTH_RATE_LIMIT)

    phone = _normalize_phone(body.phone)
    if not PHONE_RE.match(phone):
        raise HTTPException(status_code=400, detail="Invalid phone number.")

    rows, _ = await db.select(
        table=USERS_TABLE,
        columns="id,name,phone,password_hash",
        filters={"phone": f"eq.{phone}"},
        limit=1,
    )
    if not rows:
        raise HTTPException(
            status_code=404,
            detail="No account found with this phone. Please sign up first.",
        )

    user = rows[0]
    if not _verify_password(body.password, user.get("password_hash") or ""):
        raise HTTPException(status_code=401, detail="Incorrect password.")

    try:
        await db.update(
            USERS_TABLE,
            {"id": f"eq.{user['id']}"},
            {"last_login_at": datetime.now(timezone.utc).isoformat()},
        )
    except Exception:
        pass

    token = _create_token(str(user["id"]), phone)
    return _ok(
        {
            "token": token,
            "user": {"id": user["id"], "name": user["name"], "phone": user["phone"]},
        }
    )


@app.post("/auth/reset-password")
async def auth_reset_password(body: ResetPasswordBody, request: Request) -> dict[str, Any]:
    """
    No-verification password reset. Resets the password for the given phone
    number directly if an account exists. (Use with care — for production-grade
    security you'd want OTP verification.)
    """
    await _ip_rate_limit(request, "reset", RESET_RATE_LIMIT)

    phone = _normalize_phone(body.phone)
    if not PHONE_RE.match(phone):
        raise HTTPException(status_code=400, detail="Invalid phone number.")

    rows, _ = await db.select(
        table=USERS_TABLE,
        columns="id,phone",
        filters={"phone": f"eq.{phone}"},
        limit=1,
    )
    if not rows:
        raise HTTPException(
            status_code=404,
            detail="No account found with this phone number.",
        )

    user = rows[0]
    pw_hash = _hash_password(body.new_password)
    await db.update(
        USERS_TABLE,
        {"id": f"eq.{user['id']}"},
        {"password_hash": pw_hash},
    )
    return _ok({"reset": True})


@app.get("/auth/me")
async def auth_me(current: dict[str, Any] = Depends(get_current_user)) -> dict[str, Any]:
    rows, _ = await db.select(
        table=USERS_TABLE,
        columns="id,name,phone,created_at,last_login_at",
        filters={"id": f"eq.{current['id']}"},
        limit=1,
    )
    if not rows:
        raise HTTPException(status_code=404, detail="User not found.")
    user = rows[0]

    wl_rows, wl_total = await db.select(
        table=WISHLIST_TABLE,
        columns="id",
        filters={"user_id": f"eq.{current['id']}"},
        limit=1,
    )
    al_rows, al_total = await db.select(
        table=ALERTS_TABLE,
        columns="id",
        filters={"user_id": f"eq.{current['id']}", "active": "eq.true"},
        limit=1,
    )

    return _ok(
        {
            "id": user["id"],
            "name": user["name"],
            "phone": user["phone"],
            "createdAt": user.get("created_at"),
            "stats": {
                "wishlist": wl_total,
                "alerts": al_total,
                "saved": 0,
            },
        }
    )


# ---------------------------------------------------------------------------
# Wishlist & alerts
# ---------------------------------------------------------------------------
@app.get("/me/wishlist")
async def my_wishlist(current: dict[str, Any] = Depends(get_current_user)) -> dict[str, Any]:
    rows, _ = await db.select(
        table=WISHLIST_TABLE,
        columns="product_id,created_at",
        filters={"user_id": f"eq.{current['id']}"},
        order="created_at.desc.nullslast",
        limit=200,
    )
    if not rows:
        return _ok([])

    ids = list({r["product_id"] for r in rows if r.get("product_id")})
    if not ids:
        return _ok([])

    or_clause = ",".join(f"normalized_product_id.eq.{pid}" for pid in ids)
    products, _ = await db.select(
        table=PRODUCTS_TABLE,
        columns=SAFE_COLUMNS,
        filters={"or": f"({or_clause})"},
        limit=len(ids),
    )
    transformed = [await _transform_product(p, include_extras=True) for p in products]
    return _ok(transformed)


@app.post("/me/wishlist")
async def add_wishlist(
    body: WishlistBody, current: dict[str, Any] = Depends(get_current_user)
) -> dict[str, Any]:
    if not re.match(r"^[A-Za-z0-9_\-:.]{1,200}$", body.product_id):
        raise HTTPException(status_code=400, detail="Invalid product id.")
    try:
        await db.insert(
            WISHLIST_TABLE,
            {"user_id": current["id"], "product_id": body.product_id},
        )
    except HTTPException as e:
        if e.status_code != 409:
            raise
    return _ok({"added": True})


@app.delete("/me/wishlist/{product_id}")
async def remove_wishlist(
    product_id: str, current: dict[str, Any] = Depends(get_current_user)
) -> dict[str, Any]:
    if not re.match(r"^[A-Za-z0-9_\-:.]{1,200}$", product_id):
        raise HTTPException(status_code=400, detail="Invalid product id.")
    await db.delete(
        WISHLIST_TABLE,
        {"user_id": f"eq.{current['id']}", "product_id": f"eq.{product_id}"},
    )
    return _ok({"removed": True})


@app.get("/me/alerts")
async def my_alerts(current: dict[str, Any] = Depends(get_current_user)) -> dict[str, Any]:
    rows, _ = await db.select(
        table=ALERTS_TABLE,
        columns="id,product_id,target_price,active,created_at",
        filters={"user_id": f"eq.{current['id']}"},
        order="created_at.desc.nullslast",
        limit=200,
    )
    return _ok(rows)


@app.post("/me/alerts")
async def add_alert(
    body: AlertBody, current: dict[str, Any] = Depends(get_current_user)
) -> dict[str, Any]:
    if not re.match(r"^[A-Za-z0-9_\-:.]{1,200}$", body.product_id):
        raise HTTPException(status_code=400, detail="Invalid product id.")
    row = await db.insert(
        ALERTS_TABLE,
        {
            "user_id": current["id"],
            "product_id": body.product_id,
            "target_price": body.target_price,
            "active": True,
        },
    )
    return _ok(row)


@app.delete("/me/alerts/{alert_id}")
async def remove_alert(
    alert_id: str, current: dict[str, Any] = Depends(get_current_user)
) -> dict[str, Any]:
    if not re.match(r"^[A-Za-z0-9\-]{1,80}$", alert_id):
        raise HTTPException(status_code=400, detail="Invalid alert id.")
    await db.delete(
        ALERTS_TABLE,
        {"id": f"eq.{alert_id}", "user_id": f"eq.{current['id']}"},
    )
    return _ok({"removed": True})


# ---------------------------------------------------------------------------
# Product routes
# ---------------------------------------------------------------------------
@app.get("/api/products")
async def list_products(
    page: int = Query(1, ge=1, le=10_000),
    limit: int = Query(20, ge=1, le=100),
    category: Optional[str] = Query(None, max_length=100),
    sub_category: Optional[str] = Query(None, max_length=100),
    brand: Optional[str] = Query(None, max_length=100),
    sort: str = Query("popular", pattern="^(popular|price_asc|price_desc|newest|rating)$"),
) -> dict[str, Any]:
    cache_key = f"products:{page}:{limit}:{category}:{sub_category}:{brand}:{sort}"
    cached = await cache.get(cache_key)
    if cached:
        return cached

    filters: dict[str, str] = {}
    if category:
        filters["main_category"] = f"eq.{category}"
    if sub_category:
        filters["sub_category"] = f"eq.{sub_category}"
    if brand:
        filters["brand"] = f"eq.{brand}"

    order_map = {
        "popular": "reviews.desc.nullslast",
        "price_asc": "lowest_price.asc.nullslast",
        "price_desc": "lowest_price.desc.nullslast",
        "newest": "created_at.desc.nullslast",
        "rating": "rating.desc.nullslast",
    }

    offset = (page - 1) * limit
    rows, total = await db.select(
        columns=SAFE_COLUMNS,
        filters=filters or None,
        order=order_map[sort],
        limit=limit,
        offset=offset,
    )

    products = [await _transform_product(r, include_extras=True) for r in rows]
    response = _ok(
        products,
        pagination={
            "page": page,
            "limit": limit,
            "total": total,
            "pages": (total + limit - 1) // limit if limit else 1,
        },
    )
    await cache.set(cache_key, response)
    return response


@app.get("/api/product/{normalized_product_id}")
async def get_product(normalized_product_id: str) -> dict[str, Any]:
    if not re.match(r"^[A-Za-z0-9_\-:.]{1,200}$", normalized_product_id):
        raise HTTPException(status_code=400, detail="Invalid product id.")

    cache_key = f"product:{normalized_product_id}"
    cached = await cache.get(cache_key)
    if cached:
        return cached

    rows, _ = await db.select(
        columns=SAFE_COLUMNS,
        filters={"normalized_product_id": f"eq.{normalized_product_id}"},
        limit=1,
    )
    if not rows:
        raise HTTPException(status_code=404, detail="Product not found.")

    product = await _transform_product(rows[0], include_extras=True)
    response = _ok(product)
    await cache.set(cache_key, response, ttl=CACHE_TTL_SECONDS)
    return response


@app.get("/api/search")
async def search(
    q: str = Query(..., min_length=1, max_length=120),
    page: int = Query(1, ge=1, le=200),
    limit: int = Query(20, ge=1, le=50),
    suggest: bool = Query(False),
) -> dict[str, Any]:
    query = q.strip().lower()
    _search_analytics[query] += 1

    cache_key = f"search:{query}:{page}:{limit}:{int(suggest)}"
    cached = await cache.get(cache_key)
    if cached:
        return cached

    safe_q = re.sub(r"[,()*\\]", " ", query).strip()
    if not safe_q:
        response = _ok(
            [],
            pagination={"page": 1, "limit": limit, "total": 0, "pages": 0},
            query=query,
        )
        await cache.set(cache_key, response, ttl=SEARCH_CACHE_TTL)
        return response

    pattern = f"*{safe_q}*"
    or_clause = (
        f"title.ilike.{pattern},"
        f"brand.ilike.{pattern},"
        f"main_category.ilike.{pattern},"
        f"sub_category.ilike.{pattern},"
        f"product_type.ilike.{pattern}"
    )

    candidates, _ = await db.select(
        columns=SAFE_COLUMNS,
        filters={"or": f"({or_clause})"},
        limit=300,
    )

    q_tokens = _tokenize(query)
    scored = [(r, _score_product(r, query, q_tokens)) for r in candidates]
    scored.sort(key=lambda x: x[1], reverse=True)
    scored = [s for s in scored if s[1] > 0.5]

    if suggest:
        suggestions: list[str] = []
        seen: set[str] = set()
        for row, _ in scored[:10]:
            title = (row.get("title") or "").strip()
            if title and title.lower() not in seen:
                suggestions.append(title)
                seen.add(title.lower())
        response = _ok(suggestions)
        await cache.set(cache_key, response, ttl=SEARCH_CACHE_TTL)
        return response

    total = len(scored)
    start = (page - 1) * limit
    paged_rows = [r for r, _ in scored[start : start + limit]]
    products = [await _transform_product(r, include_extras=True) for r in paged_rows]

    response = _ok(
        products,
        pagination={
            "page": page,
            "limit": limit,
            "total": total,
            "pages": (total + limit - 1) // limit if limit else 1,
        },
        query=query,
    )
    await cache.set(cache_key, response, ttl=SEARCH_CACHE_TTL)
    return response


@app.get("/api/categories")
async def categories() -> dict[str, Any]:
    cache_key = "categories:all"
    cached = await cache.get(cache_key)
    if cached:
        return cached

    rows, _ = await db.select(columns="main_category,sub_category", limit=10_000)
    tree: dict[str, set[str]] = defaultdict(set)
    for r in rows:
        main = (r.get("main_category") or "").strip()
        sub = (r.get("sub_category") or "").strip()
        if not main:
            continue
        if sub:
            tree[main].add(sub)
        else:
            tree[main]

    data = [
        {"category": main, "subcategories": sorted(subs)}
        for main, subs in sorted(tree.items())
    ]
    response = _ok(data)
    await cache.set(cache_key, response, ttl=600)
    return response


@app.get("/api/brands")
async def brands() -> dict[str, Any]:
    cache_key = "brands:all"
    cached = await cache.get(cache_key)
    if cached:
        return cached

    rows, _ = await db.select(columns="brand", limit=10_000)
    counts: dict[str, int] = defaultdict(int)
    for r in rows:
        b = (r.get("brand") or "").strip()
        if b:
            counts[b] += 1
    data = [
        {"brand": b, "count": c}
        for b, c in sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    ]
    response = _ok(data)
    await cache.set(cache_key, response, ttl=600)
    return response


@app.get("/api/deals")
async def deals(
    limit: int = Query(20, ge=1, le=100),
    min_discount: float = Query(10.0, ge=0.0, le=99.0),
) -> dict[str, Any]:
    cache_key = f"deals:{limit}:{min_discount}"
    cached = await cache.get(cache_key)
    if cached:
        return cached

    rows, _ = await db.select(
        columns=SAFE_COLUMNS,
        order="lowest_price.asc.nullslast",
        limit=400,
    )

    deals_list: list[tuple[float, dict[str, Any]]] = []
    for r in rows:
        low = _to_float(r.get("lowest_price"))
        high = _to_float(r.get("highest_price"))
        if low <= 0 or high <= 0 or high <= low:
            continue
        discount = round((high - low) / high * 100, 2)
        if discount >= min_discount:
            deals_list.append((discount, r))

    deals_list.sort(key=lambda x: x[0], reverse=True)
    deals_list = deals_list[:limit]

    products = []
    for discount, r in deals_list:
        p = await _transform_product(r, include_extras=True)
        p["discountPercent"] = discount
        products.append(p)

    response = _ok(products)
    await cache.set(cache_key, response, ttl=CACHE_TTL_SECONDS)
    return response


@app.get("/api/trending")
async def trending(limit: int = Query(20, ge=1, le=100)) -> dict[str, Any]:
    cache_key = f"trending:{limit}"
    cached = await cache.get(cache_key)
    if cached:
        return cached

    rows, _ = await db.select(
        columns=SAFE_COLUMNS,
        order="reviews.desc.nullslast",
        limit=limit * 2,
    )
    ranked = sorted(
        rows,
        key=lambda r: _to_int(r.get("reviews")) * max(_to_float(r.get("rating")), 1.0),
        reverse=True,
    )[:limit]
    products = [await _transform_product(r, include_extras=True) for r in ranked]
    response = _ok(products)
    await cache.set(cache_key, response, ttl=CACHE_TTL_SECONDS)
    return response


@app.get("/api/related/{normalized_product_id}")
async def related(
    normalized_product_id: str,
    limit: int = Query(10, ge=1, le=30),
) -> dict[str, Any]:
    if not re.match(r"^[A-Za-z0-9_\-:.]{1,200}$", normalized_product_id):
        raise HTTPException(status_code=400, detail="Invalid product id.")

    cache_key = f"related:{normalized_product_id}:{limit}"
    cached = await cache.get(cache_key)
    if cached:
        return cached

    base_rows, _ = await db.select(
        columns=SAFE_COLUMNS,
        filters={"normalized_product_id": f"eq.{normalized_product_id}"},
        limit=1,
    )
    if not base_rows:
        raise HTTPException(status_code=404, detail="Product not found.")
    base = base_rows[0]

    filters: dict[str, str] = {}
    if base.get("product_type"):
        filters["product_type"] = f"eq.{base['product_type']}"
    elif base.get("sub_category"):
        filters["sub_category"] = f"eq.{base['sub_category']}"
    elif base.get("main_category"):
        filters["main_category"] = f"eq.{base['main_category']}"

    rows, _ = await db.select(
        columns=SAFE_COLUMNS,
        filters=filters or None,
        order="reviews.desc.nullslast",
        limit=limit * 3,
    )

    base_price = _to_float(base.get("lowest_price"))
    related_items: list[tuple[float, dict[str, Any]]] = []
    for r in rows:
        if r.get("normalized_product_id") == normalized_product_id:
            continue
        score = 0.0
        if base.get("brand") and r.get("brand") == base.get("brand"):
            score += 2.0
        if base_price > 0:
            p = _to_float(r.get("lowest_price"))
            if p > 0:
                score += max(0.0, 2.0 - abs(p - base_price) / max(base_price, 1.0))
        score += min(_to_int(r.get("reviews")) / 1000.0, 2.0)
        related_items.append((score, r))

    related_items.sort(key=lambda x: x[0], reverse=True)
    products = [
        await _transform_product(r, include_extras=True) for _, r in related_items[:limit]
    ]
    response = _ok(products)
    await cache.set(cache_key, response, ttl=CACHE_TTL_SECONDS)
    return response


@app.get("/api/price-history/{normalized_product_id}")
async def price_history(normalized_product_id: str) -> dict[str, Any]:
    if not re.match(r"^[A-Za-z0-9_\-:.]{1,200}$", normalized_product_id):
        raise HTTPException(status_code=400, detail="Invalid product id.")

    cache_key = f"history:{normalized_product_id}"
    cached = await cache.get(cache_key)
    if cached:
        return cached

    rows, _ = await db.select(
        columns="normalized_product_id,price_history,lowest_price,highest_price,average_price",
        filters={"normalized_product_id": f"eq.{normalized_product_id}"},
        limit=1,
    )
    if not rows:
        raise HTTPException(status_code=404, detail="Product not found.")

    row = rows[0]
    history = _normalize_price_history(row.get("price_history"))
    response = _ok(
        {
            "history": history,
            "lowestPrice": round(_to_float(row.get("lowest_price")), 2),
            "highestPrice": round(_to_float(row.get("highest_price")), 2),
            "averagePrice": round(_to_float(row.get("average_price")), 2),
        }
    )
    await cache.set(cache_key, response, ttl=CACHE_TTL_SECONDS)
    return response


@app.get("/go/{offer_id}")
async def affiliate_redirect(offer_id: str, request: Request) -> RedirectResponse:
    if not offer_id.startswith("offer_") or len(offer_id) > 64:
        raise HTTPException(status_code=400, detail="Invalid offer.")

    async with _offer_lock:
        entry = _offer_registry.get(offer_id)

    if not entry:
        raise HTTPException(status_code=404, detail="Offer expired. Please refresh.")

    _click_analytics[offer_id] += 1
    _click_log.append(
        {
            "offer_id": offer_id,
            "store": entry.get("store"),
            "product_id": entry.get("product_id"),
            "ip": (request.client.host if request.client else None),
            "ua": request.headers.get("user-agent", "")[:200],
            "ts": time.time(),
        }
    )
    if len(_click_log) > _MAX_CLICK_LOG:
        del _click_log[: len(_click_log) - _MAX_CLICK_LOG]

    target = entry["url"]
    return RedirectResponse(url=target, status_code=302)


# ---------------------------------------------------------------------------
# Admin
# ---------------------------------------------------------------------------
@app.get("/admin/stats", dependencies=[Depends(require_admin)])
async def admin_stats() -> dict[str, Any]:
    rows, total = await db.select(columns="normalized_product_id", limit=1)
    return _ok(
        {
            "total_products": total,
            "cache_entries": cache.size(),
            "offers_registered": len(_offer_registry),
            "tracked_searches": len(_search_analytics),
            "tracked_clicks": len(_click_analytics),
        }
    )


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=PORT,
        log_level="info",
        proxy_headers=True,
        forwarded_allow_ips="*",
    )
