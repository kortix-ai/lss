"""Generate a realistic test corpus for search quality evaluation.

The corpus simulates a real-world project with files across multiple domains:
auth, database, deployment, monitoring, API, security, testing, and docs.

Each file has carefully crafted content so we can assign precise relevance
judgments for evaluation queries.
"""

from pathlib import Path


# ── File definitions ─────────────────────────────────────────────────────────
# Each entry: (relative_path, content)
# Content is written to be realistic and searchable.

CORPUS_FILES = {
    # ── Authentication ────────────────────────────────────────────────────
    "src/auth/jwt_handler.py": """\
\"\"\"JWT token creation and verification using RS256.

This module handles JSON Web Token operations including:
- Creating access tokens with configurable expiration (default 15 minutes)
- Creating refresh tokens (7 day expiry)
- Verifying tokens and extracting claims
- RSA key pair management

Uses PyJWT with RS256 algorithm for asymmetric signing.
Private key signs tokens, public key verifies them.
\"\"\"

import jwt
from datetime import datetime, timedelta
from cryptography.hazmat.primitives import serialization

ALGORITHM = "RS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 15
REFRESH_TOKEN_EXPIRE_DAYS = 7

def create_access_token(user_id: str, roles: list[str], private_key: bytes) -> str:
    payload = {
        "sub": user_id,
        "roles": roles,
        "exp": datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
        "iat": datetime.utcnow(),
        "type": "access"
    }
    return jwt.encode(payload, private_key, algorithm=ALGORITHM)

def create_refresh_token(user_id: str, private_key: bytes) -> str:
    payload = {
        "sub": user_id,
        "exp": datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS),
        "type": "refresh"
    }
    return jwt.encode(payload, private_key, algorithm=ALGORITHM)

def verify_token(token: str, public_key: bytes) -> dict:
    return jwt.decode(token, public_key, algorithms=[ALGORITHM])
""",

    "src/auth/oauth2_provider.py": """\
\"\"\"OAuth2 provider supporting authorization code flow and PKCE.

Implements OAuth2 with the following grant types:
- Authorization Code (with PKCE for public clients)
- Client Credentials (for service-to-service)

Supports Google and GitHub as external identity providers.
Stores authorization codes in Redis with 10-minute expiry.
\"\"\"

from dataclasses import dataclass

@dataclass
class OAuth2Config:
    client_id: str
    client_secret: str
    redirect_uri: str
    scopes: list[str]

GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
GITHUB_AUTH_URL = "https://github.com/login/oauth/authorize"

def generate_pkce_challenge(code_verifier: str) -> str:
    \"\"\"Generate S256 code challenge from verifier for PKCE flow.\"\"\"
    import hashlib, base64
    digest = hashlib.sha256(code_verifier.encode()).digest()
    return base64.urlsafe_b64encode(digest).rstrip(b"=").decode()

def exchange_authorization_code(code: str, config: OAuth2Config) -> dict:
    \"\"\"Exchange auth code for access and refresh tokens.\"\"\"
    pass  # Implementation calls token endpoint
""",

    "src/auth/middleware.py": """\
\"\"\"Authentication middleware for FastAPI.

Extracts Bearer tokens from the Authorization header,
verifies them using the JWT handler, and injects the
authenticated user into the request state.

Supports role-based access control (RBAC) with decorators:
- @require_auth: any authenticated user
- @require_role("admin"): specific role required
- @require_any_role(["editor", "admin"]): any of the listed roles

Rate limiting is applied per-user: 100 requests per minute for
standard users, 1000 for admin users.
\"\"\"

from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware

class AuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        token = request.headers.get("Authorization", "").replace("Bearer ", "")
        if token:
            try:
                claims = verify_token(token)
                request.state.user = claims
            except Exception:
                raise HTTPException(status_code=401, detail="Invalid token")
        return await call_next(request)

def require_role(role: str):
    \"\"\"Decorator that enforces RBAC role requirement.\"\"\"
    def decorator(func):
        async def wrapper(request: Request, *args, **kwargs):
            user = getattr(request.state, "user", None)
            if not user or role not in user.get("roles", []):
                raise HTTPException(status_code=403, detail="Insufficient permissions")
            return await func(request, *args, **kwargs)
        return wrapper
    return decorator
""",

    "src/auth/password_hash.py": """\
\"\"\"Password hashing using bcrypt with configurable work factor.

Uses bcrypt for password hashing with a default cost factor of 12.
Includes password strength validation:
- Minimum 8 characters
- At least one uppercase, one lowercase, one digit, one special character
- Not in the top 10,000 common passwords list

Also supports Argon2id for new installations (memory-hard hashing).
\"\"\"

import bcrypt

BCRYPT_ROUNDS = 12

def hash_password(plain: str) -> str:
    return bcrypt.hashpw(plain.encode(), bcrypt.gensalt(rounds=BCRYPT_ROUNDS)).decode()

def verify_password(plain: str, hashed: str) -> bool:
    return bcrypt.checkpw(plain.encode(), hashed.encode())

def validate_password_strength(password: str) -> tuple[bool, list[str]]:
    errors = []
    if len(password) < 8:
        errors.append("Password must be at least 8 characters")
    if not any(c.isupper() for c in password):
        errors.append("Must contain uppercase letter")
    if not any(c.isdigit() for c in password):
        errors.append("Must contain a digit")
    return (len(errors) == 0, errors)
""",

    # ── Database ──────────────────────────────────────────────────────────
    "src/db/connection_pool.py": """\
\"\"\"PostgreSQL connection pool using asyncpg.

Manages a pool of database connections with:
- Min 5, max 20 connections (configurable via DATABASE_POOL_MIN, DATABASE_POOL_MAX)
- Connection health checks every 30 seconds
- Automatic reconnection on failure
- SSL/TLS support for production environments
- Statement caching for prepared queries

Connection string format: postgresql://user:pass@host:5432/dbname
\"\"\"

import asyncpg

class ConnectionPool:
    def __init__(self, dsn: str, min_size: int = 5, max_size: int = 20):
        self.dsn = dsn
        self.min_size = min_size
        self.max_size = max_size
        self._pool = None

    async def initialize(self):
        self._pool = await asyncpg.create_pool(
            dsn=self.dsn,
            min_size=self.min_size,
            max_size=self.max_size,
            command_timeout=30,
        )

    async def acquire(self):
        return await self._pool.acquire()

    async def release(self, conn):
        await self._pool.release(conn)

    async def close(self):
        await self._pool.close()
""",

    "src/db/migrations.py": """\
\"\"\"Database migration framework using Alembic.

Manages schema versioning and migrations for PostgreSQL.
Each migration has an upgrade() and downgrade() function.

Migration workflow:
1. alembic revision --autogenerate -m "description"
2. Review generated migration in migrations/versions/
3. alembic upgrade head (apply all pending)
4. alembic downgrade -1 (rollback last migration)

Current schema version tracking is stored in the alembic_version table.
Supports both online (zero-downtime) and offline migrations.
\"\"\"

from alembic import op
import sqlalchemy as sa

def create_users_table():
    op.create_table(
        "users",
        sa.Column("id", sa.UUID(), primary_key=True),
        sa.Column("email", sa.String(255), unique=True, nullable=False),
        sa.Column("password_hash", sa.String(255), nullable=False),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(), onupdate=sa.func.now()),
    )
    op.create_index("idx_users_email", "users", ["email"])

def create_sessions_table():
    op.create_table(
        "sessions",
        sa.Column("id", sa.UUID(), primary_key=True),
        sa.Column("user_id", sa.UUID(), sa.ForeignKey("users.id")),
        sa.Column("refresh_token_hash", sa.String(255)),
        sa.Column("expires_at", sa.DateTime()),
        sa.Column("ip_address", sa.String(45)),
    )
""",

    "src/db/query_builder.py": """\
\"\"\"SQL query builder with parameterized queries to prevent SQL injection.

Provides a fluent interface for building SELECT, INSERT, UPDATE, DELETE queries.
All values are parameterized (never string-interpolated).

Example:
    query = Query.select("users").where("email", "=", email).limit(1).build()
    # Returns: ("SELECT * FROM users WHERE email = $1 LIMIT 1", [email])

Supports:
- JOINs (INNER, LEFT, RIGHT, FULL OUTER)
- Subqueries
- CTEs (Common Table Expressions)
- Window functions
- RETURNING clause for PostgreSQL
\"\"\"

class Query:
    def __init__(self, table: str):
        self.table = table
        self._conditions = []
        self._params = []
        self._limit = None
        self._offset = None
        self._order_by = None

    @classmethod
    def select(cls, table: str) -> "Query":
        q = cls(table)
        q._type = "SELECT"
        return q

    def where(self, column: str, op: str, value) -> "Query":
        self._params.append(value)
        self._conditions.append(f"{column} {op} ${len(self._params)}")
        return self

    def limit(self, n: int) -> "Query":
        self._limit = n
        return self

    def build(self) -> tuple[str, list]:
        sql = f"SELECT * FROM {self.table}"
        if self._conditions:
            sql += " WHERE " + " AND ".join(self._conditions)
        if self._limit:
            sql += f" LIMIT {self._limit}"
        return (sql, self._params)
""",

    "src/db/redis_cache.py": """\
\"\"\"Redis caching layer for frequently accessed data.

Implements a two-tier caching strategy:
1. L1: In-process LRU cache (1000 items, 60s TTL)
2. L2: Redis (configurable TTL, default 5 minutes)

Cache invalidation strategies:
- Time-based: automatic expiry via TTL
- Event-based: publish/subscribe for cache invalidation across instances
- Manual: explicit cache.delete(key) or cache.flush(pattern)

Serialization uses MessagePack for compact binary encoding.
Supports Redis Cluster mode for horizontal scaling.

Connection: REDIS_URL=redis://localhost:6379/0
\"\"\"

import redis
import msgpack

class RedisCache:
    def __init__(self, url: str = "redis://localhost:6379/0", default_ttl: int = 300):
        self.client = redis.from_url(url)
        self.default_ttl = default_ttl

    def get(self, key: str):
        data = self.client.get(key)
        if data:
            return msgpack.unpackb(data)
        return None

    def set(self, key: str, value, ttl: int = None):
        packed = msgpack.packb(value)
        self.client.setex(key, ttl or self.default_ttl, packed)

    def delete(self, key: str):
        self.client.delete(key)

    def flush_pattern(self, pattern: str):
        \"\"\"Delete all keys matching a glob pattern.\"\"\"
        keys = self.client.keys(pattern)
        if keys:
            self.client.delete(*keys)
""",

    # ── Deployment ────────────────────────────────────────────────────────
    "deploy/kubernetes/deployment.yaml": """\
# Kubernetes Deployment for the API service
# Runs 3 replicas behind a LoadBalancer service
# Uses rolling update strategy with maxSurge=1, maxUnavailable=0

apiVersion: apps/v1
kind: Deployment
metadata:
  name: api-server
  namespace: production
  labels:
    app: api-server
    version: v2.3.0
spec:
  replicas: 3
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
  selector:
    matchLabels:
      app: api-server
  template:
    metadata:
      labels:
        app: api-server
    spec:
      containers:
      - name: api
        image: registry.example.com/api-server:v2.3.0
        ports:
        - containerPort: 8080
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
        readinessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 10
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 15
          periodSeconds: 20
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: db-credentials
              key: url
        - name: REDIS_URL
          valueFrom:
            configMapKeyRef:
              name: app-config
              key: redis-url
""",

    "deploy/docker/Dockerfile": """\
# Multi-stage build for the Python API service
# Stage 1: Build dependencies in a full image
# Stage 2: Copy only necessary files to slim runtime image

FROM python:3.12-slim AS builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --target=/install -r requirements.txt

FROM python:3.12-slim AS runtime
WORKDIR /app
COPY --from=builder /install /usr/local/lib/python3.12/site-packages
COPY src/ ./src/
COPY config/ ./config/

# Non-root user for security
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s CMD curl -f http://localhost:8080/health || exit 1
CMD ["python", "-m", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8080"]
""",

    "deploy/terraform/main.tf": """\
# Terraform configuration for AWS EKS cluster
# Provisions: VPC, EKS cluster, node groups, IAM roles

terraform {
  required_version = ">= 1.5.0"
  required_providers {
    aws = { source = "hashicorp/aws", version = "~> 5.0" }
  }
  backend "s3" {
    bucket = "terraform-state-prod"
    key    = "eks/terraform.tfstate"
    region = "us-east-1"
  }
}

module "vpc" {
  source  = "terraform-aws-modules/vpc/aws"
  version = "5.0.0"
  name    = "production-vpc"
  cidr    = "10.0.0.0/16"
  azs     = ["us-east-1a", "us-east-1b", "us-east-1c"]
  private_subnets = ["10.0.1.0/24", "10.0.2.0/24", "10.0.3.0/24"]
  public_subnets  = ["10.0.101.0/24", "10.0.102.0/24", "10.0.103.0/24"]
  enable_nat_gateway = true
}

module "eks" {
  source          = "terraform-aws-modules/eks/aws"
  cluster_name    = "production"
  cluster_version = "1.28"
  vpc_id          = module.vpc.vpc_id
  subnet_ids      = module.vpc.private_subnets

  eks_managed_node_groups = {
    general = {
      instance_types = ["t3.medium"]
      min_size       = 2
      max_size       = 10
      desired_size   = 3
    }
  }
}
""",

    "deploy/ci/github_actions.yaml": """\
# CI/CD pipeline using GitHub Actions
# Runs on every push to main and pull requests
# Steps: lint, test, build Docker image, push to registry, deploy to staging

name: CI/CD Pipeline

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    services:
      postgres:
        image: postgres:16
        env:
          POSTGRES_PASSWORD: test
        ports: ["5432:5432"]
      redis:
        image: redis:7
        ports: ["6379:6379"]
    steps:
    - uses: actions/checkout@v4
    - uses: actions/setup-python@v5
      with:
        python-version: "3.12"
    - run: pip install -r requirements.txt -r requirements-dev.txt
    - run: pytest tests/ --cov=src --cov-report=xml
    - run: ruff check src/

  build-and-push:
    needs: test
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    - uses: docker/login-action@v3
      with:
        registry: ghcr.io
        username: ${{ github.actor }}
        password: ${{ secrets.GITHUB_TOKEN }}
    - uses: docker/build-push-action@v5
      with:
        push: true
        tags: ghcr.io/${{ github.repository }}:${{ github.sha }}

  deploy:
    needs: build-and-push
    runs-on: ubuntu-latest
    steps:
    - run: kubectl set image deployment/api-server api=ghcr.io/${{ github.repository }}:${{ github.sha }}
""",

    # ── Monitoring ────────────────────────────────────────────────────────
    "src/monitoring/metrics.py": """\
\"\"\"Prometheus metrics collection for the API service.

Exports the following metrics:
- http_requests_total: Counter of all HTTP requests (labels: method, path, status)
- http_request_duration_seconds: Histogram of request latency
- active_connections: Gauge of current WebSocket connections
- db_query_duration_seconds: Histogram of database query latency
- cache_hit_ratio: Gauge tracking Redis cache hit percentage

Metrics are exposed on /metrics endpoint for Prometheus scraping.
Grafana dashboards are configured in deploy/grafana/dashboards/.
\"\"\"

from prometheus_client import Counter, Histogram, Gauge

http_requests_total = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "path", "status"]
)

http_request_duration = Histogram(
    "http_request_duration_seconds",
    "HTTP request duration",
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 5.0]
)

active_connections = Gauge(
    "active_connections",
    "Number of active WebSocket connections"
)

db_query_duration = Histogram(
    "db_query_duration_seconds",
    "Database query duration",
    buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
)
""",

    "src/monitoring/alerting.py": """\
\"\"\"Alert rules and notification channels for production monitoring.

Alert conditions:
- High error rate: >5% of requests returning 5xx in 5 minutes
- Slow responses: p99 latency >2 seconds for 10 minutes
- Database connection exhaustion: pool usage >80% for 5 minutes
- Memory pressure: container memory >90% of limit
- Disk space: <10% free on data volumes

Notification channels:
- PagerDuty: critical alerts (SEV1, SEV2)
- Slack #alerts: all warning and critical alerts
- Email: daily digest of all alerts

Uses Alertmanager with grouping by service and severity.
Silencing rules prevent duplicate alerts during maintenance windows.
\"\"\"

ALERT_RULES = {
    "high_error_rate": {
        "expr": 'rate(http_requests_total{status=~"5.."}[5m]) / rate(http_requests_total[5m]) > 0.05',
        "severity": "critical",
        "for": "5m",
    },
    "slow_responses": {
        "expr": 'histogram_quantile(0.99, rate(http_request_duration_seconds_bucket[10m])) > 2',
        "severity": "warning",
        "for": "10m",
    },
    "db_pool_exhaustion": {
        "expr": 'db_pool_active / db_pool_max > 0.8',
        "severity": "warning",
        "for": "5m",
    },
}
""",

    "src/monitoring/logging_config.py": """\
\"\"\"Structured logging configuration using structlog.

All logs are emitted as JSON for easy parsing by log aggregation systems.
Supports shipping to:
- stdout (for Kubernetes log collection)
- Elasticsearch via Filebeat
- Datadog via the Datadog agent

Log levels:
- DEBUG: detailed diagnostic information
- INFO: general operational events
- WARNING: unexpected but handled situations
- ERROR: failures requiring attention
- CRITICAL: system-level failures

Correlation IDs are automatically injected via middleware to trace
requests across microservices. Format: X-Request-ID header.
\"\"\"

import structlog
import logging

def configure_logging(level: str = "INFO"):
    structlog.configure(
        processors=[
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
    )
    logging.basicConfig(level=getattr(logging, level))
""",

    # ── API ───────────────────────────────────────────────────────────────
    "src/api/routes.py": """\
\"\"\"API route definitions for the REST service.

Endpoints:
  POST   /api/v1/auth/login          - Authenticate and receive JWT tokens
  POST   /api/v1/auth/refresh         - Refresh access token
  GET    /api/v1/users/me             - Get current user profile
  PUT    /api/v1/users/me             - Update user profile
  GET    /api/v1/users/{id}           - Get user by ID (admin only)
  POST   /api/v1/items                - Create a new item
  GET    /api/v1/items                - List items with pagination
  GET    /api/v1/items/{id}           - Get item by ID
  DELETE /api/v1/items/{id}           - Delete item (owner or admin)
  GET    /api/v1/health               - Health check endpoint

All endpoints return JSON. Pagination uses cursor-based pagination
with ?cursor=<opaque_string>&limit=20 parameters.

CORS is configured to allow requests from the frontend domains.
Rate limiting: 100 requests/minute per API key.
\"\"\"

from fastapi import APIRouter, Depends, HTTPException

router = APIRouter(prefix="/api/v1")

@router.post("/auth/login")
async def login(email: str, password: str):
    pass

@router.get("/users/me")
async def get_current_user():
    pass

@router.get("/items")
async def list_items(cursor: str = None, limit: int = 20):
    pass

@router.get("/health")
async def health_check():
    return {"status": "healthy"}
""",

    "src/api/websocket.py": """\
\"\"\"WebSocket endpoint for real-time notifications and chat.

Supports the following message types:
- notification: server-push notifications to connected clients
- chat: bidirectional chat messages between users
- presence: online/offline status updates
- typing: typing indicator events

Connection lifecycle:
1. Client connects with JWT token as query parameter
2. Server validates token and registers connection
3. Messages are routed through Redis Pub/Sub for multi-instance support
4. On disconnect, server cleans up and broadcasts presence update

Heartbeat: ping/pong every 30 seconds to detect stale connections.
Max connections per user: 5 (across all instances).
\"\"\"

from fastapi import WebSocket
import json

class ConnectionManager:
    def __init__(self):
        self.active_connections: dict[str, list[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, user_id: str):
        await websocket.accept()
        if user_id not in self.active_connections:
            self.active_connections[user_id] = []
        self.active_connections[user_id].append(websocket)

    async def disconnect(self, websocket: WebSocket, user_id: str):
        self.active_connections[user_id].remove(websocket)

    async def broadcast(self, message: dict):
        for user_id, connections in self.active_connections.items():
            for conn in connections:
                await conn.send_json(message)
""",

    "src/api/rate_limiter.py": """\
\"\"\"Token bucket rate limiter using Redis for distributed rate limiting.

Algorithm: Token Bucket
- Each API key gets a bucket with max_tokens capacity
- Tokens refill at refill_rate per second
- Each request consumes 1 token (configurable per endpoint)

Default limits:
- Standard tier: 100 requests/minute, burst of 20
- Premium tier: 1000 requests/minute, burst of 100
- Admin tier: unlimited

Implementation uses Redis MULTI/EXEC for atomic token operations.
Sliding window fallback when Redis is unavailable.
Returns X-RateLimit-* headers in every response.
\"\"\"

import time
import redis

class TokenBucketLimiter:
    def __init__(self, redis_client: redis.Redis, max_tokens: int = 100,
                 refill_rate: float = 1.67):
        self.redis = redis_client
        self.max_tokens = max_tokens
        self.refill_rate = refill_rate

    def allow_request(self, key: str) -> tuple[bool, dict]:
        now = time.time()
        pipe = self.redis.pipeline()
        bucket_key = f"ratelimit:{key}"
        # Atomic check-and-decrement
        pipe.get(bucket_key)
        pipe.ttl(bucket_key)
        tokens, ttl = pipe.execute()

        if tokens is None:
            self.redis.setex(bucket_key, 60, self.max_tokens - 1)
            return True, {"remaining": self.max_tokens - 1}

        tokens = int(tokens)
        if tokens > 0:
            self.redis.decr(bucket_key)
            return True, {"remaining": tokens - 1}
        return False, {"remaining": 0, "retry_after": ttl}
""",

    # ── Security ──────────────────────────────────────────────────────────
    "src/security/encryption.py": """\
\"\"\"Data encryption utilities using AES-256-GCM.

Provides field-level encryption for sensitive data at rest:
- Credit card numbers
- Social security numbers
- Personal health information (PHI)

Uses AES-256-GCM for authenticated encryption with associated data (AEAD).
Key management: keys are stored in AWS KMS, fetched at startup, and
rotated every 90 days. Previous keys are retained for decryption of
existing data during rotation period.

Envelope encryption: a data encryption key (DEK) is generated per record,
encrypted with the master key (KEK) from KMS, and stored alongside the
ciphertext.
\"\"\"

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
import os

class FieldEncryptor:
    def __init__(self, master_key: bytes):
        assert len(master_key) == 32, "AES-256 requires 32-byte key"
        self.master_key = master_key

    def encrypt(self, plaintext: str) -> bytes:
        nonce = os.urandom(12)
        aes = AESGCM(self.master_key)
        ciphertext = aes.encrypt(nonce, plaintext.encode(), None)
        return nonce + ciphertext

    def decrypt(self, data: bytes) -> str:
        nonce, ciphertext = data[:12], data[12:]
        aes = AESGCM(self.master_key)
        return aes.decrypt(nonce, ciphertext, None).decode()
""",

    "src/security/csrf_protection.py": """\
\"\"\"CSRF protection using the double-submit cookie pattern.

How it works:
1. Server sets a CSRF token in a cookie (HttpOnly=False so JS can read it)
2. Client includes the token in X-CSRF-Token header with every mutating request
3. Server compares the cookie value with the header value
4. If they don't match, the request is rejected with 403 Forbidden

Token generation: HMAC-SHA256 of session ID + timestamp + random bytes.
Tokens rotate every 1 hour but previous tokens remain valid for 2 hours
to handle in-flight requests.

Exempt paths: /api/v1/auth/login, /api/v1/webhook/* (use webhook signatures instead)
\"\"\"

import hmac, hashlib, os, time

SECRET = os.environ.get("CSRF_SECRET", os.urandom(32).hex())

def generate_csrf_token(session_id: str) -> str:
    timestamp = str(int(time.time()))
    random_bytes = os.urandom(16).hex()
    message = f"{session_id}:{timestamp}:{random_bytes}"
    signature = hmac.new(SECRET.encode(), message.encode(), hashlib.sha256).hexdigest()
    return f"{message}:{signature}"

def validate_csrf_token(token: str, session_id: str) -> bool:
    parts = token.rsplit(":", 1)
    if len(parts) != 2:
        return False
    message, signature = parts
    expected = hmac.new(SECRET.encode(), message.encode(), hashlib.sha256).hexdigest()
    return hmac.compare_digest(signature, expected)
""",

    "src/security/audit_log.py": """\
\"\"\"Audit logging for security-relevant events.

Records all security events to an immutable audit trail:
- Authentication: login, logout, failed login attempts
- Authorization: permission changes, role assignments
- Data access: reads of sensitive fields (PII, PHI)
- Configuration: changes to security settings
- Admin actions: user creation, deletion, suspension

Each event includes:
- Timestamp (UTC, microsecond precision)
- Actor (user ID or system)
- Action (verb + resource)
- Target (affected resource ID)
- IP address and user agent
- Result (success/failure)
- Metadata (request details, changes made)

Storage: PostgreSQL with row-level security. Audit records are
append-only (INSERT only, no UPDATE or DELETE permissions).
Retention: 7 years for compliance (SOC2, HIPAA).
\"\"\"

from datetime import datetime, timezone
from dataclasses import dataclass

@dataclass
class AuditEvent:
    timestamp: datetime
    actor_id: str
    action: str
    target: str
    ip_address: str
    user_agent: str
    result: str  # "success" | "failure"
    metadata: dict

def log_event(event: AuditEvent):
    \"\"\"Persist audit event to the database.\"\"\"
    pass  # INSERT INTO audit_log ...

def query_events(actor_id: str = None, action: str = None,
                 start: datetime = None, end: datetime = None):
    \"\"\"Query audit events with filters.\"\"\"
    pass
""",

    # ── Testing ───────────────────────────────────────────────────────────
    "tests/test_auth.py": """\
\"\"\"Tests for the authentication module.

Covers:
- JWT token creation and verification
- Token expiration handling
- Invalid token rejection
- Role-based access control
- OAuth2 authorization code flow
- Password hashing and verification
- Brute force protection (account lockout after 5 failed attempts)

Uses pytest fixtures for test user creation and cleanup.
Mocks external OAuth providers (Google, GitHub).
\"\"\"

import pytest
from datetime import datetime, timedelta

def test_create_access_token():
    token = create_access_token("user123", ["admin"], PRIVATE_KEY)
    claims = verify_token(token, PUBLIC_KEY)
    assert claims["sub"] == "user123"
    assert "admin" in claims["roles"]

def test_expired_token_rejected():
    token = create_access_token("user123", [], PRIVATE_KEY)
    # Fast-forward time
    with freeze_time(datetime.utcnow() + timedelta(hours=1)):
        with pytest.raises(jwt.ExpiredSignatureError):
            verify_token(token, PUBLIC_KEY)

def test_invalid_signature_rejected():
    token = create_access_token("user123", [], PRIVATE_KEY)
    with pytest.raises(jwt.InvalidSignatureError):
        verify_token(token, WRONG_PUBLIC_KEY)

def test_password_hash_roundtrip():
    hashed = hash_password("SecureP@ss1")
    assert verify_password("SecureP@ss1", hashed)
    assert not verify_password("WrongPassword", hashed)

def test_rbac_admin_required():
    \"\"\"Non-admin user should get 403 when accessing admin endpoint.\"\"\"
    pass
""",

    "tests/test_api.py": """\
\"\"\"Integration tests for the REST API.

Uses pytest with httpx.AsyncClient for async endpoint testing.
Tests cover:
- Authentication flow (login, token refresh, logout)
- CRUD operations on items with pagination
- Input validation and error responses
- Rate limiting behavior
- CORS preflight handling
- Content negotiation (JSON only)

Each test runs in a transaction that is rolled back after completion
to ensure test isolation without needing to reset the database.
\"\"\"

import pytest
from httpx import AsyncClient

@pytest.fixture
async def client():
    from src.main import app
    async with AsyncClient(app=app, base_url="http://test") as c:
        yield c

@pytest.mark.asyncio
async def test_login_success(client):
    response = await client.post("/api/v1/auth/login", json={
        "email": "test@example.com",
        "password": "TestP@ss123"
    })
    assert response.status_code == 200
    assert "access_token" in response.json()

@pytest.mark.asyncio
async def test_list_items_pagination(client):
    response = await client.get("/api/v1/items?limit=10")
    assert response.status_code == 200
    data = response.json()
    assert "items" in data
    assert "next_cursor" in data

@pytest.mark.asyncio
async def test_rate_limiting(client):
    for _ in range(110):
        await client.get("/api/v1/items")
    response = await client.get("/api/v1/items")
    assert response.status_code == 429
""",

    # ── Documentation ─────────────────────────────────────────────────────
    "docs/architecture.md": """\
# System Architecture

## Overview

The system follows a microservices architecture deployed on Kubernetes.
Each service is independently deployable and communicates via REST APIs
and asynchronous message queues (RabbitMQ).

## Components

### API Gateway
- Handles authentication, rate limiting, and request routing
- Built with FastAPI (Python 3.12)
- Deployed as 3 replicas behind an AWS ALB

### User Service
- Manages user accounts, profiles, and authentication
- PostgreSQL database with read replicas
- Redis cache for session data

### Item Service
- CRUD operations for items with full-text search
- PostgreSQL with pg_trgm extension for fuzzy matching
- Elasticsearch for advanced search queries

### Notification Service
- Real-time notifications via WebSocket
- Email notifications via SendGrid
- Push notifications via Firebase Cloud Messaging

## Infrastructure

- **Cloud**: AWS (us-east-1)
- **Orchestration**: Kubernetes (EKS) v1.28
- **CI/CD**: GitHub Actions + ArgoCD for GitOps
- **Monitoring**: Prometheus + Grafana
- **Logging**: ELK Stack (Elasticsearch, Logstash, Kibana)
- **Secrets**: AWS Secrets Manager + Kubernetes Secrets
""",

    "docs/api_reference.md": """\
# API Reference

## Authentication

All API requests require a Bearer token in the Authorization header:
```
Authorization: Bearer <access_token>
```

### POST /api/v1/auth/login
Request body: `{"email": "string", "password": "string"}`
Response: `{"access_token": "string", "refresh_token": "string", "expires_in": 900}`

### POST /api/v1/auth/refresh
Request body: `{"refresh_token": "string"}`
Response: `{"access_token": "string", "expires_in": 900}`

## Users

### GET /api/v1/users/me
Returns the current authenticated user's profile.
Response: `{"id": "uuid", "email": "string", "name": "string", "roles": ["string"]}`

### PUT /api/v1/users/me
Update the current user's profile.
Request body: `{"name": "string", "avatar_url": "string"}`

## Items

### GET /api/v1/items
List items with cursor-based pagination.
Query parameters:
- `cursor`: opaque pagination cursor
- `limit`: number of items (default 20, max 100)
- `sort`: field to sort by (default "created_at")
- `order`: "asc" or "desc" (default "desc")

### POST /api/v1/items
Create a new item.
Request body: `{"title": "string", "description": "string", "tags": ["string"]}`

### DELETE /api/v1/items/{id}
Delete an item. Requires owner or admin role.
""",

    "docs/runbook.md": """\
# Operations Runbook

## Common Procedures

### Deploying a new version
1. Merge PR to main branch
2. GitHub Actions builds and pushes Docker image
3. ArgoCD detects new image tag and syncs
4. Monitor rollout: `kubectl rollout status deployment/api-server`
5. Check error rates in Grafana dashboard

### Rolling back a deployment
1. `kubectl rollout undo deployment/api-server`
2. Or in ArgoCD: click "Sync" and select previous revision
3. Verify health: `curl https://api.example.com/health`

### Scaling the service
- Horizontal: `kubectl scale deployment/api-server --replicas=5`
- HPA is configured: auto-scales 3-10 pods based on CPU (70% target)

## Incident Response

### High Error Rate (>5%)
1. Check Grafana dashboard for affected endpoints
2. Review logs: `kubectl logs -l app=api-server --tail=100`
3. Check database connectivity
4. If DB issue: failover to read replica
5. If code issue: rollback deployment

### Database Connection Exhaustion
1. Check pool metrics: `db_pool_active / db_pool_max`
2. Identify long-running queries: `SELECT * FROM pg_stat_activity WHERE state = 'active'`
3. Kill blocking queries if necessary
4. Increase pool max if persistent: update DATABASE_POOL_MAX env var
5. Consider adding read replicas for read-heavy workloads
""",

    "docs/onboarding.md": """\
# Developer Onboarding Guide

## Prerequisites
- Python 3.12+
- Docker and Docker Compose
- kubectl configured for the staging cluster
- Access to AWS console (request via IT ticket)

## Local Development Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/example/api-service.git
   cd api-service
   ```

2. Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt -r requirements-dev.txt
   ```

3. Start dependencies:
   ```bash
   docker compose up -d postgres redis
   ```

4. Run database migrations:
   ```bash
   alembic upgrade head
   ```

5. Start the dev server:
   ```bash
   uvicorn src.main:app --reload
   ```

## Running Tests
```bash
pytest tests/ -v
pytest tests/test_auth.py -k "test_login"  # specific test
```

## Code Style
- Black for formatting (line length 100)
- Ruff for linting
- Type hints required for all public functions
- Docstrings required for all modules and classes

## Architecture
See [architecture.md](./architecture.md) for the full system overview.
""",

    # ── Configuration ─────────────────────────────────────────────────────
    "config/settings.py": """\
\"\"\"Application settings loaded from environment variables.

Uses Pydantic BaseSettings for type-safe configuration with validation.

Environment variables:
- DATABASE_URL: PostgreSQL connection string
- REDIS_URL: Redis connection string
- JWT_PRIVATE_KEY_PATH: path to RSA private key file
- JWT_PUBLIC_KEY_PATH: path to RSA public key file
- CORS_ORIGINS: comma-separated allowed origins
- LOG_LEVEL: logging level (default: INFO)
- ENVIRONMENT: development | staging | production

Settings are loaded once at startup and cached.
Sensitive values are marked as SecretStr to prevent accidental logging.
\"\"\"

from pydantic_settings import BaseSettings
from pydantic import SecretStr

class Settings(BaseSettings):
    database_url: SecretStr
    redis_url: str = "redis://localhost:6379/0"
    jwt_private_key_path: str = "/etc/secrets/jwt-private.pem"
    jwt_public_key_path: str = "/etc/secrets/jwt-public.pem"
    cors_origins: list[str] = ["http://localhost:3000"]
    log_level: str = "INFO"
    environment: str = "development"
    debug: bool = False

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
""",

    "config/database.yaml": """\
# Database configuration for different environments

development:
  host: localhost
  port: 5432
  database: app_dev
  pool_min: 2
  pool_max: 5
  ssl: false
  statement_timeout: 30s

staging:
  host: staging-db.internal
  port: 5432
  database: app_staging
  pool_min: 5
  pool_max: 10
  ssl: true
  ssl_ca: /etc/ssl/rds-ca.pem
  statement_timeout: 15s

production:
  host: prod-db.internal
  port: 5432
  database: app_prod
  pool_min: 10
  pool_max: 50
  ssl: true
  ssl_ca: /etc/ssl/rds-ca.pem
  statement_timeout: 10s
  read_replicas:
    - host: prod-db-replica-1.internal
    - host: prod-db-replica-2.internal
""",

    # ── Additional files for diversity ────────────────────────────────────
    "src/search/full_text.py": """\
\"\"\"Full-text search using PostgreSQL ts_vector and ts_query.

Implements BM25-like scoring using PostgreSQL's built-in text search:
- ts_vector for document tokenization and normalization
- ts_query for query parsing with boolean operators
- ts_rank_cd for relevance scoring (cover density ranking)

Search features:
- Stemming via English dictionary
- Fuzzy matching with pg_trgm (trigram similarity)
- Phrase search with <-> (followed by) operator
- Weighted fields: title (A), tags (B), description (C), body (D)

Index: GIN index on the ts_vector column for fast lookups.
\"\"\"

def build_search_query(query: str, table: str = "items") -> str:
    return f\"\"\"
        SELECT id, ts_rank_cd(search_vector, plainto_tsquery('english', $1)) AS rank
        FROM {table}
        WHERE search_vector @@ plainto_tsquery('english', $1)
        ORDER BY rank DESC
        LIMIT 20
    \"\"\"
""",

    "src/tasks/background_worker.py": """\
\"\"\"Background task worker using Celery with Redis broker.

Handles long-running tasks that shouldn't block API responses:
- Email sending (via SendGrid API)
- PDF report generation
- Image resizing and thumbnail creation
- Data export to CSV/Excel
- Webhook delivery with retry logic
- Scheduled cleanup of expired sessions

Configuration:
- Broker: Redis (same instance as cache, different DB)
- Result backend: Redis with 24h TTL
- Concurrency: 4 workers per pod
- Max retries: 3 with exponential backoff (30s, 120s, 480s)
- Task timeout: 5 minutes (configurable per task)
\"\"\"

from celery import Celery

app = Celery("tasks", broker="redis://localhost:6379/1")

@app.task(bind=True, max_retries=3, default_retry_delay=30)
def send_email(self, to: str, subject: str, body: str):
    try:
        # Call SendGrid API
        pass
    except Exception as exc:
        self.retry(exc=exc, countdown=30 * (2 ** self.request.retries))

@app.task
def generate_report(user_id: str, report_type: str):
    \"\"\"Generate a PDF report and email it to the user.\"\"\"
    pass

@app.task
def cleanup_expired_sessions():
    \"\"\"Delete sessions that have passed their expiry date.\"\"\"
    pass
""",

    "src/utils/pagination.py": """\
\"\"\"Cursor-based pagination utilities.

Uses opaque cursors (base64-encoded JSON) instead of offset-based pagination.
This provides:
- Consistent results even when new items are inserted
- Better performance (no OFFSET scanning)
- No way to skip to arbitrary pages (by design)

Cursor format (internal): {"id": "uuid", "created_at": "2024-01-01T00:00:00Z"}
Cursor format (external): base64url encoded string

Usage:
    items, next_cursor = paginate(query, cursor=request.cursor, limit=20)
\"\"\"

import base64, json
from datetime import datetime

def encode_cursor(data: dict) -> str:
    return base64.urlsafe_b64encode(json.dumps(data).encode()).decode()

def decode_cursor(cursor: str) -> dict:
    return json.loads(base64.urlsafe_b64decode(cursor.encode()).decode())

def paginate(query, cursor: str = None, limit: int = 20):
    if cursor:
        data = decode_cursor(cursor)
        query = query.where("created_at", "<", data["created_at"])
    
    items = query.limit(limit + 1).fetch()
    has_more = len(items) > limit
    items = items[:limit]
    
    next_cursor = None
    if has_more and items:
        next_cursor = encode_cursor({
            "id": str(items[-1].id),
            "created_at": items[-1].created_at.isoformat()
        })
    
    return items, next_cursor
""",

    "scripts/seed_database.py": """\
\"\"\"Seed script to populate the database with test data.

Creates:
- 10 test users with various roles (admin, editor, viewer)
- 100 sample items with tags and descriptions
- 5 API keys for integration testing

Usage:
    python scripts/seed_database.py --env development
    python scripts/seed_database.py --env staging --users 50 --items 1000

Idempotent: running multiple times will not create duplicates.
Uses deterministic UUIDs based on email/name for reproducibility.
\"\"\"

import uuid
import hashlib

SEED_USERS = [
    {"email": "admin@example.com", "name": "Admin User", "roles": ["admin"]},
    {"email": "editor@example.com", "name": "Editor User", "roles": ["editor"]},
    {"email": "viewer@example.com", "name": "Viewer User", "roles": ["viewer"]},
]

def deterministic_uuid(seed: str) -> str:
    return str(uuid.UUID(hashlib.md5(seed.encode()).hexdigest()))

def seed_users(count: int = 10):
    for user in SEED_USERS:
        uid = deterministic_uuid(user["email"])
        # INSERT INTO users ...
        pass

def seed_items(count: int = 100):
    for i in range(count):
        # INSERT INTO items ...
        pass
""",
}


def generate_corpus(base_path: Path) -> dict[str, Path]:
    """Write all corpus files to base_path. Returns {relative_path: absolute_path}."""
    result = {}
    for rel_path, content in CORPUS_FILES.items():
        full_path = base_path / rel_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(content)
        result[rel_path] = full_path
    return result
