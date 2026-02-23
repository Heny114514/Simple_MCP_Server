"""
安全认证中间件

提供API密钥认证、IP白名单、速率限制等功能
"""

import time
from collections import defaultdict
from typing import Optional
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse
from modules.YA_Common.utils.logger import get_logger
from modules.YA_Common.utils.config import get_config

logger = get_logger("security")


class RateLimiter:
    """简单的内存速率限制器"""
    
    def __init__(self, requests_per_minute: int = 60, burst: int = 10):
        self.requests_per_minute = requests_per_minute
        self.burst = burst
        self.requests = defaultdict(list)
    
    def is_allowed(self, client_ip: str) -> bool:
        """检查请求是否允许"""
        now = time.time()
        minute_ago = now - 60
        
        self.requests[client_ip] = [
            t for t in self.requests[client_ip] if t > minute_ago
        ]
        
        if len(self.requests[client_ip]) >= self.requests_per_minute:
            return False
        
        self.requests[client_ip].append(now)
        return True


class SecurityMiddleware(BaseHTTPMiddleware):
    """安全认证中间件"""
    
    def __init__(self, app):
        super().__init__(app)
        self.enabled = get_config("security.enabled", False)
        self.api_key = get_config("security.api_key", "")
        self.ip_whitelist = get_config("security.ip_whitelist", ["127.0.0.1", "::1"])
        
        rate_limit_enabled = get_config("security.rate_limit.enabled", True)
        if rate_limit_enabled:
            requests_per_minute = get_config("security.rate_limit.requests_per_minute", 60)
            burst = get_config("security.rate_limit.burst", 10)
            self.rate_limiter = RateLimiter(requests_per_minute, burst)
        else:
            self.rate_limiter = None
        
        if self.enabled:
            logger.info(f"Security enabled - API Key: {self.api_key[:8]}..." if len(self.api_key) > 8 else "Security enabled")
    
    async def dispatch(self, request: Request, call_next):
        if not self.enabled:
            return await call_next(request)
        
        client_ip = request.client.host if request.client else "unknown"
        
        if self.ip_whitelist and client_ip not in self.ip_whitelist:
            logger.warning(f"Blocked IP: {client_ip}")
            return JSONResponse(
                status_code=403,
                content={"error": "Access denied", "message": "IP not allowed"}
            )
        
        if self.rate_limiter and not self.rate_limiter.is_allowed(client_ip):
            logger.warning(f"Rate limit exceeded for IP: {client_ip}")
            return JSONResponse(
                status_code=429,
                content={"error": "Rate limit exceeded", "message": "Too many requests"}
            )
        
        api_key = request.headers.get("X-API-Key") or request.query_params.get("api_key")
        if self.api_key and api_key != self.api_key:
            logger.warning(f"Invalid API key from {client_ip}")
            return JSONResponse(
                status_code=401,
                content={"error": "Unauthorized", "message": "Invalid API key"}
            )
        
        return await call_next(request)


class HealthCheckMiddleware(BaseHTTPMiddleware):
    """健康检查中间件"""
    
    def __init__(self, app):
        super().__init__(app)
        self.enabled = get_config("monitoring.health_check.enabled", True)
        self.endpoint = get_config("monitoring.health_check.endpoint", "/health")
    
    async def dispatch(self, request: Request, call_next):
        if self.enabled and request.url.path == self.endpoint:
            return JSONResponse({
                "status": "healthy",
                "service": get_config("server.name", "TimeSeriesMCPServer"),
                "version": get_config("server.version", "0.1.0"),
                "timestamp": time.time()
            })
        
        return await call_next(request)
