"""评测平台的 FastAPI 后端；通过 HTTP 和可选的只读 SQL 访问 Akasha。"""

from .settings import Settings, load_settings

__all__ = ["Settings", "load_settings"]
