"""Cloud Client Factory and Provider Resolver for RoboSahaya Studio.

Gracefully handles configuration, secrets, and fallback between Supabase and Offline mode.
"""
from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple

from .base import BaseCatalogueProvider, BaseStorageProvider
from .offline_provider import OfflineCatalogueProvider, OfflineStorageProvider
from .supabase_provider import SupabaseCatalogueProvider, SupabaseStorageProvider

_cached_catalogue: Optional[BaseCatalogueProvider] = None
_cached_storage: Optional[BaseStorageProvider] = None


def get_cloud_catalogue() -> BaseCatalogueProvider:
    """Get active catalogue provider (Supabase if configured, otherwise Offline fallback)."""
    global _cached_catalogue
    if _cached_catalogue is not None:
        return _cached_catalogue

    url = os.environ.get("SUPABASE_URL", "").strip()
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "").strip()

    if url and key:
        prov = SupabaseCatalogueProvider(base_url=url, service_key=key)
        if prov.is_connected():
            _cached_catalogue = prov
            return _cached_catalogue

    _cached_catalogue = OfflineCatalogueProvider()
    return _cached_catalogue


def get_cloud_storage() -> BaseStorageProvider:
    """Get active storage provider (Supabase if configured, otherwise Offline fallback)."""
    global _cached_storage
    if _cached_storage is not None:
        return _cached_storage

    url = os.environ.get("SUPABASE_URL", "").strip()
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "").strip()
    bucket = os.environ.get("SUPABASE_ROBOT_BUCKET", "robot-model-packages").strip()

    if url and key:
        prov = SupabaseStorageProvider(base_url=url, service_key=key, bucket_name=bucket)
        if prov.is_connected():
            _cached_storage = prov
            return _cached_storage

    _cached_storage = OfflineStorageProvider()
    return _cached_storage


def get_cloud_status() -> Dict[str, Any]:
    """Return diagnostic cloud connection status without leaking credentials."""
    url = os.environ.get("SUPABASE_URL", "").strip()
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "").strip()
    bucket = os.environ.get("SUPABASE_ROBOT_BUCKET", "robot-model-packages").strip()

    cat = get_cloud_catalogue()
    connected = cat.is_connected() and isinstance(cat, SupabaseCatalogueProvider)

    # Sanitize URL for display
    sanitized_url = ""
    if url:
        from urllib.parse import urlparse
        parsed = urlparse(url)
        sanitized_url = f"{parsed.scheme}://{parsed.netloc}"

    return {
        "connected": connected,
        "mode": "supabase" if connected else "offline",
        "provider_type": cat.__class__.__name__,
        "endpoint": sanitized_url if connected else "local",
        "storage_bucket": bucket if connected else "local_mock",
        "has_credentials": bool(url and key),
        "status_badge": "🟢 Supabase Cloud Active" if connected else "🟡 Offline Mode (Local Cache Only)",
        "diagnostics": (
            "Connected to live Supabase Postgres and Storage bucket."
            if connected else
            "SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY not configured. Running in local/offline catalogue mode."
        )
    }


def reset_cloud_clients() -> None:
    """Reset cached singletons for testing or credential reload."""
    global _cached_catalogue, _cached_storage
    _cached_catalogue = None
    _cached_storage = None
