"""Cloud Robot Model Library subsystem for RoboSahaya Studio."""
from .base import (
    BaseCatalogueProvider,
    BaseStorageProvider,
    CloudAssetRecord,
    CloudModelRecord,
    CloudVersionRecord,
    RuntimeCompatibilityRecord,
    VerificationRunRecord,
)
from .client import get_cloud_catalogue, get_cloud_status, get_cloud_storage, reset_cloud_clients
from .offline_provider import OfflineCatalogueProvider, OfflineStorageProvider
from .supabase_provider import SupabaseCatalogueProvider, SupabaseStorageProvider

__all__ = [
    "BaseCatalogueProvider",
    "BaseStorageProvider",
    "CloudAssetRecord",
    "CloudModelRecord",
    "CloudVersionRecord",
    "RuntimeCompatibilityRecord",
    "VerificationRunRecord",
    "get_cloud_catalogue",
    "get_cloud_storage",
    "get_cloud_status",
    "reset_cloud_clients",
    "OfflineCatalogueProvider",
    "OfflineStorageProvider",
    "SupabaseCatalogueProvider",
    "SupabaseStorageProvider",
]
